"""Ledgered client: atomic budget reservation, raw request/response persistence, deterministic resume.

Merges the guarantees of the two experiment clients:
  * every request is written to ``<run_dir>/raw/<cache_key>.json`` before submission and atomically rewritten
    with the response; keys never appear in records;
  * a SQLite ledger reserves a conservative cost bound before submission (global ceiling plus optional per-task
    ceiling), so concurrent processes cannot overspend;
  * explicit, unbilled rejections (policy.retry_http, e.g. 429) are resubmitted with the identical payload;
    uncertain failures (timeouts, dropped connections) are never replayed automatically and keep their reserve;
  * three transport failures within 60 s open a sticky circuit breaker that requires an audited reset;
  * the cache key covers the run directory, case ID, repeat index, conversation step, model and the exact payload,
    so repeated samples are independent API calls and a changed prompt can never reuse an old response.
"""
import datetime
import hashlib
import json
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from .policies import get_policy, policy_for


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def transport_failure_kind(error):
    if isinstance(error, urllib.error.HTTPError):
        return None
    message = str(error).lower()
    if any(s in message for s in ('nodename nor servname', 'name or service not known', 'temporary failure in name resolution', 'getaddrinfo failed')):
        return 'dns'
    if any(s in message for s in ('no route to host', 'network is unreachable', 'network unreachable')):
        return 'unreachable'
    if any(s in message for s in ('connection reset', 'connection aborted', 'connection refused', 'remote end closed connection')):
        return 'connection'
    if isinstance(error, TimeoutError) or 'timed out' in message:
        return 'timeout'
    return None


class HTTPTransport:
    """POST JSON to the provider. Authorization headers come from the environment and are never stored."""

    def __init__(self, timeout=900):  # earlier pipeline: 300 -> 900 s; long hidden-reasoning responses exceeded 300 s
        self.timeout = timeout

    def __call__(self, url, headers, payload):
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.load(resp)


# Scientist's rule (2026-10-04): never send Anthropic models or OpenAI's GPT-5.6/6.x Sol/Luna models through OpenRouter;
# they must use the Anthropic and OpenAI APIs directly. Checked before any reservation or submission.
OPENROUTER_FORBIDDEN_PREFIXES = ('anthropic/', '~anthropic/', 'openai/gpt-5.6', 'openai/gpt-6')


def forbidden_route(url, payload):
    model = str(payload.get('model', ''))
    if 'openrouter.ai' in url and model.startswith(OPENROUTER_FORBIDDEN_PREFIXES):
        raise ValueError(f'{model} must not be routed through OpenRouter (use the provider API directly)')


class MockTransport:
    """Offline transport for tests and dry runs. ``responder(payload) -> str | dict`` returns content or a full
    provider response. Records every payload it receives."""

    def __init__(self, responder=None):
        self.responder = responder or (lambda payload: '')
        self.payloads = []
        self._lock = threading.Lock()

    def __call__(self, url, headers, payload):
        with self._lock:
            self.payloads.append(payload)
        out = self.responder(payload)
        if isinstance(out, dict):
            return out
        if 'anthropic.com' in url:
            return {'model': payload['model'], 'content': [{'type': 'text', 'text': out}], 'stop_reason': 'end_turn',
                    'usage': {'input_tokens': 0, 'output_tokens': 0}}
        return {'model': payload['model'], 'provider': 'mock',
                'choices': [{'message': {'content': out}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'cost': 0.0}}


class BudgetExceeded(RuntimeError):
    pass


class LedgerClient:
    """One client per run directory. Thread-safe; the ledger may be shared by several runs and processes.

    ``subject_v2`` scopes every call to its model under test, including auxiliary
    calls with otherwise identical prompts. ``legacy_v1`` keeps historical keys
    unchanged for direct clients and reproduction of old runs. New runners opt in
    to subject_v2 and bind the schema in their run configuration.
    """

    def __init__(self, run_dir, ledger, ceiling_usd, task=None, task_budget_usd=None, transport=None,
                 policy=None, max_attempts=8, backoff_cap=120, sleep=time.sleep, pacing=True, retry_errors=False,
                 breaker_threshold=3, cache_schema='legacy_v1'):
        if cache_schema not in {'legacy_v1', 'subject_v2'}:
            raise ValueError(f'unknown cache_schema {cache_schema!r}')
        self.cache_schema = cache_schema
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = Path(ledger)
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        if not 0 < float(ceiling_usd) < float('inf'):
            raise ValueError('positive finite ceiling required')
        self.ceiling = float(ceiling_usd)
        self.task = task
        self.task_budget = None if task_budget_usd is None else float(task_budget_usd)
        self.transport = transport or HTTPTransport()
        self.fixed_policy = None if policy is None else (get_policy(policy) if isinstance(policy, str) else policy)
        self.max_attempts, self.backoff_cap, self.sleep, self.pacing = max_attempts, backoff_cap, sleep, pacing
        self.retry_errors = retry_errors  # opt-in: resubmit transient errors (see RETRYABLE_ERRORS), archiving the old record
        self.breaker_threshold = breaker_threshold  # transport failures within 60 s that open the sticky circuit
        with self.connect() as c:
            c.execute('CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, task TEXT, model TEXT, policy TEXT, '
                      'status TEXT, reserve REAL, charged REAL, path TEXT, created TEXT)')
            c.execute('CREATE TABLE IF NOT EXISTS pacing (model TEXT PRIMARY KEY, next_slot REAL)')
            c.execute('CREATE TABLE IF NOT EXISTS transport_failures (key TEXT PRIMARY KEY, timestamp REAL, kind TEXT)')
            c.execute('CREATE TABLE IF NOT EXISTS runtime_guards (name TEXT PRIMARY KEY, opened TEXT, reason TEXT)')

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.ledger, timeout=120)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def policy(self, model):
        if self.fixed_policy is not None and self.fixed_policy.accepts(model):
            return self.fixed_policy
        return policy_for(model)

    # ---------------- identity ----------------
    def cache_key(self, model, case_id, repeat, step, payload, *, root_model=None):
        identity = {'run': str(self.run_dir.resolve()), 'case_id': case_id, 'repeat': repeat, 'step': step,
                    'model': model, 'payload': payload}
        if self.cache_schema == 'subject_v2':
            if not isinstance(root_model, str) or not root_model:
                raise ValueError('subject_v2 requires an explicit root_model (the model under test)')
            identity.update(cache_schema=self.cache_schema, root_model=root_model)
        return hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def plan(self, model, item):
        """Payload, adaptations and cost bound without contacting a provider (dry runs)."""
        policy = self.policy(model)
        payload, adaptations = policy.adapt(model, item)
        return payload, adaptations, policy.estimate(model, payload)

    # ---------------- budget ----------------
    def committed(self, c, task=None):
        q = 'SELECT COALESCE(SUM(COALESCE(charged,reserve)),0) FROM calls' + (' WHERE task=?' if task else '')
        return c.execute(q, (task,) if task else ()).fetchone()[0]

    def totals(self):
        with self.connect() as c:
            rows = c.execute('SELECT task,model,status,count(*),COALESCE(SUM(COALESCE(charged,reserve)),0) '
                             'FROM calls GROUP BY task,model,status').fetchall()
        return [{'task': r[0], 'model': r[1], 'status': r[2], 'calls': r[3], 'usd': r[4]} for r in rows]

    def _wait_for_slot(self, policy, model):
        rate = policy.pacing.get(model) if self.pacing else None
        if not rate:
            return
        with self.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            row = c.execute('SELECT next_slot FROM pacing WHERE model=?', (model,)).fetchone()
            slot = max(time.time(), row[0] if row else 0)
            c.execute('INSERT OR REPLACE INTO pacing VALUES (?,?)', (model, slot + 60 / rate))
        while slot > time.time():
            self.sleep(min(slot - time.time(), 10))

    # ---------------- call ----------------
    RETRYABLE_HTTP = frozenset({402, 408, 409, 429, 500, 502, 503, 504, 520, 522, 524, 529})  # 402: unbilled credit rejection

    def _retryable(self, record):
        """Transient failures only: retryable HTTP statuses, or transport exceptions with no HTTP status.
        Deterministic 4xx rejections (e.g. unsupported parameters) are never resubmitted."""
        status = record.get('http_status')
        if status is None:
            return record.get('error_type') not in (None, 'HTTPError')
        return int(status) in self.RETRYABLE_HTTP

    def call(self, model, item, repeat=0, step='main', *, root_model=None):
        """Submit one case. Returns the raw record (dict). Cached records are returned without a new call."""
        policy = self.policy(model)
        payload, adaptations = policy.adapt(model, item)
        if not isinstance(self.transport, MockTransport):  # offline replay/tests never reach a provider
            forbidden_route(policy.endpoint(model)[0], payload)
        key = self.cache_key(model, item['id'], repeat, step, payload, root_model=root_model)
        path = self.run_dir / 'raw' / f'{key}.json'
        path.parent.mkdir(exist_ok=True)
        reserve = policy.estimate(model, payload)
        with self.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            existing = c.execute('SELECT status,path FROM calls WHERE key=?', (key,)).fetchone()
            if existing:
                old = json.loads(Path(existing[1]).read_text()) if existing[0] != 'pending' and Path(existing[1]).exists() else None
                if old is None:
                    raise RuntimeError('Unresolved existing request; inspect ledger before retry: ' + key)
                if not (self.retry_errors and existing[0] == 'error' and self._retryable(old)):
                    return old
                # Archive the failed attempt under a new key (its charge stays in the ledger), then resubmit.
                n = 1 + c.execute('SELECT COUNT(*) FROM calls WHERE key LIKE ?', (key + '#err%',)).fetchone()[0]
                archived = Path(existing[1]).parent.parent / 'raw_retried' / f'{key}.err{n}.json'
                archived.parent.mkdir(exist_ok=True)
                Path(existing[1]).replace(archived)
                c.execute('UPDATE calls SET key=?, path=? WHERE key=?', (f'{key}#err{n}', str(archived), key))
            guard = c.execute("SELECT reason FROM runtime_guards WHERE name='transport'").fetchone()
            if guard:
                raise RuntimeError('Transport circuit open: ' + guard[0])
            total = self.committed(c)
            if total + reserve > self.ceiling:
                raise BudgetExceeded(f'Budget: committed {total:.4f} + request bound {reserve:.4f} > ceiling {self.ceiling}')
            if self.task_budget is not None:
                task_total = self.committed(c, self.task)
                if task_total + reserve > self.task_budget:
                    raise BudgetExceeded(f'Task budget {self.task}: committed {task_total:.4f} + {reserve:.4f} > {self.task_budget}')
            c.execute('INSERT INTO calls VALUES (?,?,?,?,?,?,?,?,?)',
                      (key, self.task, model, policy.name, 'pending', reserve, None, str(path), now()))
        record = {'cache_key': key, 'task': self.task, 'model': model, 'policy': policy.name, 'case': item,
                  'case_id': item['id'], 'repeat': repeat, 'step': step, 'request': payload,
                  'adaptations': adaptations, 'started': now(), 'reserve_usd': reserve, 'retries': []}
        if self.cache_schema == 'subject_v2':
            record.update(cache_schema=self.cache_schema, root_model=root_model)
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
        start = time.monotonic()
        charged, failure_kind = reserve, None
        url, headers = policy.endpoint(model)
        for attempt in range(self.max_attempts):
            try:
                self._wait_for_slot(policy, model)
                response = self.transport(url, headers, payload)
                record['response'] = response
                if response.get('error'):
                    raise RuntimeError(json.dumps(response['error']))
                summary = policy.summarize(model, response, reserve)
                charged = summary.pop('charged_usd')
                record.update(summary)
                finish = summary['finish_reason']
                record['status'] = 'truncated' if finish in {'length', 'max_tokens'} else ('ok' if summary['content'] else 'empty')
                break
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors='replace')
                if e.code in policy.retry_http and attempt < self.max_attempts - 1:
                    record['retries'].append({'http_status': e.code, 'at': now(), 'body': body[:500]})
                    self.sleep(min(self.backoff_cap, 5 * 2 ** attempt))
                    continue
                record.update(status='error', http_status=e.code, error_body=body, error_type='HTTPError')
                charged, record['cost_kind'] = policy.http_error_cost(model, e.code, reserve)
                break
            except Exception as e:  # uncertain billing: never auto-replayed
                record.update(status='error', error_type=type(e).__name__, error=str(e))
                failure_kind = transport_failure_kind(e)
                if failure_kind:
                    record['transport_failure_kind'] = failure_kind
                charged, record['cost_kind'] = reserve, 'conservative_bound_failed_request'
                break
        record['charged_usd'] = charged
        record['elapsed_seconds'] = time.monotonic() - start
        record['finished'] = now()
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False))
        tmp.replace(path)
        with self.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            c.execute('UPDATE calls SET status=?,charged=? WHERE key=?', (record['status'], charged, key))
            if failure_kind:
                timestamp = time.time()
                c.execute('INSERT OR IGNORE INTO transport_failures VALUES (?,?,?)', (key, timestamp, failure_kind))
                count = c.execute('SELECT COUNT(*) FROM transport_failures WHERE timestamp>=?', (timestamp - 60,)).fetchone()[0]
                if count >= self.breaker_threshold:
                    c.execute('INSERT OR IGNORE INTO runtime_guards VALUES (?,?,?)',
                              ('transport', now(), f'{count} transport failures in 60 seconds; sticky stop requires audited recovery'))
        return record
