"""Read saved outputs of earlier runs (both earlier harnesses and this package) for re-scoring and parity tests."""
import json
from pathlib import Path

from .paths import PROJECT


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else PROJECT / p


def read_jsonl(path):
    with open(resolve(path)) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def legacy_records(run_dir, filename='responses.jsonl'):
    """Rows of a legacy run's responses.jsonl, deduplicated by cache_key (latest occurrence wins).

    ``filename`` selects another stream in the run directory (e.g. 'merged_responses.jsonl' or the latest
    'responses_continuation_NNN.jsonl'). Legacy rows carry: cache_key, case (the exact item sent), case_id, model (or record_id for panel runs),
    repeat, request, adaptations, content, status, finish_reason, score (legacy scorer output), charged_usd.
    """
    run_dir = resolve(run_dir)
    out = {}
    path = run_dir / filename
    if path.exists():
        for row in read_jsonl(path):
            key = row.get('cache_key') or (row.get('record_id') or row.get('model'), row.get('case_id'), row.get('repeat'))
            out[key] = row
    return list(out.values())


def raw_records(run_dir):
    """All raw/<key>.json request/response files of a run (including ones absent from responses.jsonl)."""
    run_dir = resolve(run_dir)
    return [json.loads(p.read_text()) for p in sorted((run_dir / 'raw').glob('*.json'))]


def model_key(row):
    """Model identity of a legacy row: panel record id if present, else the model string."""
    return row.get('record_id') or row.get('model')


def legacy_cases(run_dir):
    """The cases a legacy run sent: from code_snapshot/cases*.jsonl when unique, else from saved records."""
    run_dir = resolve(run_dir)
    snaps = sorted((run_dir / 'code_snapshot').glob('*.jsonl')) if (run_dir / 'code_snapshot').exists() else []
    if len(snaps) == 1:
        return read_jsonl(snaps[0])
    seen = {}
    for row in legacy_records(run_dir):
        seen.setdefault(row['case_id'], row['case'])
    if not seen and (run_dir / 'raw').exists():   # raw-only runs (no responses.jsonl)
        for rec in raw_records(run_dir):
            if 'case' in rec:
                seen.setdefault(rec['case']['id'], rec['case'])
    return list(seen.values())


# ---------------------------------------------------------------------------------------------------------------
# Multi-stream run reading (one rule for every claim instead of per-claim merges)

STATUS_RANK = {'complete': 4, 'ok': 4, 'truncated': 3, 'empty': 2, 'call_error': 1, 'error': 1,
               'not_submitted': 0}


def stream_files(run_dir):
    """Response streams of a run in merge order: responses.jsonl, then responses_*.jsonl (continuations, recovery)
    sorted by name, then merged*.jsonl. Later files win ties."""
    run_dir = resolve(run_dir)
    first = [run_dir / 'responses.jsonl'] if (run_dir / 'responses.jsonl').exists() else []
    rest = sorted(p for p in run_dir.glob('responses_*.jsonl'))
    merged = sorted(p for p in run_dir.glob('merged*.jsonl'))
    return first + rest + merged


def _call_key(row):
    return (row.get('root_model') or model_key(row), model_key(row),
            row.get('case_id') or row.get('item_id'), row.get('repeat', 0), row.get('step', 'main'))


def run_records(run_dir, streams=None):
    """Every saved output of a run, merged across streams.

    Rule: one record per (root subject when present, model, case id, repeat, step); a usable status beats an unusable one
    (complete/ok > truncated > empty > call_error/error > not_submitted); among equal status the later stream wins (continuations and
    recoveries supersede the original attempt). Falls back to raw/*.json when the run has no response stream.
    ``streams``: explicit list of file names to restrict to (e.g. ['merged_responses.jsonl']).
    """
    run_dir = resolve(run_dir)
    files = [run_dir / s for s in streams] if streams else stream_files(run_dir)
    rows = []
    for f in files:
        if f.exists():
            rows.extend(read_jsonl(f))
    if not rows and not streams:
        rows = [r for r in raw_records(run_dir) if 'case' in r or 'case_id' in r]
    best = {}
    for row in rows:
        key = _call_key(row)
        rank = STATUS_RANK.get(row.get('status'), -1)
        if key not in best or rank >= best[key][0]:
            best[key] = (rank, row)
    return [row for _, row in best.values()]


def is_shelflife_run(run_dir):
    """True for runs written by shelflife.runner (rows carry item_id and a call trace)."""
    path = resolve(run_dir) / 'configuration.json'
    return path.exists() and 'variant_fingerprint' in json.loads(path.read_text())


# ---------------------------------------------------------------------------------------------------------------
# Generic multi-turn replay for runs made by shelflife.runner

def replay_shelflife_run(exp, variant, run_dir, limit=None):
    """Run the current program offline against saved conversations, checking exact provider payloads.

    Conversation identity is (subject model, item, repeat); trace cache keys identify auxiliary calls. Identical
    prompts from another repeat, item, step, or subject are never interchangeable. Saved failures preserve their
    status. A trace's exact ``not_submitted`` step may fail without a raw record: its prompt cannot be verified,
    and is counted separately as ``unsent_tolerated``. No ledger, provider, or run file is modified.

    Returns ``conversations``, ``requests`` (selected raw records), ``matched``, ``unmatched`` (current payloads),
    ``unmatched_details`` (identities and reasons), ``unused`` (selected raw records not replayed), and
    ``unsent_tolerated``, and ``incomplete_tail`` (calls beyond a raw-only interrupted conversation's last record).
    Those tails have no saved evidence and do not establish parity. A clean check needs no unmatched calls or unused
    records; complete coverage additionally needs no unsent calls or incomplete tails. ``limit``
    bounds conversations, not calls; records of unselected conversations do not count as unused.
    """
    from .policies import get_policy, policy_for
    from .program import LM, CallFailed

    if limit is not None and (not isinstance(limit, int) or limit < 0):
        raise ValueError('limit must be a nonnegative integer or None')
    run_dir = resolve(run_dir)
    configuration = json.loads((run_dir / 'configuration.json').read_text())
    subjects = set(configuration['models'])
    raw = [r for r in raw_records(run_dir) if 'request' in r and 'step' in r]
    by_cache = {r['cache_key']: r for r in raw}

    def identity(row):
        return row['model'], row.get('item_id', row.get('case_id')), str(row.get('repeat', 0))

    def conversation_identity(row):
        _, item, repeat = identity(row)
        return row.get('root_model') or row['model'], item, repeat

    by_call, by_conversation = {}, {}
    for rec in raw:
        by_call.setdefault((*identity(rec), rec['step']), []).append(rec)
        by_conversation.setdefault(conversation_identity(rec), []).append(rec)
    rows = {}
    # Later successful continuations supersede old partial traces. Never let a later blocked resume erase one.
    rank = {'complete': 4, 'call_error': 2, 'not_submitted': 1}
    for path in stream_files(run_dir):
        for row in read_jsonl(path):
            if 'item_id' not in row or row.get('model') not in subjects:
                continue
            key = identity(row)
            old = rows.get(key)
            if old is None or rank.get(row.get('status'), 3) >= rank.get(old.get('status'), 3):
                rows[key] = row

    variant = exp.variant(variant) if isinstance(variant, str) else variant
    items = {x['id']: x for x in exp.items(variant)}
    program = exp.program(variant)
    traced = {entry.get('cache_key') for row in rows.values() for entry in row.get('trace') or []}
    convs = sorted(set(rows) | {conversation_identity(r) for r in raw
                               if conversation_identity(r)[0] in subjects and r['cache_key'] not in traced})
    if limit is not None:
        convs = convs[:limit]
    selected = set(by_cache) if limit is None else set()
    used, details, incomplete = set(), [], []
    matched = unsent = 0

    class ReplayClient:
        def __init__(self, conversation):
            self.conversation = conversation
            self.run_dir = run_dir  # compatibility for historical programs that read raw usage/logprobs
            self.trace = rows.get(conversation, {}).get('trace')
            self.position = 0
            self.known = {r['cache_key'] for r in by_conversation.get(conversation, [])}
            for rec in by_conversation.get(conversation, []):
                if rec['cache_key'] not in traced:
                    selected.add(rec['cache_key'])
            for entry in self.trace or []:
                if entry.get('cache_key') in by_cache:
                    selected.add(entry['cache_key'])

        def policy(self, model, record=None):
            name = (record or {}).get('policy') or configuration.get('policy')
            fixed = get_policy(name) if name else None
            return fixed if fixed is not None and fixed.accepts(model) else policy_for(model)

        def call(self, model, case, repeat=0, step='main'):
            nonlocal matched, unsent
            expected = (self.trace[self.position] if self.trace is not None and self.position < len(self.trace)
                        else None)
            self.position += 1
            rec = by_cache.get(expected.get('cache_key')) if expected else None
            candidates = by_call.get((model, case['id'], str(repeat), step), [])
            candidates = [r for r in candidates if r.get('root_model', self.conversation[0]) == self.conversation[0]]
            unresolved = (expected is not None and expected.get('status') == 'not_submitted'
                          and not expected.get('cache_key'))
            if (self.trace is None or unresolved) and len(candidates) == 1:
                rec = candidates[0]
            payload, _ = self.policy(model, rec).adapt(model, case)
            encoded = json.dumps(payload, sort_keys=True)

            def fail(reason):
                details.append({'conversation': self.conversation, 'model': model, 'step': step,
                                'reason': reason, 'request': payload})
                raise RuntimeError('replay mismatch: ' + reason)

            if expected is not None:
                expected_model = expected.get('model') or (rec or {}).get('model') or self.conversation[0]
                if (expected_model, expected['step']) != (model, step):
                    fail('call model or step order differs from the saved trace')
                if expected.get('status') == 'not_submitted' and rec is None:
                    unsent += 1
                    raise RuntimeError(expected.get('error') or 'step was not submitted in the saved run')
                if rec is None:
                    fail('raw record referenced by the saved trace is missing')
            elif self.trace is not None:
                fail('call extends beyond the saved trace')
            else:
                # Interrupted before the runner wrote its row: raw records still permit exact identity checking.
                if not candidates and self.known <= used:
                    incomplete.append({'conversation': self.conversation, 'model': model, 'step': step})
                    raise RuntimeError('no evidence beyond the saved interrupted conversation')
                candidates = [r for r in candidates if json.dumps(r['request'], sort_keys=True) == encoded]
                if len(candidates) != 1:
                    fail('no unique saved call with this model, item, repeat, step and payload')
                rec = candidates[0]
                selected.add(rec['cache_key'])
            if (*identity(rec), rec['step']) != (model, case['id'], str(repeat), step):
                fail('saved raw record has a different call identity')
            if rec.get('root_model', self.conversation[0]) != self.conversation[0]:
                fail('saved raw record belongs to another subject model')
            if json.dumps(rec['request'], sort_keys=True) != encoded:
                fail('provider payload differs from the saved request')
            used.add(rec['cache_key'])
            matched += 1
            # Preserve recorded status/content/usage, including uncertain failures and truncated completions.
            status = 'not_submitted' if unresolved else (rec.get('status') or 'error')
            return {**rec, 'status': status}

    for conversation in convs:
        model, case_id, repeat = conversation
        client = ReplayClient(conversation)
        if case_id not in items:
            details.append({'conversation': conversation, 'reason': 'saved item is missing from current items',
                            'request': {}})
            continue
        try:
            program.forward(items[case_id], LM(client, model, case_id, int(repeat),
                                               case_extra=getattr(program, 'case_extra', None)))
        except CallFailed:
            pass
        else:
            if client.trace is not None and client.position < len(client.trace):
                details.append({'conversation': conversation, 'reason': 'program omitted saved trace steps',
                                'request': {}, 'steps': [t['step'] for t in client.trace[client.position:]]})
    return {'conversations': len(convs), 'requests': len(selected), 'matched': matched,
            'unmatched': [d['request'] for d in details], 'unmatched_details': details,
            'unused': len(selected - used), 'unsent_tolerated': unsent, 'incomplete_tail': incomplete}
