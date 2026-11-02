# Implementing a paper in `shelflife`

Start here. You normally implement five methods in one module and tests in one file. The runner already
handles model routing, budgets, retries, saved calls, repeated samples, missing rows, and output artifacts. This guide
is the API you need for an ordinary port; core internals are linked only for less common cases.

## The shortest path to an implementation

Run commands from `code/` using `.venv/bin/python`:

```bash
.venv/bin/python -m shelflife.cli new C123 --slug paper_name --claim-json /path/to/claim.json --repo /path/to/author/repo
.venv/bin/python -m shelflife.cli check C123
```

`new` creates the draft module, claim tests, findings, and a small JSON run manifest. It reads the claim record given
with `--claim-json`, which also accepts a nested claim with
`statement` or `text`. No repository is cloned unless you request `--clone URL` or `--clone auto`. Drafts remain
outside the shared ready-claim suite; `check` includes their intentionally failing TODO tests.

Before coding, write down the paper version, exact source locations for prompts and data, treatment and control,
independent unit, numerator, denominator, and how refusals, parse failures, truncations, and unavailable calls count.
Then fill the five methods. These scientific choices account for most of the work and belong in the claim module
and its findings, not in a replacement runner.

| Method | Contract | Common pitfall |
|---|---|---|
| `build_variants()` | Return `{name: Variant(...)}`; parameters and prompt sources are explicit. | Calling a powered subset, substitute judge, or altered prompt paper-faithful without qualification. |
| `items(variant, limit=None)` | Return rows with unique stable `id`; apply a deterministic prefix only when `limit is not None`. | Losing pairs in a prefix, unstable random subsets, or treating `limit=0` as unlimited. |
| `program(variant)` | Return `SingleTurn`, or a `Program` with `forward(item, lm) -> dict`. | Reimplementing API calls or sending the judge to the model under test. |
| `score(item, output, variant)` | Pure, offline scoring of one available output into a JSON-compatible dictionary. | Quietly replacing the released scorer with a more forgiving parser. |
| `aggregate(rows, variant)` | Paper metric and diagnostics over all intended item/repeat rows for one model. | Dropping failed calls or using calls as independent observations when the unit is a base question. |

`Variant.override(...)` gives another named protocol; changed prompts, parameters, or strategy mark it non-faithful.
Document other scientific changes, including repeats and subsets, with explicit deviations. Use `PromptTemplate`
and `Source` for every prompt. Prefer `PromptTemplate.from_source(...)` or the authors' own prompt builder loaded
with `shelflife.sourcecode.load_functions`. Pin source snapshots under `shelflife/data/sources/` with `PROVENANCE.md`.

## A complete small example

This executable teaching example is deliberately unregistered and non-faithful: its two arithmetic items are
fixtures, not evidence about a paper. Real modules use `@register`, the actual claim ID, paper targets, and source
snapshots supplied by the scaffold. Paste the Python blocks into one scratch file or run them in the same session.

```python
from shelflife import stats
from shelflife.program import SingleTurn
from shelflife.prompts import PromptTemplate, Source
from shelflife.registry import Experiment, Variant
from shelflife.results import Headline

DEMO_SOURCE = Source(repo='.', path='code/docs/INTEGRATING.md', locator='small example')
DEMO_PROMPT = PromptTemplate('user', 'Answer with one integer: {question}', source=DEMO_SOURCE)


class ArithmeticExample(Experiment):
    claim_id = 'DEMO'  # intentionally not registered
    default_variant = 'demo'

    def build_variants(self):
        return {'demo': Variant('demo', 'Documentation fixture', prompts={'user': DEMO_PROMPT},
                                params={'temperature': 0, 'max_tokens': 32}, faithful=False,
                                deviations=('Invented data and prompt for this API example',))}

    def items(self, variant, limit=None):
        rows = [{'id': 'sum-1', 'question': '2 + 2', 'answer': '4'},
                {'id': 'sum-2', 'question': '3 + 3', 'answer': '6'}]
        return rows[:limit] if limit is not None else rows

    def program(self, variant):
        return SingleTurn(lambda item: [{'role': 'user', 'content': variant.prompts['user'].render(
            question=item['question'])}], variant.params)

    def score(self, item, output, variant):
        text = (output.get('content') or '').strip()
        # This example's specified rule is strict exact match. Verbose answers stay incorrect.
        return {'correct': text == item['answer'], 'unparsed': not text.isdecimal()}

    def aggregate(self, rows, variant):
        result = stats.rate_summary(rows, lambda r: r['score']['correct'])
        result['unparsed'] = sum(r['score']['unparsed'] for r in rows if r['score'] is not None)
        result['shelflife'] = Headline(
            metric='accuracy', value=result['rate'], n=result['scored'], direction=1, unit='fraction',
            denominator='Available item/repeat rows; unavailable calls excluded from point estimate',
            ci=result['ci95'] if result['rate'] is not None else None, missing=result['unscored'],
            missingness_note='Outer result n counts all intended rows; headline n counts scored rows; bounds include missing',
            headline_reliable=None,
        ).to_dict()
        return result


example = ArithmeticExample()
variant = example.variant()
assert example.items(variant, limit=0) == []
assert example.program(variant).preview(example.items(variant)[0]) == [
    {'role': 'user', 'content': 'Answer with one integer: 2 + 2'}]
```

The rows received by `aggregate` look like the following. Failed calls have `output=None` and `score=None`; they
are different from an available empty response, which the scorer receives normally. Do not filter the rows first.

```python
rows = [
    {'item': {'id': 'a'}, 'output': {'content': '4'}, 'score': {'correct': True, 'unparsed': False},
     'model': 'demo-model', 'repeat': 0, 'status': 'complete'},
    {'item': {'id': 'b'}, 'output': {'content': '5'}, 'score': {'correct': False, 'unparsed': False},
     'model': 'demo-model', 'repeat': 0, 'status': 'complete'},
    {'item': {'id': 'c'}, 'output': None, 'score': None,
     'model': 'demo-model', 'repeat': 0, 'status': 'call_error'},
]
result = example.aggregate(rows, variant)
assert (result['n'], result['scored'], result['unscored']) == (3, 2, 1)
assert result['shelflife']['value'] == 0.5
assert result['bounds_with_missing'] == [1 / 3, 2 / 3]
assert result['shelflife']['missing'] == 1

# Independent literal expectations, including output formats the released parser may reject.
item = {'id': 'sum-1', 'question': '2 + 2', 'answer': '4'}
for text, expected in [('4', True), (' 4\n', True), ('**4**', False),
                       ('After checking, my answer is 4.', False), ('I cannot answer.', False), ('', False)]:
    assert example.score(item, {'content': text}, variant)['correct'] is expected
```

The point estimate here deliberately uses a scored-row denominator. If the paper counts failed calls as wrong,
use `stats.rate_summary(rows, success=lambda r: bool((r.get('score') or {}).get('correct')),
usable=lambda r: True)` and separately report unavailable-call counts. On these fixtures its rate must be `1/3`.
Choose this from the paper, not from whichever denominator gives the more favorable result.

`Headline(...).to_dict()` is an optional, validated way to produce `aggregate()['shelflife']`. Specify units,
denominator and direction (`+1` means larger values support the claim, `-1` smaller). Anchors, confidence intervals,
and reliability can be `None` when unknown; use `value=None` for an unavailable estimate. A sensitivity metric can
be another `Headline(role='sensitivity', ...)` under a separate key. The helper validates structure and finite
numbers; it cannot decide scientific validity. Existing claim-specific aggregate dictionaries remain supported.

For paired conditions, either put the entire pair in one item and make both calls in its program, or give separate
condition rows a shared `base_id`. Aggregate matched pairs explicitly, count unmatched pairs, and bootstrap the
base item (`cluster=lambda r: r['item']['base_id']`). Repeated calls are not independent base items. Record whether
the primary metric is an unconditional rate, a rate conditional on initial correctness, a paired difference, or a
macro average. A `--limit` is a prefix of items, not a statistical sampling design.

## Multi-turn, judges, and response-dependent programs

Keep ordinary Python control flow. Every model call goes through the provided `lm`; each `(model, step)` must be
unique within the item/repeat. Declare fixed auxiliary models so preflight can validate them before spending.
New runs scope call identity to the root subject model as well as item, repeat, called model and step, so identical
judge prompts from different subjects still produce independent draws. Historical runs retain their saved semantics.
The following complete program extends the example with a challenge and a fixed judge. Its prompts are examples,
not a faithful implementation of any paper.

```python
from shelflife.program import Program
from shelflife.models import PANEL20
from shelflife.planning import plan_program


class ChallengeAndJudge(Program):
    name = 'challenge_and_judge'

    def __init__(self, variant, judge_model):
        self.variant = variant
        self.judge_model = judge_model
        self.auxiliary_models = (judge_model,)

    def preview(self, item):
        return [{'role': 'user', 'content': self.variant.prompts['user'].render(question=item['question'])}]

    def forward(self, item, lm):
        messages = self.preview(item)
        initial = lm(messages, self.variant.params, step='initial')
        messages = messages + [{'role': 'assistant', 'content': initial.content},
                               {'role': 'user', 'content': 'Check your answer. Reply with one integer.'}]
        revised = lm(messages, self.variant.params, step='revision')
        judge = lm.for_model(self.judge_model)
        verdict = judge([{'role': 'user', 'content':
            f'Expected: {item["answer"]}\nAnswer: {revised.content}\nReply CORRECT or WRONG.'}],
            {'temperature': 0, 'max_tokens': 16}, step='judge')
        return {'content': revised.content, 'initial': initial.content, 'verdict': verdict.content}


# For a paper use its pinned judge model, or an explicitly non-faithful substitute variant.
subject_key, judge_key = list(PANEL20)[:2]
program = ChallengeAndJudge(variant, judge_key)
plan = plan_program(program, example.items(variant, limit=1), [subject_key],
                    policy='panel_v2', responder=lambda request: {
                        'initial': '5', 'revision': '4', 'judge': 'CORRECT'}[request['step']])
assert plan['totals']['calls'] == 3
assert plan['totals']['completed_paths'] == 1
assert plan['calls'][-1]['model'] == judge_key
assert plan['totals']['whole_run_bound_usd'] is None  # no proof about all possible response paths
```

Change only `Experiment.program` to return this program; score the saved `initial`, `content`, and `verdict` fields
according to the paper. A failed call raises `CallFailed`; normally let the runner retain the failed row and partial
trace. Catch it only if the paper defines a recovery branch, then test that branch. Keep network and filesystem
side effects out of `forward` except model calls through `lm`: planning and replay execute the same Python code.
Load fixed retrieval corpora or precomputed features as items or local source artifacts.

`Completion` exposes `content`, `reasoning`, `status`, `finish_reason`, `usage`, and `raw_usage`. `raw_response` is a
detached copy of the provider response for less common fields such as logprobs; it is not duplicated in trace
summaries. Do not reopen raw files from a new program. Normalized usage
keys include `input_tokens`, `output_tokens`, `total_tokens`, `cached_input_tokens`, `cache_creation_input_tokens`,
and `reasoning_tokens`; unknown values are `None`, not zero. Persist any values that scoring needs in the returned
output dictionary. Token accounting differs by provider; explain which quantity the paper measures.

## Plan and validate before a paid run

```bash
.venv/bin/python -m shelflife.cli plan C123 --models panel20 --limit 3 --repeats 2 --out /tmp/C123-plan.json
.venv/bin/python -m shelflife.cli plan C123 --models panel20 --limit 3 --responses-json /path/to/scenario.json --out /tmp/C123-scenario.json
```

The scenario JSON maps step names to text or completion dictionaries, with an optional `"*"` default. For the
program above, a successful scenario is `{"initial": "5", "revision": "4", "judge": "CORRECT"}`; a failed judge
can use `"judge": {"status": "error", "error": "fixture timeout"}`. More detailed branching by model, item or
repeat uses the Python callable responder shown above.

Planning executes `forward` with synthetic responses and adapts every visited request, including auxiliary calls.
Inspect prompts, policies, adaptations, call counts, and scenario errors. A scenario uses chosen placeholder
responses; a response-dependent loop or branch may behave differently on real outputs. Its scenario cost is never
a whole-run upper bound for a multi-turn program. Test at least the important success, refusal, malformed-output,
and failure paths using a callable responder; bound loops explicitly in the claim protocol. Route adaptations
such as unsupported temperature removal, message layout changes, or output-cap changes affect faithfulness and
must be reviewed. An offline plan does not establish that an API route is live.
Historical programs that directly read a raw file may report incomplete scenario paths; they have not been silently
rewritten. For new programs use completion fields and supply required provider metadata via the responder's
`raw_response`. Missing metadata should fail visibly, not be invented by the planner.

Implement claim tests with independent fixtures: verbatim author prompts, known author scores, printed paper
examples, literal expected counts and paired denominators, and modern output formats. Never compare a scorer to
itself, weaken an assertion to pass, or silently drop unparsed text. A more permissive parser belongs in a separately
labeled diagnostic or documented scoring revision. Use `PaperTarget` to anchor paper numbers against released
outputs when possible; record truly unverifiable targets and investigated discrepancies explicitly.

Run `check C123`, have the port reviewed independently against the paper, resolve the findings, and set `maturity='ready'`. Integration and these examples make no paid calls. Before a paid run,
first smoke-test its chosen routes, then use the normal runner with an explicit ledger and ceiling.
`panel20` is this project's default; `all` is every registered model. Retain earlier model choices only for reproducing old
runs. Do not rewrite source snapshots or previous evidence.
`TECHNICAL CHECKS PASS` is not scientific approval; `--no-tests` reports only static checks and explicitly says tests
were not run.

## Saved runs without hand-written registration blocks

The scaffold's `run_manifest` points to a project-relative JSON file. Keep run metadata there:

```json
{
  "version": 1,
  "claim_id": "C123",
  "prior_runs": [],
  "excluded_runs": {}
}
```

For existing runs, build a candidate from saved configurations:

```bash
.venv/bin/python -m shelflife.cli index-runs C123 --roots ../experiments/my-experiment/runs --out /tmp/C123-runs.json
```

Review each variant mapping, description, known deviation and exclusion before attaching it. Referenced manifests
must exist. New snapshots use new filenames; do not silently rewrite a frozen run index. Inline legacy `PriorRun`
declarations still work and are merged with the manifest by `exp.all_prior_runs()` / `exp.all_excluded_runs()`.
The indexer flags missing raw files for review and never infers `no_requests=True`: missing or pruned archives
cannot prove that a request was never sent. Set that flag only with evidence that no calls were submitted.

For scorer or aggregate changes on a standardized saved run:

```bash
.venv/bin/python -m shelflife.cli rescore C123 --run-dir ../experiments/my-experiment/runs/original --out-dir ../experiments/my-experiment/analysis/rescore-v2
```

This offline operation uses saved items and outputs and writes a new derived analysis. It is for changing scoring,
not prompts, item selection, or model execution. Review the derived provenance and counts and preserve the original
results. `Experiment.rescore(...)` remains useful for legacy formats; use `record_item` / `record_output` hooks only
when legacy IDs or record shapes require them. Modern saved-run replay checks the actual request sequence,
including fixed auxiliary models.
Run rescoring after generation stops. Its hashes and package snapshot record the Python implementation and saved
input streams, not a full environment lock: separately preserve pinned author-source artifacts, dependencies and
precomputed resources. Scorers must remain pure/offline; the helper is not a sandbox for arbitrary Python code.

Resume remains conservatively bound to code and configuration. If a scorer edit changes the package fingerprint,
use offline `rescore` for existing outputs. A new generation protocol gets a new variant and run directory; never
bypass the resume check or merge results across protocols just to recover a partially finished job.
Resumes append a new response stream and preserve the original configuration, items and response streams, including
when execution is interrupted. This new runner does not automatically resume runs created by older code snapshots.

## Where to look when this guide is insufficient

| Need | Focused reference |
|---|---|
| Exact template extraction or loading selected author functions | [prompts.py](../shelflife/prompts.py), [sourcecode.py](../shelflife/sourcecode.py) |
| Paired/self-correction scientific example | [C024](../shelflife/experiments/c024_are_you_sure.py) and its claim tests |
| Judge pipeline scientific example | [C068](../shelflife/experiments/c068_abstentionbench.py) and its claim tests |
| Validated headline fields | [results.py](../shelflife/results.py) |
| Synthetic response scenarios | [planning.py](../shelflife/planning.py) |
| Historical output formats | [replay.py](../shelflife/replay.py) |

If the paper needs an unsupported shared capability, describe the missing capability and propose a small core
extension. Keep claim-specific scientific rules in the claim module. Porting another paper should not require
learning the ledger, transport adapter, retry engine, or runner internals.
