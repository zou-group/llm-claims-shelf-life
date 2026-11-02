# `shelflife`: package reference

> Part of the release described in [the top-level README](../README.md). This release registers the 49 configs used
> in the post. Commands that scan saved runs (`index-runs`, the `slow` tests)
> need the saved model outputs, which are not included; see [docs/DATA.md](docs/DATA.md). Input files larger
> than 5 MB are not in git either: download them as described in [DOWNLOADS.md](../DOWNLOADS.md).

A registry-driven pipeline for reproducing published LLM claims and re-running them on newer models. Every claim is
one registered config (`shelflife/experiments/cNNN_*.py`) with verbatim prompts from the original source, named
prompting-strategy variants, the original scorer, the paper's target numbers, and the earlier runs it must reproduce.

**Implementing a paper? Start with [the integration guide](docs/INTEGRATING.md).** It documents the five methods you
write, with executable single-turn and judge examples. You do not need to read the runner or ledger to add a claim.

## Setup

Install in editable mode from a full repository checkout: claim inputs and analysis files live beside `code/` and are required at runtime. A standalone wheel is not a complete distribution of the experiments.

```bash
cd code
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e '.[test]'   # add ,scoring for the original scorers' dependencies, ,dspy for the DSPy bridge
# exact versions used: requirements-lock.txt
```

API keys are read from exported `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, and `ANTHROPIC_API_KEY` at call time.
The core CLI does not automatically load `.env` files; see the [provider setup](../README.md) and root `.env.example`.
Request authentication headers are not stored in run records. Keep credentials out of prompts and custom model URLs.

## Common commands

```bash
.venv/bin/python -m shelflife.cli check C101                             # readiness checklist + the claim's tests
.venv/bin/python -m shelflife.cli list                                   # registered claims and default variants
.venv/bin/python -m shelflife.cli show C043 --variant faireval_vanilla   # prompts (sha256 + provenance), params, deviations
.venv/bin/python -m shelflife.cli plan C043 --models M206 M001 --limit 3 # offline execution with synthetic responses
.venv/bin/python -m shelflife.cli plan C043 --models panel20 --limit 3   # a whole panel: panel20 (the post's panel) or all (every registered model)
.venv/bin/python -m shelflife.cli plan C043 --models panel20 --limit 2 --responses-json scenario.json --out plan.json
.venv/bin/python -m shelflife.cli index-runs C043 --roots ../experiments/my-experiment/runs --out candidate-runs.json
.venv/bin/python -m shelflife.cli rescore C043 --run-dir ../experiments/my-experiment/runs/original --out-dir ../experiments/my-experiment/analysis/rescore-v2
.venv/bin/python -m shelflife.cli targets                                # recompute paper numbers offline
.venv/bin/python -m shelflife.cli run C043 --variant faireval_vanilla --models panel20 \
    --run-dir ../experiments/<exp>/runs/<name> --ledger ../experiments/<exp>/logs/budget.sqlite3 --ceiling 50
# Install scoring dependencies before the full suite:
# uv pip install --python .venv/bin/python -e .[test,scoring]
.venv/bin/python -m pytest -q                 # all offline tests
.venv/bin/python -m pytest -q --live -m live  # paid smoke tests; opt-in only
```

The `index-runs` and `rescore` examples require an existing run directory; `scenario.json` is your own map of named steps to synthetic replies.
For a new claim, use `new` with your claim record and author repository as shown in the [integration guide](docs/INTEGRATING.md).

## Layout

| Module | Role |
|---|---|
| `policies.py` | Request adaptation per provider route. Five policies ported verbatim from the experiment harnesses (`initial_openrouter_v1`, `initial_anthropic_v1`, `initial_openai_v1`, `initial_gpt6_v1`, `panel_v1`), plus `panel_v2`: `panel_v1`'s rules over the `MODELS` table. |
| `models.py` | Model tables: `MODELS` (the registered models; route rule Anthropic direct > OpenAI direct > OpenRouter), `PANEL20` (the post's 20 models) and the earlier table `PANEL` (kept verbatim for reproducing earlier runs); `--models` accepts `panel20` and `all`. |
| `ledger.py` | `LedgerClient`: atomic budget reservation (global + per-task ceilings), raw request/response record per call, retries only for explicit unbilled rejections, sticky circuit breaker, deterministic resume. `MockTransport` for offline tests. |
| `prompts.py` | `PromptTemplate` (verbatim text, substitution style, `Source` provenance, optional re-extraction from the original repo) |
| `program.py` | Prompting strategies: `SingleTurn`, or a `Program` subclass for multi-turn, judge pipelines and agent loops. Each call is a named step through the ledger. |
| `planning.py` | Execute programs offline with synthetic completions; inspect every visited call, including repeats and auxiliary models. Multi-turn scenario costs are **not** whole-run bounds. |
| `registry.py` | `Experiment`, `Variant` (faithful flag, deviations, fingerprint, `override`), `PaperTarget`, `PriorRun`, `@register`. |
| `results.py` | Optional validated `Headline` metadata: units, direction, denominator, missingness, reliability and scientific anchors. |
| `run_manifest.py` | Explicit JSON prior-run metadata, merged with legacy inline declarations; index saved configurations without editing claim code. |
| `runner.py` | Parallel (model × item × repeat) execution, configuration binding, code snapshot, per-model metrics. |
| `rescore.py` | Offline derived analyses from saved items/outputs, with the full intended row grid and provenance; never overwrites generation evidence. |
| `replay.py`, `stats.py`, `sourcecode.py` | Reading earlier runs; cluster bootstrap, exact bounds; loading the authors' own functions without running their scripts. |
| `dspy_bridge.py` | Optional DSPy front end (`VerbatimAdapter`, `LedgerEngine`, `DSPyProgram`). Same ledger and byte-identical payloads; DSPy cache disabled. |

## Adding a claim

Read [the integration guide](docs/INTEGRATING.md). Start with `shelflife.cli new`, iterate with `shelflife.cli check`,
and have the port reviewed independently against the paper. The core already provides:

- `shelflife.cli rescore` for immutable derived analyses of standardized runs, and `Experiment.rescore` for legacy formats;
- `replay.run_records` for merging response streams;
- `stats.rate_summary` for the standard rate metric;
- generic multi-turn replay;
- token usage on completions and `SingleTurn` outputs, without reopening raw files;
- a generated JSON run manifest, so configs do not carry large Python registration blocks;
- the runner preflight for judge models. Short version: register the paper's protocol as a `faithful` variant
with prompts taken verbatim from the original source; express prompt or strategy changes as `Variant.override(...)`
(automatically `faithful=False`); add `PaperTarget`s checked against released outputs where possible; list earlier runs
as `PriorRun`s.

## Changing prompts or strategies

```python
from shelflife import registry
exp = registry.get('C043')
base = exp.variant('faireval_vanilla')
alt = exp.add_variant(base.override(
    'vanilla_no_order_instruction',
    prompts={'rubric': base.prompts['rubric'].text.replace(', avoiding any potential bias and ensuring that the order in '
                                                       'which the responses were presented does not affect your judgment', '')},
    description='ablation: drop the order-invariance instruction'))
assert not alt.faithful   # recorded in configuration.json of every run
```

A run directory is bound to its variant fingerprint, items, models and package hash; changing any of them requires a new
run directory, so outputs from different protocols can never be mixed. A scorer-only change uses `rescore` with a new
output directory, not generation resume. First-request parity cannot prove a response-dependent program unchanged.
New runs use subject-scoped cache identity, so two models producing the same judge prompt still receive independent
judge calls. Resumes write new response streams rather than truncating earlier evidence. Historical snapshots retain
their original execution/cache semantics; this version does not automatically resume runs made by older code.

## Why DSPy is optional

DSPy's default adapters rewrite prompts into signature formats, and its cache returns the same completion for repeated
identical requests. Both break faithful reproduction (verbatim prompts, independent repeated samples). The bridge
keeps DSPy available for non-faithful prompting-strategy variants and for composing modules, while sending payloads
identical to the native path (`tests/test_dspy_bridge.py`). DSPy 3.4's custom-LM interface is in transition, so the core
does not depend on it.
