# Running experiments

Paid runs go through a budget ledger with a hard spending ceiling. Always check the plan first. Every claim reads the original paper's files: install them first
as described in [`DOWNLOADS.md`](../../DOWNLOADS.md).

```bash
cd code
set -a; source ../.env; set +a                  # or export the keys yourself

# 1. offline: payloads and a cost bound
.venv/bin/python -m shelflife.cli plan C043 --variant robust --models M206 --limit 3

# 2. a small paid run (stops before spending more than $5)
.venv/bin/python -m shelflife.cli run C043 --variant robust --models M206 --limit 3 \
    --run-dir ../runs/C043-robust-M206 --ledger ../runs/ledger.sqlite3 --ceiling 5

# 3. offline: rescore saved outputs with the current scorer, into a new directory
.venv/bin/python -m shelflife.cli rescore C043 --run-dir ../runs/C043-robust-M206 \
    --out-dir ../runs/C043-robust-M206-rescored
```

| Option | Meaning |
|---|---|
| `--variant` | The protocol: the paper-faithful variant, a smaller `minimal`-style variant, or `robust` (our modern-model variant). `show CLAIM` lists them. |
| `--models` | Model IDs from `models.py`, space-separated, or a panel name: `panel20` (the post's 20 models) or `all` (every registered model). |
| `--limit`, `--repeats` | Run fewer items or repeats than the variant's default. |
| `--ceiling`, `--task-budget` | Hard ceilings in USD for the whole ledger and for this task. |
| `--workers` | Parallel requests (default 8). |
| `--retry-errors` | Resubmit cached transient failures (429, 5xx, transport) when resuming a run. |

**The post's 20 models.** `--models panel20` is the post's panel (see [The model panel](MODELS.md)):

```bash
.venv/bin/python -m shelflife.cli plan C043 --variant robust --models panel20 --limit 1
```

For new models, use the `robust` variant, which the post used for its most recent models. Use the paper-faithful
variant to check a reproduction on the paper's own model. A few claims have extra requirements:

- **C010** (fabricated citations) checks every reference against Crossref and other bibliographic services, so
  scoring makes network calls. It caches the lookups under `code/.cache/`.
- **C089** (lost in conversation) runs model-written code inside macOS `sandbox-exec`, so it needs macOS.
- **C013** extracts its prompt from the paper's PDF with `pdftotext` (poppler).
- **C140** and **C143** read parquet files and need `pyarrow` (the `scoring` extra).
- Configs with a fixed judge or helper model (e.g. C141, C142, C166) call that model too. The runner checks before
  starting that every model, including helpers, has a route.

## Where outputs go

`run` writes one directory per model under `--run-dir`:

| File | Contents |
|---|---|
| `configuration.json` | Variant fingerprint (hash of prompts, parameters and strategy), items, models, package hash |
| `items.jsonl` | The exact test items |
| `responses.jsonl` | One row per call: output, status, error, cost and score |
| `raw/` | Raw request and response bodies (no credentials) |
| `metrics.json` | The headline value (`shelflife.value`, `ci`, `n`) and diagnostics |
| `completion.json` | Counts of completed, failed and errored calls, and the amount charged |
| `code_snapshot/` | The exact code that produced the run |

The ledger (`--ledger`, an SQLite file) records every reservation and charge, and lets an interrupted run resume
without paying twice. A run directory is bound to its configuration, so outputs from different protocols cannot
mix. `runs/` and `*.sqlite3` are ignored by git. To place a new model on a claim's curve, normalise its
`shelflife.value` with the claim's `null_value` and `paper_anchor` from [`claims/claims.csv`](../../claims/claims.csv).

## Regenerating our model outputs

We do not distribute our saved model outputs. Instead, [`claims/run_manifest.csv`](../../claims/run_manifest.csv) records,
for each of the 1,000 plotted claim × model values, the run that produced it: config, variant, model, request
policy, item limit, repeats, the sha256 of the exact items and requests, and what the run
cost. [`scripts/regenerate_runs.py`](../../scripts/regenerate_runs.py) turns those rows into `shelflife.cli` commands.
Install the input files first ([`DOWNLOADS.md`](../../DOWNLOADS.md)).

```bash
python scripts/regenerate_runs.py --claims C4                         # print the commands (plan = offline, free)
python scripts/regenerate_runs.py --claims C4 --execute plan          # run the plans: payloads and a cost bound
python scripts/regenerate_runs.py --claims C4 --execute run --ceiling 20    # paid; needs API keys
python scripts/regenerate_runs.py --execute run --ceiling 1500        # everything (we spent about $1,270)
```

Runs go to `runs/regenerated/`. Compare each model's `configuration.json` (`items_sha256`, `requests_sha256`) with
the manifest, then rescore and normalise as in [Where outputs go](#where-outputs-go). Two kinds of row will not
match byte for byte:

| Manifest rows | Why | What a rerun does |
|---|---|---|
| `same_protocol_today = False` (59 rows, 11 claims) | Generated with an earlier revision of the `robust` variant, then rescored | Uses today's `robust` protocol |
| `origin = earlier pipeline` (50 rows, 8 claims) | Generated by the pre-`shelflife` harness | Uses the variant whose requests reproduce those runs, with its default item count and repeats |

Models are sampled, often at temperature above zero, so new outputs will differ from ours within the reported
intervals. Multi-step programs (dialogues, judges, agents) can stop early in an offline `plan` because the plan uses
empty placeholder replies; this does not affect paid runs.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `error {"error":{"message":"Missing Authentication header","code":401}}` in `responses.jsonl` | The key for that model's route is not set in the shell that ran the command. Check the [route table](MODELS.md), export the key (or `source .env` with `set -a`), and rerun into a **new** `--run-dir`: a failed call is cached and is not retried on resume, because 401 is not a transient error. |
| `ModuleNotFoundError: No module named 'pydantic'` (or `pyarrow`, `eyecite`, `spacy`) | Reinstall into the venv: `uv pip install --python .venv/bin/python -e '.[test,scoring]'`. An old venv can miss dependencies added since it was built. |
| `FileNotFoundError` naming a file under `data/sources/` or `initial-reproduction/` | An input file is not installed. Run `python scripts/install_inputs.py` ([`DOWNLOADS.md`](../../DOWNLOADS.md)). |
| `preflight failed, nothing submitted` | A model or helper model has no route, or the variant rejects that model (for example a judge equal to the model under test). The message names it. |
| A route rejects a parameter (HTTP 400) | Parameter support for the newest direct-API models is assumed from model-family behaviour (`code/shelflife/data/model_capabilities.json`). Run a one-item smoke test before a large run. |
| C089 fails on Linux | It needs macOS `sandbox-exec` to run model-written code. |
