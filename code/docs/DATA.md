# Data: inputs, provenance and what is not included

## What the configs read

The 49 configs read the original papers' code, prompts and data at the relative paths they had in the research
workspace, so the paths written in configs and prompt `Source`s stay valid:

1. **Source snapshots** in `code/shelflife/data/sources/<CLAIM>_<name>/`: the files of the original release that
   the configs read, pinned to a GitHub commit, a Hugging Face revision, an arXiv e-print or a release archive.
2. **Original-repository files** in `experiments/initial-reproduction/claims/<ID>_<slug>/code/<repo>/`, pinned to
   the commit recorded in the config (`Experiment.source`), with the claim record `claim.json` next to them.
3. **Reproduction helpers** in `experiments/initial-reproduction/reproductions/<ID>/` and
   `code/shelflife/experiments/helpers/<ID>/`: scorer helpers, calibration sets and frozen item lists that the
   configs import.

The files in 1 and 2 are not stored in git. [`scripts/inputs.json`](../../scripts/inputs.json) lists all 1,428 of them
with path, size, sha256, origin and the claims that read them; `python scripts/install_inputs.py` downloads and
verifies them ([`DOWNLOADS.md`](../../DOWNLOADS.md)). `.gitignore` covers the installed paths. Files the project
prepared from the sources (prepared item lists, transcribed tables, extracted paper text, derived tables) are in
git, each described in the folder's `PROVENANCE.md`.

## Not included

| What | Size | Effect in this release |
|---|---|---|
| Saved model outputs (requests, responses, raw records, per-run metrics) of all panel and reproduction runs | ~14 GB | The collected per-model values are shipped (`results_long.csv`, `claims/panel_values.csv`). |
| C143 cached embeddings of model outputs (`text-embedding-3-small`) | ~240 MB | C143 rescoring would need new embedding calls |
| C010 Crossref/bibliographic lookup cache | ~200 MB | C010 rescoring repeats the web lookups |
| Upstream `.git` histories | — | `test_source_commit` skips; the commits are in the configs and `PROVENANCE.md` |

One variant re-judges our earlier outputs, so it cannot build its items without them: C041 `gpt4_judge_full`, the
judge step of the C041 reproduction.

## Licences

Third-party material keeps its original licence. The tables below list the upstream source of each snapshot and
whether it includes a licence file. Snapshots without a licence file (marked "none (reviewed)") were reviewed by the
authors for redistribution.

### Source snapshots

| Snapshot | Upstream | Licence file in snapshot |
|---|---|---|
| `C013_arxiv_v1` | https://arxiv.org/e-print/2412.01708v1 | none (reviewed) |
| `C026_elephant_v1` | https://github.com/myracheng/elephant | none (reviewed) |
| `C029_paper` | https://github.com/ApolloResearch/insider-trading | none (reviewed) |
| `C037_anthropics_evals` | https://github.com/anthropics/evals | LICENSE |
| `C043_fastchat` | https://github.com/lm-sys/FastChat | none (reviewed) |
| `C044_self_recognition` | https://github.com/ArjunPanickssery/self_recognition | none (reviewed) |
| `C047_Humans_LLMs_Judgement_Bias` | https://github.com/FreedomIntelligence/Humans_LLMs_Judgement_Bias | LICENSE |
| `C054_datasets` | https://s3.amazonaws.com/commensenseqa/dev_rand_split.jsonl | none (reviewed) |
| `C055_gsm8k` | https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl | none (reviewed) |
| `C055_overthinking` | https://github.com/galaxyChen/overthinking | none (reviewed) |
| `C059_self-ask` | https://github.com/ofirpress/self-ask | license.txt |
| `C067_dahl_dataverse_V4ON8H` | see PROVENANCE.md | none (reviewed) |
| `C067_reglab_legal_hallucinations_hf` | https://huggingface.co/datasets/reglab/legal_hallucinations | none (reviewed) |
| `C068_coconot` | see PROVENANCE.md | none (reviewed) |
| `C073_LLM-MCQ-Bias_data` | https://github.com/chujiezheng/LLM-MCQ-Bias_data | none (reviewed) |
| `C077_kotek2023` | https://arxiv.org/pdf/2308.14921 | none (reviewed) |
| `C078_dialect_prejudice` | https://github.com/valentinhofmann/dialect-prejudice | LICENSE |
| `C083_8values` | https://github.com/8values/8values.github.io | LICENSE |
| `C083_pct_scoring` | https://github.com/politicalcompass/politicalcompass.github.io | none (reviewed) |
| `C083_political_preferences` | https://zenodo.org/records/10553530 | none (reviewed) |
| `C089_lost_in_conversation` | https://github.com/microsoft/lost_in_conversation | LICENSE |
| `C093_RGB` | https://github.com/chen700564/RGB | LICENSE.txt |
| `C102_PromptInject` | https://github.com/agencyenterprise/PromptInject | LICENSE |
| `C104_PackageHallucination` | https://github.com/Spracks/PackageHallucination | none (reviewed) |
| `C104_pypi_first_upload` | https://pypi.org/pypi/<name | none (reviewed) |
| `C104_pypi_simple_index_2026-10-04` | https://pypi.org/simple/ | none (reviewed) |
| `C108_sharma2023_arxiv_v4` | https://arxiv.org/e-print/2310.13548 | none (reviewed) |
| `C109_persist_labels` | https://huggingface.co/datasets/AIM-Harvard/PERSIST | none (reviewed) |
| `C109_PERSIST` | https://huggingface.co/datasets/AIM-Harvard/PERSIST | none (reviewed) |
| `C120_InjecAgent` | https://github.com/uiuc-kang-lab/InjecAgent | LICENCE |
| `C129_MedLitSpin` | https://github.com/hyesunyun/MedLitSpin | LICENSE |
| `C137_borderlines` | https://github.com/manestay/borderlines | none (reviewed) |
| `C138_guardrails` | https://github.com/vli31/llm-guardrail-sensitivity | LICENSE |
| `C139_farm` | https://github.com/pillowsofwind/llms-believe-the-earth-is-flat | none (reviewed) |
| `C140_mask` | https://github.com/centerforaisafety/mask | LICENSE |
| `C141_reversal_curse` | https://github.com/lukasberglund/reversal_curse | none (reviewed) |
| `C142_sycophantic_ai_osf` | https://osf.io/smvw7/ (view-only link; see PROVENANCE.md) | none (reviewed) |
| `C143_hivemind_arxiv_v2` | https://arxiv.org/e-print/2510.22954v2 | none (reviewed) |
| `C143_infinity_chat_hf` | https://huggingface.co/collections/liweijiang/artificial-hivemind-6826e108da3260c02a1a2ec0 | none (reviewed) |
| `C157_aiai_bias_2407.12856` | https://arxiv.org/e-print/2407.12856 | none (reviewed) |
| `C161_confaide` | https://github.com/skywalker023/confaide | none (reviewed) |
| `C162_temporal_llms` | https://arxiv.org/html/2311.08398v2 | none (reviewed) |
| `C166_censorship` | https://arxiv.org/abs/2504.03803v1 | none (reviewed) |

### Repository clones

| Clone (`experiments/initial-reproduction/claims/`) | Licence file |
|---|---|
| `C014_llm-reviews-nondiscriminating/code/ReviewCritique` | LICENSE |
| `C023_random-number-bias/code/diffuse-distributions` | LICENSE |
| `C024_are-you-sure-sycophancy/code/sycophancy-eval` | none (reviewed) |
| `C026_social-sycophancy-elephant/code/elephant` | LICENSE |
| `C028_agentic-misalignment/code/agentic-misalignment` | LICENSE |
| `C041_xstest-overrefusal/code/xstest` | LICENSE |
| `C043_judge-position-bias/code/FairEval` | none (reviewed) |
| `C046_judge-length-bias/code/alpaca_eval` | LICENSE |
| `C057_embers-of-autoregression/code/embers-of-autoregression` | LICENSE |
| `C058_counterfactual-tasks/code/counterfactual-evaluation` | LICENSE |
| `C068_abstentionbench/code/AbstentionBench` | LICENSE |
| `C073_mcq-option-id-bias/code/LLM-MCQ-Bias` | none (reviewed) |
| `C079_implicit-bias-iat/code/llm-implicit-bias` | LICENSE |
| `C080_marked-personas/code/markedpersonas` | none (reviewed) |
| `C086_lost-in-the-middle/code/lost-in-the-middle` | LICENSE |

