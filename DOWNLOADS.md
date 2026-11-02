# Downloading the input files

The original papers' code, prompts and data are not stored in this repository. Every file the configs read is
listed in [`scripts/inputs.json`](scripts/inputs.json) with its path, size, sha256 and origin, pinned to the exact
version our runs used:

| Origin | Files | How it is fetched |
|---|---:|---|
| GitHub repositories at a pinned commit | 1,398 | `raw.githubusercontent.com/<repo>/<commit>/<path>` |
| Hugging Face datasets at a pinned revision | 14 | `huggingface.co/datasets/<repo>/resolve/<revision>/<path>` |
| Release archives and plain downloads (GitHub, OSF, S3) | 16 | the release URL, or one member of a zip |

Files the project prepared from these sources (prepared item lists, transcribed tables, extracted paper text) are
in git next to a `PROVENANCE.md` that says how they were made.

## Install

Run from the repository root. Only Python 3 is needed; about 576 MB is downloaded in total, of which
504 MB are the release archives.

```bash
python scripts/install_inputs.py                 # download what is missing, verify every file
python scripts/install_inputs.py --check         # report only
python scripts/install_inputs.py --claims C7 C10 # only the files those claims read
```

Downloads are cached under `.cache/inputs/`. A file is written only after its sha256 matches the manifest; a
mismatch that persists after three downloads means the upstream file changed, so please open an issue.

You need the inputs only to run or rescore claims. The analysis scripts, the post and the tests that do not read
them work without them; a config that needs a missing file stops with a `FileNotFoundError` naming the path, and
the corresponding tests are skipped.

## The release archives

The 15 files over 5 MB, all public releases of the original papers:

| Claim | File | Source |
|---|---|---|
| C089 | `sharded_instructions_600.json` | https://raw.githubusercontent.com/microsoft/lost_in_conversation/c865793fe34a929d316119b0451d01bd9183bcfd/data/sharded_instructions_600.json |
| C093 | `en.json` | https://raw.githubusercontent.com/chen700564/RGB/65ec39e40e7dc9abb50e9bf1b4f32be3f6f16615/data/en.json |
| C104 | `pypi_package_names.csv` | https://raw.githubusercontent.com/Spracks/PackageHallucination/bd3031cce6a0007997a0604d3ad3365031ed79b2/Data/Python/pypi_package_names.csv |
| C142 | `endorse4_elephant_oeq3k_human.csv` | https://osf.io/download/68dd6b227d6f3663dfaa5310/ |
| C142 | `endorse4_elephant_oeq_pas_gpt4o.csv` | https://osf.io/download/68dd6b227d6f3663dfaa5310/ |
| C013, C014 | `ReviewCritique.jsonl` | https://raw.githubusercontent.com/jiangshdd/ReviewCritique/629932bcfa31879b1df1e011e3cd7188aa3b9b46/data/ReviewCritique.jsonl |
| C024 | `are_you_sure.jsonl` | https://raw.githubusercontent.com/meg-tong/sycophancy-eval/9a1694221e3639887138f61deae344335eca6752/datasets/are_you_sure.jsonl |
| C108 | `feedback.jsonl` | https://raw.githubusercontent.com/meg-tong/sycophancy-eval/9a1694221e3639887138f61deae344335eca6752/datasets/feedback.jsonl |
| C046 | `annotations.json` | https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/cd543a149df89434d8a54582c0151c0b945c3d20/results/gpt4_1106_preview_verbose/weighted_alpaca_eval_gpt4_turbo/annotations.json |
| C086 | `nq-open-20_total_documents_gold_at_0.jsonl.gz` | https://raw.githubusercontent.com/nelson-liu/lost-in-the-middle/29b8a6d042ce29abccee3db1a73171a107d7e6af/qa_data/20_total_documents/nq-open-20_total_documents_gold_at_0.jsonl.gz |
| C086 | `nq-open-20_total_documents_gold_at_14.jsonl.gz` | https://raw.githubusercontent.com/nelson-liu/lost-in-the-middle/29b8a6d042ce29abccee3db1a73171a107d7e6af/qa_data/20_total_documents/nq-open-20_total_documents_gold_at_14.jsonl.gz |
| C086 | `nq-open-20_total_documents_gold_at_19.jsonl.gz` | https://raw.githubusercontent.com/nelson-liu/lost-in-the-middle/29b8a6d042ce29abccee3db1a73171a107d7e6af/qa_data/20_total_documents/nq-open-20_total_documents_gold_at_19.jsonl.gz |
| C086 | `nq-open-20_total_documents_gold_at_4.jsonl.gz` | https://raw.githubusercontent.com/nelson-liu/lost-in-the-middle/29b8a6d042ce29abccee3db1a73171a107d7e6af/qa_data/20_total_documents/nq-open-20_total_documents_gold_at_4.jsonl.gz |
| C086 | `nq-open-20_total_documents_gold_at_9.jsonl.gz` | https://raw.githubusercontent.com/nelson-liu/lost-in-the-middle/29b8a6d042ce29abccee3db1a73171a107d7e6af/qa_data/20_total_documents/nq-open-20_total_documents_gold_at_9.jsonl.gz |
| C026, C142 | `elephant_full_results.zip` | https://osf.io/download/cjwfe/?view_only=37ee66a8020a45c29a38bd704ca61067 |

Every file keeps its original licence; see [`code/docs/DATA.md`](code/docs/DATA.md#licences).
