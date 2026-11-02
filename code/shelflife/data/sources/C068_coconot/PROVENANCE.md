# CoCoNot test splits (C068 AbstentionBench)

- Source: Hugging Face `allenai/coconot` @ revision `2cbe16aabf9069f17e48c8daad8aeabc29469eb7`
  - `original/test-00000-of-00001.parquet` (sha256 prefix e84a986e967c19ea, 1001 rows)
  - `contrast/test-00000-of-00001.parquet` (sha256 prefix d2d4f9ea33eac017, 379 rows)
- Retrieved: 2026-10-03, re-downloaded from the pinned URLs; bytes identical to
  `experiments/initial-reproduction/reproductions/C068/source_data/coconot_*_test.parquet`.
- Transformation: `pyarrow.parquet.read_table(...).to_pylist()` written one JSON object per line, row order and all
  five string fields (`id, category, subcategory, prompt, response`) unchanged. Done with the read-only
  initial-reproduction C068 venv (pyarrow is not installed in `code/.venv`; nothing was installed anywhere).
- Why: the official `CoCoNotDataset` loader concatenates original + contrast test splits; `c068_abstentionbench.py`
  runs that loader on these rows to rebuild the 180 CoCoNot items of the project subset.
  `tests/claims/test_c068.py` checks the rebuilt items equal the frozen `cases.jsonl` byte for byte.
- sha256: coconot_original_test.jsonl a2621fd0756db21c5912414da353a98aa629dee4c173f0b858ffc75a15c978db;
  coconot_contrast_test.jsonl fc9435a1eb03c08a6c700ed7931415d819f8c8866a6cab2874f4bb996a6d4672
