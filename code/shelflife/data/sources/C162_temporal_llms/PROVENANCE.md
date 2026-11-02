# C162 source provenance

Retrieved2026-10-15 for the temporal-grounding reproduction. No model calls.

- Paper: Qiu et al. *Are Large Language Models Temporally Grounded?*, arXiv2311.08398v2, 2023-11-16. https://arxiv.org/html/2311.08398v2. Target cells: Table2 GPT4 few-shot Inc27.23%, Acc67.41%; zero-shot Inc31.25%, Acc64.29%. Dataset Section3.1:224eventpairs/448questions.
- Author repository: https://github.com/yfqiu-nlp/temporal-llms, commit`fabcd07d4342bef01460def7e48334e7f22192ae`, cloned in`repo/`. Data`dataset/tempeval-bi/`; inference`tempeval-gpt.py`; orchestration`run-tempeval-gpt.sh`; scorer`eval-tempeval-bi.py`; released GPT4 answers+summary`gpt-output/tempeval-qa-bi/{few-shot-icl,zero-shot}`. Retain full repo, including other-model results, as original history.
- GPT2 tokenizer: https://huggingface.co/gpt2/resolve/main/tokenizer.json (downloaded as`gpt2-tokenizer.json`), public model revision`607a30d783dfa663caf39e06633721c8d4cfcd7e` from`gpt2-model-info.json`. GPT2 uses no added BOS/EOS; author's [1:-1] slices actual content tokens.
- Tokenizer cleanup source: https://raw.githubusercontent.com/huggingface/transformers/v4.35.2/src/transformers/tokenization_utils_base.py, downloaded unchanged as`tokenization_utils_base_v4.35.2.py`. Only static`clean_up_tokenization` extracted by AST. Original dependency version unspecified; this contemporaneous2023version is an explicit assumption. Current installed`tokenizers` reads GPT2 tokenizer JSON; cleanup applied separately to reproduce original AutoTokenizer.decode default.
- `prepare.py` executes only original pure prompt functions with this tokenizer adapter and CSV-dataset shim. It writes`prepared.jsonl` (224pairs,896questionprompts across2protocols) and AST-extracted`templates.json`. Rebuild: from project root,`PYTHONPATH=code code/.venv/bin/python code/shelflife/data/sources/C162_temporal_llms/prepare.py`. No network or generation at rebuild time. Tests independently invoke original builders for every prepared request.

SHA256 of frozen artifacts:

- `gpt2-tokenizer.json`: `8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6`
- `tokenization_utils_base_v4.35.2.py`: `7f4cd2d3eca2165d3f901ca65eac7baa5990526bd92c2d812e93cfeac9f9c605`
- `prepared.jsonl`: `7510fd85b4c736a290847bc890acb7f2bd0cf78bb25b46db3b74faaa566c2573`
- `templates.json`: `0c332c681626fb63f350c73fd3212d3286ef073e33636e6dd37f8ec27315bb8f`
