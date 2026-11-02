# C067 released queries and model outputs (HF reglab/legal_hallucinations)

- Source: https://huggingface.co/datasets/reglab/legal_hallucinations at revision
  c1d87c069a3b49a564dc82323d93b276a34cb568 (lastModified 2024-06-04). Downloaded `dataset.csv` on 2026-10-04:
  424,081,806 bytes, sha256 c55876bf6165f00cb7e7f222a9a2ef1645b315ff5628b526afc682bea7c52e40. The file's header line
  is repeated as its first data row; that row was dropped.
- Kept: `dataset_zero_shot_minimal_cells_gpt.csv.gz` holds the 54,172 rows with prompt_style == zero_shot, llm in
  {GPT 4, GPT 3.5} and (task, court_level) in {(court_id, coa), (court_id, usdc), (citation_retrieval, scotus|coa|usdc),
  (majority_author, scotus)}. All columns are unchanged. It was read with dtype=str and keep_default_na=False, so
  outputs like "None" stay strings, and written with gzip mtime=0. sha256
  4ed1ccb3e92bb77c83f9db9cc88d10ac5fb23fd45fb5c7101b323786e683af25.
- Why: the `query` column is the exact user message the authors sent (models.Query.query). `llm_output` and
  `correctness_score` are the authors' greedy outputs and scores, used for the archived-output targets.
  `example_correct_answer` is the true answer: the circuit number, the citation, or the SCDB justice id. For USDC Court
  it is the state in GPT-4 rows and the full district name in GPT-3.5 rows; see the module docstring. About 10% of cases
  (reserve.csv) are embargoed and are not included.
