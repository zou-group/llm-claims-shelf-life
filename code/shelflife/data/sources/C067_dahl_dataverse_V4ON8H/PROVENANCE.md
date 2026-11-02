# C067 replication package (Dahl, Magesh, Suzgun, Ho — Large Legal Fictions)

- Harvard Dataverse doi:10.7910/DVN/V4ON8H, version 1.0 (released 2024-04-26). Retrieved 2026-10-04 through the
  Dataverse API (`/api/access/datafile/<id>`). `datafile_ids_md5.txt` lists the file id, path and Dataverse md5 of every
  downloaded file. correctness_checks.py md5 169538086063acabf3259a40c1a6a7a3 matches.
- Kept verbatim: all .py/.R/.md/.tex/.txt files of the package (code, prompts, the R tables script, the LaTeX
  result tables), under `legal_hallucinations/`. The 270-odd `*_results_temp=*.tab` raw-output files and the figures
  were not downloaded. The released outputs used here come from the HF dataset (see C067_reglab_legal_hallucinations_hf).
- Added: `legal_hallucinations/data/covariates/scdb_justice_name_map.csv`, Dataverse file 10146103
  (covariates_scdb_justice_name_map.tab) in its original CSV format (`?format=original`). It maps SCDB justice IDs to
  names and is used for the Author true answer and the clean_judge_name regexes.
- Why: tasks_{scotus,coa,usdc}.py hold the verbatim prompts and Task params, api.py the request layout, models.py
  the greedy protocol, correctness_checks.py the scorer, tables.R the hallucination rule, results/tables the paper's
  tables. The results tables are identical to the arXiv v2 LaTeX `Tables/CategoryOneTable.tex`.
- Scorer dependencies, not vendored: eyecite==2.5.2, reporters-db==3.2.36, courts-db==0.10.17, thefuzz==0.19.0,
  word2number==1.1 (the authors' pins), python-Levenshtein 0.27.5 (paper 0.21.0, which has no Python 3.12 wheel), and
  spaCy en_core_web_sm 3.8.0 (paper 3.5.0). They were installed into code/.venv on 2026-10-04 for C067.
