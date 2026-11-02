# C141 source snapshot: official Reversal Curse repository

- URL: https://github.com/lukasberglund/reversal_curse
- Commit: 6af5f418755d528304e406361e86477637d21c84 (HEAD of default branch, committed 2023-10-06), shallow clone (`--depth 1`).
  `.git` is 1.9 MB and kept.
- Retrieved: 2026-10-09.
- Why: verbatim Experiment 2 prompts and scoring (`src/tasks/celebrity_relations/parent_reversals.py`,
  `scripts/celebrity_relations/test_parent_child_pairs.py`, `scripts/celebrity_relations/find_non_reversals_parents.py`),
  the released item set (`data/celebrity_relations/parent_child_pairs.csv`, 1,513 rows) and released per-pair model
  outputs (`data/celebrity_relations/{gpt-3.5-turbo,davinci,llama-7b,llama-30b,llama-65b}_reversal_test_results.csv`),
  which reproduce the paper's Figure (Experiment 2) and the GPT-4 33% reverse number offline. Also holds the
  Experiment 1 (`data/reverse_experiments/june_version_7921032488/`) and Experiment 3
  (`data/instructions/copypaste_ug100_rg1000_main/`) finetuning datasets.
- Also published as HF dataset `lberglund/reversal_curse` (not downloaded).
- Untrusted third-party code: do not import it; read files only.
