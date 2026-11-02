# C129 source snapshot

Official https://github.com/hyesunyun/MedLitSpin, commit `4ecc829da6733eca6ed196536c4f103a151052f3`, fetched 2026-10-05 with `git clone --depth 1`.
Paper version: arXiv 2502.07963v2 (2025-05-01). `data/spin_and_no_spin_abstracts.csv` is the original Boutron 60-abstract set; do not use the later extended dataset or generated unspun replacements.
`code/run_spin_interpretation_evaluation.py` and combined counterpart contain literal prompts, five questions, scorer, cap 50, seed 42. `code/models/gpt4.py` gives one-user-message API layout, temperature 0, GPT-4o-2024-08-06 / GPT-4o-mini, logprobs/top_logprobs/seed.
`code/eval_outputs/` contains the 22-model original outputs. Baseline files generally release cleaned answers without raw text; joint files release raw answers too. This limits claims about baseline raw parsing validation.
