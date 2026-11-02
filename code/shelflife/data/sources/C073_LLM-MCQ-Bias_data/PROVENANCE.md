# C073 released gpt-3.5 outputs (compact snapshot)

- Source: https://github.com/chujiezheng/LLM-MCQ-Bias_data, commit `29fcbb4807bc53a84ac26a64ddbad9526c940958`,
  `results_mmlu/0s_ichat/{mmlu, mmlu_move_a..d, mmlu_shuffle_both, mmlu_noid, mmlu_cot, mmlu_cyclic, mmlu_simple}/*.jsonl`
  (570 files, 57 subjects x 10 conditions), plus `debias_pride/debias_0.05.json` (copied unchanged).
- Retrieved 2026-10-03 from raw.githubusercontent.com. Every file's git blob SHA-1 matched the commit tree, and the 65
  files initial-reproduction had already fetched (`experiments/initial-reproduction/reproductions/C073/artifacts`)
  were byte-identical. Per-file SHA-256 of the originals: `source_sha256.json`.
- Why: these are the authors' gpt-3.5-turbo outputs behind Tables 1–3. They allow offline `archived_outputs` targets
  (answer-moving accuracies, RStd for default/shuffled IDs/no IDs/CoT, Cyclic Perm and PriDe) and byte checks of every
  prompt. Initial-reproduction fetched only the default run and abstract_algebra ablations.
- Compaction (`build_compact.py`, run once): one gzip JSONL per condition. Lines are the original `result`/`broken`
  records, plus `subject`. `prompt` is replaced by `prompt_sha256` = sha256(json.dumps(prompt, sort_keys=True,
  ensure_ascii=False)). abstract_algebra keeps the full `prompt`, and mmlu_cot keeps the generated chain of thought
  (`cot`). All other fields (`idx`, `options`, `ideal`, `sampled`, `correct`, `observed`) are unchanged. `metric`
  lines are dropped.
- `released_question_errata.json`: the one released question text (high_school_computer_science idx 54, 'Step l')
  that differs from the repo CSV ('Step 1'). It appears in every released condition except cyclic.
