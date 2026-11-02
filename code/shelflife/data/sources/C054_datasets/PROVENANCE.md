# C054 datasets

- `csqa_dev_rand_split.jsonl`: CommonsenseQA dev set (1,221 questions), https://s3.amazonaws.com/commensenseqa/dev_rand_split.jsonl,
  retrieved 2026-10-04. sha256 3210497fdaae614ac085d9eb873dd7f4d49b6f965a93adadc803e1229fd8a02a (identical to the copy
  in experiments/initial-reproduction/claims/C073_*/code/LLM-MCQ-Bias/code/data_csqa/dev_rand_split.jsonl).
- `gsm8k_test.jsonl`: GSM8K test set (1,319 problems), https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
  (repo HEAD 3101c7d5072418e28b9008a6636bde82a006892c), retrieved 2026-10-04. sha256
  3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14 (identical to experiments/initial-reproduction/reproductions/C068/source_data/gsm8k_test.jsonl).
- Why: Huang et al. 2023 evaluate on the CSQA dev set (Sec 3.1, following Kojima et al. / Kim et al.) and the GSM8K
  test set; their 200-question samples were not released.
