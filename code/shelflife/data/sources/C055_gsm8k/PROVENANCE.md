# GSM8K test split (C055)

- URL: https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
- Repository HEAD at retrieval: 3101c7d5072418e28b9008a6636bde82a006892c (openai/grade-school-math), 2026-10-04.
- sha256 3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14, 1,319 lines (paper Sec. 2.1: 1,319).
- Why: Chen et al. (C055) evaluate on the GSM8K test set; the official overthinking repo does not ship it. Its
  debug.jsonl item 0 equals test.jsonl line 0's question byte for byte (tested).
