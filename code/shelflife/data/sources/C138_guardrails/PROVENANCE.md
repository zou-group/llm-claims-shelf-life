# C138 source provenance

Retrieved 2026-10-06 from https://github.com/vli31/llm-guardrail-sensitivity at commit `370dff9c4e704415451f2aa13144840ffae9f1de`. Official repository linked by arXiv 2407.06866v3 §3. Source CSVs provide all 60 political requests and five personas per ideology; helpers.py provides requests and pipeline.py provides the classifier. No released response CSVs are present. Do not run source scripts: module loading extracts pure functions/classes and supplies an offline capture function.

The subject model is unpinned `gpt-3.5-turbo`; the paper uses an unpinned GPT-4o judge while source Evaluator default is GPT-3.5. Port explicitly chooses first GPT-4o snapshot M001, following the paper. This prevents exact snapshot reproduction being claimed.
