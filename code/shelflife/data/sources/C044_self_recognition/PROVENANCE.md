# C044 self recognition provenance

Official repository: https://github.com/ArjunPanickssery/self_recognition
Pinned upstream commit: `5b1cb7321eef8a1564c8c782917e3706aafcc589`.
Copied 2026-10-05 from the project's read-only clone of the official repository, preserving protocol files, article and summary data and `results/xsum/{gpt4,gpt35}_results.json`. This is a selected-file snapshot, without a nested git checkout; the commit is recorded here rather than asserted against the enclosing project's git HEAD.

Prompts are verbatim `prompts.py` constants. `models.py` fixes GPT-4-1106-preview and GPT-3.5-turbo-1106, max_tokens 10, temperature 0, logprobs true, top_logprobs 2. `experiments.py` averages probability of the author's summary across both orders. Frozen article keys and summary keys are joined explicitly. The released scores are available offline; raw two-token logprobs underlying them are absent, so recorded probability values anchor released-output analysis but cannot independently validate their full original extraction.
