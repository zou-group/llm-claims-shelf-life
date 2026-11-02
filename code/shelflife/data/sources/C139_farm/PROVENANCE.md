# C139 source provenance

Retrieved 2026-10-06 from https://github.com/pillowsofwind/llms-believe-the-earth-is-flat at commit `61e8239993bc4ea825aca5df593f5213c834ba59`. Official repository linked from https://llms-believe-the-earth-is-flat.github.io/ and arXiv 2312.09085v5. All 1,952 public Farm records and run_exp.py are included; author model outputs are not released here.

The port extracts pure author parsers without running CLI/model-loading code. Historical GPT chat requests use `gpt-3.5-turbo` and `gpt-4` aliases (no dated snapshots specified), T=.2 for probes and T=.8 for conversational replies, cap512. The released code includes BoolQ eligibility and malformed-probe quirks documented in the claim module/findings. Local Llama/Vicuna weight loading is not implemented in this chat port.
