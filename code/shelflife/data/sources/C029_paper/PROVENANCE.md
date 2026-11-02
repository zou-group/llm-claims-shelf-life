# C029 source provenance

Official prompts and released annotated outputs already existed read-only at `experiments/initial-reproduction/claims/C029_insider-trading/code/insider-trading`, commit `370fdc9f303f972fe3d2e40b71d06bc04027668b`, https://github.com/ApolloResearch/insider-trading. No archive files changed.

Retrieved arXiv 2311.07590v4 LaTeX 2026-10-06 from https://arxiv.org/e-print/2311.07590v4. Source tar SHA-256 `354670c50935fcee2e6baae87119637521fa14112e8e02fcfd586a588a399b7c`. appendix.tex A.5 describes the classifier rule and four shots but does not publish their instructions/text. Faithful execution therefore fails before calls.

`reconstructed_judge.txt` was authored for this port from A.5. The four examples are deterministic selections (two positive/two negative) from released GPT-4-32k report/reply annotations. They are a reconstruction, not the historical classifier. PromptTemplate provenance labels the distinction. Double-quote stripping before the source trade-match string is inferred because it reproduces every released annotation; raw literal matching misses 6/300 GPT-4-32k and 4/300 GPT-4 trades.

Retain original canary strings with any exported dataset. They remain item metadata and are not inserted into model requests.
