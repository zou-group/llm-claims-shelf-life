# InjecAgent source provenance (C120)

Retrieved 2026-10-05 from https://github.com/uiuc-kang-lab/InjecAgent at commit `f19c9f2c79a41046eb13c03c51a24c567a8ffa07` using git clone. The original `LICENCE` is preserved. Needed files: `data/test_cases_dh_{base,enhanced}.json`, `data/tools.json`, `src/prompts/agent_prompts.py`, `src/evaluate_prompted_agent.py`, `src/output_parsing.py`, `src/models.py`, `src/utils.py`.

`paper_arxiv_v3.html` is the full paper text from https://arxiv.org/html/2403.02691v3 (2024-08-04 version), fetched with curl. Targets refer to Table 3 direct-harm columns; total/data-stealing values are outside this claim's minimal scope.

`released_results.zip` is the authors' complete public results archive linked under README Update (2024-07-01): https://drive.google.com/file/d/1OptmQBPQ4uLRUNv95ak2fRJF7BN4xOws/view . Retrieved from the normal public Google Drive download form after confirmation. SHA256: `851eea38f394620de4724ce80dc50df8dd2d3cac3e196dcacb0b2ad90b2299cb`. `released_outputs_download` preserves the initial download-confirmation HTML, not model outputs. Archive is read in place without extraction or rewriting. GPT direct-harm outputs for the `InjecAgent` prompt anchor the four Table 3 targets. The archive is source evidence, not a shelflife prior run, and contains no original API payload logs establishing transport replay.

No benchmark tool is executed, no SDK client is constructed, and source AST loading imports only safe standard-library modules. The source parser's NLTK `ngrams(words, 8)` is supplied an equivalent standard-library sliding-window iterator to avoid requiring NLTK for its default unpadded use.
