# ConfAIde source snapshot

- Official repository: https://github.com/skywalker023/confaide
- Commit: `3174c63e689226189675a740b7bf9053f10bae50`
- Retrieved 2026-10-15 by git clone, full repository, original license retained.
- Paper: https://arxiv.org/html/2310.17884v2 (28 June 2024), ICLR 2024.
- Needed: `benchmark/tier_4.txt` (20 meetings), `eval.py` (prompts/parser/scoring), `agents/gpt.py` (actual API kwargs).
- No released model outputs found: repository tree contains agents, benchmark, eval script and assets only. No results directory, output dataset or linked output archive in README.
- Source chat generate passes only `model` and `messages`; all configured generation parameters are ignored. Faithful variants therefore specify no generation params. Runtime policy adds a finite cap and records it.
