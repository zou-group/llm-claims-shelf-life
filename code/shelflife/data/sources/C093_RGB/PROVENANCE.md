Cloned from https://github.com/chen700564/RGB at commit 65ec39e40e7dc9abb50e9bf1b4f32be3f6f16615 on 2026-10-04 by `shelflife new C093`.
Official repository for the claim; read-only source for prompts, data and scorers.
Needed for C093 (earlier pipeline): data/en.json + zh.json (negative documents), config/instruction.yaml (prompts),
evalue.py (document sampling, label rule), reject_evalue.py (Rej* judge prompt and rule), models/models.py
(OpenAIAPIModel request shape). No model outputs are released. Paper: arXiv 2309.01431 v2 LaTeX read 2026-10-04.

Not stored in git (over 5 MB): `data/en.json`. Download as described in the repository's `DOWNLOADS.md`.
