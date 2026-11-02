# C013 source snapshot

arXiv https://arxiv.org/e-print/2412.01708v1 fetched 2026-10-05 and unpacked intact. The arXiv record has only v1, submitted 2024-12-02. `main.tex` anchors Methods and Tables 2–3; `figs/llm_review_prompt.pdf` and `figs/injected_content.pdf` contain actual reviewer/attack text. Poppler recovers their lines; the attack figure is cropped to x=0..550 points to omit the 'Injected Content' label at x>=559 and font annotations. Prompt content is not rewritten into an imagined attack.

The paper trains a Llama-3-8B Review2Rating model; no weights or original 1,000 manipulated manuscript set were released in the inspected source. Original protocol is blocked explicitly.
The non-faithful runnable data comes from existing official ReviewCritique commit `629932bcfa31879b1df1e011e3cd7188aa3b9b46` under initial-reproduction C014, 50 accepted and 50 rejected real papers; select the first 25 of each in original release order, interleave, then truncate at 6,500 cl100k tokens before appending the extracted attack. This differs from injecting into a PDF after its conclusion. M015 judge/rating is an agent-written substitute, not original Review2Rating.
