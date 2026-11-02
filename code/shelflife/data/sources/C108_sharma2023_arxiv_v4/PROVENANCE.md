# C108 paper source snapshot (Sharma et al. 2023, "Towards Understanding Sycophancy in Language Models")

- Retrieved 2026-10-04 from `https://arxiv.org/e-print/2310.13548` (latest version = v4; tarball sha256
  `ecd647bc8fca9a6899778929e71064294d6b7041e885fab7309ba9185013f940`).
- Kept: `main.tex` (§3.1 "AI Assistants Can Give Biased Feedback", Fig. 1 caption, metric definition), `appendix.tex`
  (A.3 "Evaluating positivity" judge prompt; Fig. 7 per-domain panels), and the four vector figure PDFs of
  `figures/feedback/` (Fig. 1 right panel and Fig. 7 a-c).
- The four feedback figure PDFs are byte-identical between the v1 (2023-10-20) and v4 e-prints (sha256 compared on
  2026-10-04), and the A.3 "Evaluating positivity" prompt text is unchanged, so targets taken from them apply to the
  version the claim cites.
- Why: the paper reports feedback positivity only in figures. The PDFs are matplotlib vector output, so the bar heights
  are recovered exactly from the drawing operators (`shelflife.experiments.c108_feedback_sycophancy.figure_values`);
  no pixel reading is involved. The released code/data (meg-tong/sycophancy-eval @ 9a16942, shared with C024) hold the
  prompts and judge template; no model outputs were released.
