# Claim tables

- **`claims.csv` / `claims.json`**: one row per claim in the post (50 rows). Columns:
  - `blog_id`: the ID readers see (C1–C50).
  - `key`: our internal claim key.
  - `config` and `module`: the `shelflife` config that runs the claim.
  - `title`, `claim` and `area`.
  - `verdict`: strengthened / persists / weakened / dead ("dead" is shown as "expired" in the post).
  - `metric`, `null_value` and `paper_anchor`: `e = (value − null_value) / (paper_anchor − null_value)`.
  - `paper_value_note`, `paper_models`, `paper_date`, `citation` and `url`.
  - `repro_model`, `repro_value` and `repro_e`: our reproduction on the paper's model, which anchors the fitted
    curve.
  - `panel_models`: the number of panel models with a value.
  - `variants_used`: the protocol variants that produced the 20 panel values, with model counts. "Earlier-pipeline
    outputs rescored" means outputs from an earlier run of the same protocol, scored with the current scorer.
- **`panel_values.csv`**: the 1,000 claim × model values behind the plots. Columns: value, 95% CI, *n*, normalised
  effect `e`, reliability flag, the `source` run and notes. It is copied from
  `experiments/run_minimal_setups/analysis/results_long.csv`, restricted to the 50 counted claims, with local paths
  removed.

These tables are generated from the analysis outputs (`experiments/run_minimal_setups/analysis/post/results.json`, `results_long.csv` and the
runs' `configuration.json` files). They are not inputs to any script.
