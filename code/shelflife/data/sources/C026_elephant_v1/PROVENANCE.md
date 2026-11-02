# C026 ELEPHANT paper-v1 code and outputs (git history snapshot)

Source: https://github.com/myracheng/elephant, commit `83263944cffdfd8caa48f8da2436642fd4e681bc`
(2025-05-19, "Delete gemini_responses.csv"), the last commit before the September 2025 v2 rewrite. Retrieved
2026-10-03 with a full `git clone` (the project's clone under `experiments/initial-reproduction/claims/C026_*` is a
shallow clone of HEAD `61f7044f` and does not contain these files).

Why: the arXiv v1 paper (2505.13995v1, May 2025) is the source of the claim registry's numbers (emotional validation
0.76 vs 0.22, indirect language 0.87 vs 0.20, indirect action 0.53 vs 0.17, accepting framing 0.90 vs 0.60, binary
AITA FNR 0.44). The v1 code holds the original four OEQ judge prompts (including `indirect_action`, removed in v2), the
open-ended AITA moral-endorsement judge prompt, and the v1 binary AITA generation protocol (`max_tokens=2` when
constrained). HEAD contains none of these.

| File | Verbatim? | git blob at 8326394 |
|---|---|---|
| `elephant.py` | verbatim | `03a31274c21841626fa0913474c3e84916d0eb3b` |
| `get_all_AITA_with_mitigations.py` | verbatim (`full_results_from_paper/`) | `de40408cd0009b2466dc4dcc10b823c021b43def` |
| `get_all_OEQ.py` | verbatim (`full_results_from_paper/`; byte-identical to HEAD `get_model_responses.py`) | `9abbb99273ed4dfd5fe00890f034378dec9459f0` |
| `OEQ_full_results_indirect_action_columns.csv` | derived: the unnamed index column and the nine `indirect_action_*` columns of `full_results_from_paper/OEQ_full_results.csv` (blob `d3e0b507e0c58c51086656c7cc727fe4e79574f9`, sha256 `81a0730efca253c4d829ec4a74c54b9ec3f3ef5201db84106ff36ffd8dbf390c`, 54.8 MB), cell strings unchanged, written with Python `csv` | — |
| `analyze_AITA_cells_0-6.json` | derived: source text and printed outputs of the first seven code cells of `full_results_from_paper/analyze_AITA.ipynb` (blob `9cfbdffb716f70eefac9e94fd5f92ea7026ce102`, sha256 `b7c3d5010e26f3af3c2d8060b6d52a54e66014a51c7892058819d26eff1d8138`) | — |

The other v1 OEQ label columns (`emotional_validation_*`, `indirect_language_*`, `accept_framing_*` and the nine
responses) were checked on 2026-10-03 to be numerically identical, row by row, to the `validation_*`,
`indirectness_*`, `framing_*` columns of the released v2 archive `elephant_full_results/OEQ_full_results.csv` (OSF),
so they are read from that archive instead of being duplicated here. The v1 `AITA_full_results.csv` (binary and
open-ended AITA verdicts) was never committed to any branch of the repository and is not on OSF; only the notebook's
printed table survives (`analyze_AITA_cells_0-6.json`).
