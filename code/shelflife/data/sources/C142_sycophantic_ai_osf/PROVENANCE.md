# C142 — OSF project for Cheng et al., "Sycophantic AI decreases prosocial intentions and promotes dependence"

Paper: Science 391(6792), eaec8352 (2026-03-26), doi 10.1126/science.aec8352; preprint arXiv 2510.01395v1 (2025-10-01).
Source: OSF project `smvw7`, view-only link printed in the arXiv paper (sn-article.tex line 429):
`https://osf.io/smvw7/?view_only=ad71a7201c71477d921000c90c565da7`. The node is also public (API without the key
returns 200). Node `date_modified` 2026-01-18 (description updated to the Science abstract); every file below is dated
2025-10-01 except `Materials/scenarios.pdf` (2025-08-07), so the Study 1 data predates and is unchanged by the Science
revision. No GitHub repo exists for this paper (checked `gh api users/myracheng/repos`, 2026-10-09); the OEQ generation
code is ELEPHANT's (`myracheng/elephant`, see C026 sources).

Retrieved 2026-10-09 via the OSF v2 API (`osfstorage`, recursive listing; script in
`experiments/run_minimal_setups/runs/added-b/sources/science_sycophancy/scripts/osf_list.py`). Every file's sha256 equals the
OSF-reported sha256.

| Path | OSF file id | Bytes | sha256 |
|---|---|---|---|
| `Data/AITA_endorsement_results.csv` | 68dd6a787d6f3663dfaa52f7 | 488774203 | 2ed2574d37557e6f1a77edd0c22b58ebd199e58c5ac4bef292366a0ad6c733d4 |
| `Data/OEQ_and_PAS_results.zip` | 68dd6b227d6f3663dfaa5310 | 292070416 | d46b5364ed75a93e7743b416f024cc264ebf6da02a4ca876bd27d0df81ca3d7a |
| `Data/study2_responses.csv` | 68dd69ca5415eabd280db317 | 2657075 | 06cce7e83e785e68f510b4f0d06e16562a3e9a4ed991fd9cd54f16f5f8d97d6c |
| `Data/study3_responses.redacted.csv` | 68dd69bde7119203d9aa534e | 9435077 | 15260e72f44423eef2abdadaeddc49e742318d30a51eaa4b8357e54608ecd549 |
| `Code/evaluate_action_endorsement.py` | 68dd6b5297ed937a43fd049f | 8406 | 89d5a6237d3b55ded53cdd3290a70a24cb7adef43cb0e7e3b8476c4b9b9fd685 |
| `Code/Study 1 Analyses.zip` | 68dd6db5805c588e54aa53b0 | 1638179 | b632721ce54421513f73ad14753b8b5843cb2e030033c7c2d52d71460affd9a5 |
| `Code/Study_2_analysis.ipynb` | 68dd6b47e7119203d9aa538d | 1146281 | 4a81ff6deef7a1b62ca3259254c2e0b7459c0dadd39288b2069a2eabbfc7e5ed |
| `Code/Study_3_analysis.ipynb` | 68dd6b47b9729a76b1fd083f | 451942 | 503956161b93789a9e60b1283dc168e6df4a8129a1360634f384d1aa22c68a47 |
| `Materials/Study_2_Hypothetical.qsf` | 68dd6ed2d6c3cdf80bd6c81e | 366853 | 1abc804f037bb19b26d424125d75747992b2ea0ad27a8db8f6cedca88e757761 |
| `Materials/study3_survey.html` | 68dd6f109df92df1f2aa51ce | 283210 | 28e483a2834029e63aed038add082723beb4b7e1a7bd1d4c2cf1725e879d1a33 |
| `Materials/scenarios.pdf` | 6894bdb187ab480e9e1b5421 | 472892 | 5ecd51b923df5bb743c69fd8c1ac3b9343ca3395f7396fb437e8124ee26cd600 |

`extracted/` (derived, for convenience): `unzip` of `Code/Study 1 Analyses.zip` and `Data/OEQ_and_PAS_results.zip`
with `__MACOSX/*` excluded, contents unchanged:
- `extracted/Study 1 Analyses/{OEQ,PAS,AITA}_endorsement_rates.ipynb` — authors' Study 1 analysis notebooks with outputs.
- `extracted/OEQ_results/endorse4_elephant_oeq3k_<model>.csv` — 12 files (11 models + human), 3,027 rows each:
  responses + 4-class `endorse_<col>` judge label.
- `extracted/PAS_results/endorse4_elephant_oeq_pas_<model>.csv` — 11 files, 6,433 rows each: PAS sentence
  (`sentences`), GPT-4o category tags, response, `endorse_<col>` label.

The full OSF archives are not in this repository; see the note at the end for the two files the configs read.

Not stored in git (over 5 MB): `extracted/OEQ_results/endorse4_elephant_oeq3k_human.csv`, `extracted/PAS_results/endorse4_elephant_oeq_pas_gpt4o.csv`. Download as described in the repository's `DOWNLOADS.md`.
