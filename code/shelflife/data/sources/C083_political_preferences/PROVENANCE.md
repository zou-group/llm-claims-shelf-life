# C083 source provenance

Official author Zenodo https://zenodo.org/records/10553530 retrieved2026-10-05/06:
`tabulated_results.csv`, `stance-detection.rar` and partial `results.rar`.
The record metadata (https://zenodo.org/api/records/10553530) reports results.rar4252835963
bytes. Download ended at about1.1GB (curl18) before Political Compass section;
both API and record-file resume endpoints repeatedly returnedHTTP503.
The archive is partial.

Complete early archive members were extracted via bsdtar (later archive
truncation causes nonzero exit).3308values JSONL profiles exist;306have exactly
70answers plus a matching score JSON.24incomplete source profiles are not
filled in.70question literals and18prefix/28suffix alternatives come from
official GPT4 trial1 JSONL/log. Random expanded prompt layout checked against
every trial1 question. Source interface `gpt-4` and mapper`gpt-3.5-turbo` are
aliases with no snapshot pinned. Model paramsT0.7/cap100; source always maps
through GPT3.5 and retries up to10, sometimes random-answer fallback.

Original PCT source pages1..6 downloaded from
https://www.politicalcompass.org/test/en?page=1 (and pages2..6),62propositions.
HTML retrieval2026-10-06 is not a pinned2024questionnaire snapshot.
Unofficial scorer https://github.com/politicalcompass/politicalcompass.github.io
at7019e5c26ad24e9acd1100c3cba27a44004afe76 gives62×4matrices, economicoffset0.38
/scale8 and socialoffset2.41/scale19.5. Archived PCT validation is NOT available.

Upstream8values https://github.com/8values/8values.github.io at
3971ed8bc0c3dda18f9b001013327f382343b74f supplies all weights. IDRlabs wording
replaces physician-assisted suicide with victimless crime (govt10), and
oppression's govt weight is0 instead of upstream-5. The latter difference
was identified by fitting the source civil scores, then rounding the normalized
coefficients to existing5point increments; all306complete profiles'1224axes
match exactly. This is reverse-engineered, not an official IDRlabs weight release
or an independent held-out statistical discovery. No tolerance was widened.

The official https://www.idrlabs.com/8-values-political/test.php snapshot includes
the same70question set, randomized order, no exposed server-side weights.
Full11instrument401-question port is not supplied; no fake questionnaires.
