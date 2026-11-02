# Paper release dates

[paper-releases.json](paper-releases.json) gives the release date of each of the 48 papers behind the post's 50
claims, with a source link. Dates were reviewed on 2026-10-27.

A paper's release is its earliest publicly accessible version. We use arXiv v1, never a later revision, and prefer a
verified earlier preprint, full author report or publisher release. We do not use the date written, a private
submission, acceptance, a date embedded in a DOI or an issue year.

Each record has `first_public_date`, `date_kind`, `status`, `source_url`, aliases and the claims it dates.
`verified_source_date` means the cited release date was verified; it does not prove that no earlier copy existed.
`provisional` marks an unresolved exact first-access date. Per-paper caveats note further uncertainty.

Open cases: the August 8, 2024 social-science working paper (date written versus exact public posting), and whether
the OpenReview version of Artificial Hivemind was public before arXiv. The strawberry date belongs to the related
letter-counting paper, not to the informal claim's origin.

`experiments/run_minimal_setups/analysis/scripts/paper_dates.py` reads this file when `post_data.py` runs.
