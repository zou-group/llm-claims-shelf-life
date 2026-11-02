# C104 first-upload dates for post-2024 package names (frozen post-run lookup)

- Purpose: the C104 headline counts a recommended name as real for a model only if it existed on PyPI at the model's
  release date. Names absent from the paper's 10-Jan-2024 list but present in the 2026-10-04 snapshot
  (`C104_pypi_simple_index_2026-10-04`) need their first-upload date.
- Procedure (after a run exists; free public metadata, no API key, no model calls):
  ```python
  from shelflife.experiments import c104_package_hallucination as m
  names = m.candidate_names('../experiments/run_minimal_setups/runs/earlier/runs/<C104 run dir>')
  m.freeze_first_uploads(names)   # GET https://pypi.org/pypi/<name>/json for names not yet frozen
  ```
  Then rescore the run (`exp.rescore(run_dir, 'h3_direct')`); scoring reads only the frozen file.
- File: `first_upload.json` = {source, rule, names: {name: {first_upload: 'YYYY-MM-DD' | null, http_status, fetched}}}.
  first_upload = min `upload_time_iso_8601` over all files of all releases. A project with no files (or HTTP 404, e.g.
  removed since 2026-10-04) has `null` and counts as not real. Entries are only ever added, never refreshed, so
  rescoring is deterministic. Record each freeze (date, run dir, number of names) below.

## Freeze log
- (none; created 2026-10-04 before any run)
