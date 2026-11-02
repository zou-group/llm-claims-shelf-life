# PyPI project-name snapshot, 2026-10-04

- Source: https://pypi.org/simple/ requested with `Accept: application/vnd.pypi.simple.v1+json` (PEP 691 JSON simple
  index, api-version 1.4), retrieved 2026-10-05 03:46 UTC (2026-10-04 US time). `meta._last-serial` = 41822477.
  Response sha256 a591d7c8896de521133e9a5286a515244f47cbe6dac916c29925c06e1d499cb0 (44,459,647 bytes; not kept).
- File: `pypi_simple_names.txt.gz` = the `name` field of all 905,981 projects, sorted, one per line, written with
  gzip mtime=0. Uncompressed sha256 db2820e6e8901a2e78f1eff01162ab1ea8674852c189ed3936ca21e77b6bae7d.
- Why: the earlier pipeline minimal design for C104 (HL-N01) counts a recommended name as hallucinated only if it is absent
  from the paper's 10-Jan-2024 master list AND from the live index. Freezing the live index makes scoring offline and
  deterministic. Names are normalised at load time with PEP 503 (`re.sub(r'[-_.]+', '-', name).lower()`).
- Comparison with the paper's list (normalised): 905,981 now vs 500,500 distinct in the Jan-2024 list; 9,227 Jan-2024
  names are no longer on PyPI (removed projects); 414,708 names are new since.
- Caveat: names registered after being hallucinated (slopsquatting) count as real under this snapshot.
