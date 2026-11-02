# C137 source provenance

Official https://github.com/manestay/borderlines at commit
`5de21467351c8af660707c8bd59479c7574d3570`, retrieved2026-10-05.
`paper-v3.html` is https://arxiv.org/html/2305.14610v3 (NAACL2024 version).
The pinned `disputed_territories.csv` has251territories,161known controllers.
The published/released aggregate counts720unique queries, not726 in selection.
The inference release additionally calls38English claimant queries:758calls.

Prompts and native option arrays come directly from `translate/prompts_mc_q/`
and indexed `translate/terms/line_inds.*.txt`. Native choices are Python-list
literals, NOT semicolon CSV. Current country metadata omits Suriname; same
commit's `data_misc/countries_info_2021-08.json` explicitly supplies Dutch/nl.
This is a source-backed, disclosed join, not a guessed language. Actual source
`langcode2lang`, `add_lang_phrase`, `add_nationalist` functions produce system
messages, including zht Traditional Chinese and distinct nationalist wording.

`gen_response_table.py` native-name/letter/no_manual rules preserved except
out-of-range indices remain unresolved instead of crashing. `calculate_CS.py`
uses separate controller/noncontroller macro denominators, and per-territory
language-pair consistency. Released `all_results/*scores.json` are aggregates,
not raw response tables; they cannot independently validate the reported CS.
Original GPT4 route `gpt-4-0314`; vanillaT0/top_p1/seed2557/cap128English,
at least512translated. Native option order never changes. No paid calls.
