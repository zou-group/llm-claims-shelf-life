Cloned from https://github.com/Spracks/PackageHallucination at commit bd3031cce6a0007997a0604d3ad3365031ed79b2 on 2026-10-04 by `shelflife new C104`.
Official repository for the claim; read-only source for prompts, data and scorers.

Used by C104 (earlier pipeline, HL-N01): Data/Python/LLM_{LY,AT}.json (prompts), Data/Python/pypi_package_names.csv
(10-Jan-2024 PyPI master list, per the paper), Data/Python/false_positive_packages.csv, generate_package_names.py
(mode 2 prompt builder), package_detection.py + custom_parse_python.py (scorer), Plots/Data/*.csv (figure data).
The JavaScript data, Mitigation/ and the rest are unused. Per-prompt outputs are not released by the authors.

Not stored in git (over 5 MB): `Data/Python/pypi_package_names.csv`. Download as described in the repository's `DOWNLOADS.md`.
