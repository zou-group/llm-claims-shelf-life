# C140 source provenance

Retrieved 2026-10-06 from https://github.com/centerforaisafety/mask at commit `25e0b1201e6c928ebe69f7c5aad6fa9063a377ea`. Official repository linked from arXiv 2503.03750. Pure binary templates and metric functions are extracted; no source evaluation script or SDK executes.

Data downloaded 2026-10-06 from the official `cais/mask` Hugging Face repository, main revision URLs. Bytes are pinned by SHA-256 (matching HF LFS identifiers):

| File | Public source | Rows | SHA-256 |
|---|---|---:|---|
| known_facts.parquet | https://huggingface.co/datasets/cais/mask/resolve/main/known_facts/test-00000-of-00001.parquet |209|`5cc62d3ff1072252aa9819dfb5f9843924787af2278d2ac366f9d79197ea54c9`|
| provided_facts.parquet | https://huggingface.co/datasets/cais/mask/resolve/main/provided_facts/test-00000-of-00001.parquet |274|`c9a1b0ddff3d2f8ca29ef98361a868caeb5623f0b77975fd5a626c2de62a5340`|

The 483-item slice cannot reproduce the paper's full 1,500-item pooled headline or cross-model scaling correlations. Code defaults cap1000/T1, unpinned GPT-4o judge cap500 with no temperature override. Port fixes judge to M020 GPT-4o-2024-08-06 and retains source developer roles for GPT/o1/o3 families. Item canary strings are preserved in records, not added to requests.
