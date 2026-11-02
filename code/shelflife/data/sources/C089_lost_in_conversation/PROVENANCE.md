# C089 source provenance

Official https://github.com/microsoft/lost_in_conversation, commit
c865793fe34a929d316119b0451d01bd9183bcfd, retrieved 2026-10-05.
Paper snapshot `paper-v1.html` https://arxiv.org/html/2505.06120v1.
Table1 source row S6.T1.2.1.15: GPT-4o FULL Code88.4 Math93.8;
SHARDED Code61.3 Math67.9. Released output trajectories are not in this repo.
Data `data/sharded_instructions_600.json` actually627 instructions,
103Math and100Code, preserved in source order and interleaved by task in port.

Original prompt templates, task classes, user_agent.py, system_agent.py and
simulator_full/sharded.py anchor implementation. Code evaluator is the released
LiveCodeBench-derived eval_code.py. Python execution is isolated under macOS
sandbox-exec with deny-default/no network, restricted reads, temp-only writes,
no inherited credentials,30sec parent timeout and author6sec test timeout.
The research execution environment prohibits initializing that nested sandbox;
automatic approval review rejected an escalated synthetic scorer validation.
The port fails before any model call on Code if its sandbox cannot initialize.

Not stored in git (over 5 MB): `data/sharded_instructions_600.json`. Download as described in the repository's `DOWNLOADS.md`.
