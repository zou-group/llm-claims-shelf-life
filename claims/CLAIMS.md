# The 50 claims

Readers of the post see the IDs C1–C50. The pipeline uses internal keys such as `C043` or `PI-N01`; the mapping is
fixed in [`claim_ids.json`](../experiments/run_minimal_setups/analysis/claim_ids.json). C42 and C43 are two datasets from one paper
and share the config `C142`, so there are 49 configs. *Variants used* counts how many of the 20 panel models each
protocol variant produced. "Earlier-pipeline outputs rescored" means outputs from an earlier run of the same
protocol, scored with the current scorer.

| ID | Claim | Area | Config | Variants used (models) | Verdict |
|---|---|---|---|---|---|
| C1 | LMs answer sub-questions yet fail the 2-hop question | Reasoning | [`c059_compositionality_gap.py`](../code/shelflife/experiments/c059_compositionality_gap.py) | robust (10); direct_2shot (10) | Weakened |
| C2 | "Ignore previous instructions" hijacks prompted apps | Jailbreaks & safety robustness | [`c102_promptinject.py`](../code/shelflife/experiments/c102_promptinject.py) | panel3 (16); robust (4) | Weakened |
| C3 | Territorial-dispute answers depend on query language | Social bias, culture & simulated populations | [`c137_borderlines.py`](../code/shelflife/experiments/c137_borderlines.py) | paper_no_manual_eligible (18); robust (2) | Weakened |
| C4 | LLM judges flip verdicts when answer order is swapped | LLM judges & evaluation validity | [`c043_faireval.py`](../code/shelflife/experiments/c043_faireval.py) | faireval_vanilla (19); robust (1) | Weakened |
| C5 | LLM personas of Black women stress "resilience" | Social bias, culture & simulated populations | [`c080_marked_personas.py`](../code/shelflife/experiments/c080_marked_personas.py) | gpt4_source_t14 (18); robust (1); earlier pipeline outputs rescored (1) | Expired |
| C6 | LLMs add well in base 10 but fail in base 9 | Reasoning | [`c058_counterfactual.py`](../code/shelflife/experiments/c058_counterfactual.py) | paper_addition_plain_s50 (18); robust (2) | Expired |
| C7 | Long-context QA is U-shaped: facts in the middle are lost | Long context, agents & tool use | [`c086_lost_in_the_middle.py`](../code/shelflife/experiments/c086_lost_in_the_middle.py) | paper_qa20 (19); robust (1) | Expired |
| C8 | Safety-tuned models refuse clearly safe prompts | Jailbreaks & safety robustness | [`c041_xstest.py`](../code/shelflife/experiments/c041_xstest.py) | gpt4_system_safe250 (18); robust (2) | Expired |
| C9 | LLMs resolve ambiguous pronouns by gender stereotype | Social bias, culture & simulated populations | [`c077_gender_pronoun_bias.py`](../code/shelflife/experiments/c077_gender_pronoun_bias.py) | paper_session (7); robust (1); earlier pipeline outputs rescored (12) | Weakened |
| C10 | RAG models answer instead of rejecting answerless docs | Factuality, hallucination & calibration | [`c093_rgb_negative_rejection.py`](../code/shelflife/experiments/c093_rgb_negative_rejection.py) | minimal (19); robust (1) | Persists |
| C11 | ChatGPT fabricates most scholarly citations it writes | Science, research ideas & peer review | [`c010_fabricated_citations.py`](../code/shelflife/experiments/c010_fabricated_citations.py) | paper_chatgpt_ui_verify_online (6); robust (3); earlier pipeline outputs rescored (11) | Expired |
| C12 | LLM MCQ accuracy depends on which letter holds the answer | LLM judges & evaluation validity | [`c073_mcq_selection_bias.py`](../code/shelflife/experiments/c073_mcq_selection_bias.py) | robust (9); shuffled_answer_moving_cap32 (8); shuffled_answer_moving (3) | Expired |
| C13 | LLMs answer "Who is X's parent?" but not the reverse | Reasoning | [`c141_reversal_curse.py`](../code/shelflife/experiments/c141_reversal_curse.py) | robust (20) | Weakened |
| C14 | LLMs decode high-probability text better than rare text | Reasoning | [`c057_embers.py`](../code/shelflife/experiments/c057_embers.py) | robust (13); paper_base (7) | Weakened |
| C15 | Self-correction without feedback lowers reasoning accuracy | Reasoning | [`c054_self_correction.py`](../code/shelflife/experiments/c054_self_correction.py) | minimal (18); robust (2) | Expired |
| C16 | Assistants abandon correct answers when asked "Are you sure?" | Sycophancy, deception & strategic behavior | [`c024_are_you_sure.py`](../code/shelflife/experiments/c024_are_you_sure.py) | paper_mc80 (10); robust (10) | Expired |
| C17 | Assistants praise text more when the user likes it | Sycophancy, deception & strategic behavior | [`c108_feedback_sycophancy.py`](../code/shelflife/experiments/c108_feedback_sycophancy.py) | minimal (18); robust (2) | Strengthened |
| C18 | GPT-4 trading agent acts on insider tips under pressure | Sycophancy, deception & strategic behavior | [`c029_insider_trading.py`](../code/shelflife/experiments/c029_insider_trading.py) | paper_reconstructed_gpt4o (16); robust (4) | Expired |
| C19 | Persuasive multi-turn dialogue flips correct answers | Sycophancy, deception & strategic behavior | [`c139_farm.py`](../code/shelflife/experiments/c139_farm.py) | minimal (19); robust (1) | Expired |
| C20 | LLMs hallucinate most answers about US federal cases | Factuality, hallucination & calibration | [`c067_legal_hallucinations.py`](../code/shelflife/experiments/c067_legal_hallucinations.py) | zero_shot (18); robust (2) | Weakened |
| C21 | LLMs lean left on political orientation tests | Social bias, culture & simulated populations | [`c083_political_orientation.py`](../code/shelflife/experiments/c083_political_orientation.py) | two_instrument_10 (19); robust (1) | Persists |
| C22 | Shuffling premise order breaks LLM logical proofs | Reasoning | [`c051_premise_order.py`](../code/shelflife/experiments/c051_premise_order.py) | reconstructed_logic12_fs (18); robust (2) | Expired |
| C23 | Fake references sway LLM judges toward the cited answer | LLM judges & evaluation validity | [`c047_judgement_bias.py`](../code/shelflife/experiments/c047_judgement_bias.py) | minimal_reference (19); robust (1) | Expired |
| C24 | LLMs convict African American English speakers more | Social bias, culture & simulated populations | [`c078_dialect_prejudice.py`](../code/shelflife/experiments/c078_dialect_prejudice.py) | minimal (16); robust (4) | Expired |
| C25 | "IMPORTANT!!! Ignore..." in tool output raises agent ASR | Jailbreaks & safety robustness | [`c120_injecagent.py`](../code/shelflife/experiments/c120_injecagent.py) | paper (18); robust (2) | Expired |
| C26 | Many-shot prompts jailbreak models as shot count grows | Jailbreaks & safety robustness | [`c037_manyshot_psychopathy.py`](../code/shelflife/experiments/c037_manyshot_psychopathy.py) | robust (20) | Weakened |
| C27 | LLM judges prefer longer answers | LLM judges & evaluation validity | [`c046_alpacaeval_length.py`](../code/shelflife/experiments/c046_alpacaeval_length.py) | binary_paired (17); robust (2); earlier pipeline outputs rescored (1) | Weakened |
| C28 | LLM evaluators prefer their own summaries | LLM judges & evaluation validity | [`c044_self_preference.py`](../code/shelflife/experiments/c044_self_preference.py) | generated_fixed_neutral_binary (18); robust (2) | Strengthened |
| C29 | LLMs asked for a random number are far from uniform | Creativity, output diversity & homogeneity | [`c023_random_numbers.py`](../code/shelflife/experiments/c023_random_numbers.py) | rng0 (8); earlier pipeline outputs rescored (12) | Strengthened |
| C30 | Code LLMs recommend packages that do not exist | Factuality, hallucination & calibration | [`c104_package_hallucination.py`](../code/shelflife/experiments/c104_package_hallucination.py) | h3_direct (19); robust (1) | Weakened |
| C31 | LLM reviewers score accepted and rejected papers alike | Science, research ideas & peer review | [`c014_reviewcritique.py`](../code/shelflife/experiments/c014_reviewcritique.py) | table14_reconstructed_v1_cap8192 (6); robust (3); earlier pipeline outputs rescored (11) | Persists |
| C32 | ChatGPT refuses more when user persona opposes request | Sycophancy, deception & strategic behavior | [`c138_political_guardrails.py`](../code/shelflife/experiments/c138_political_guardrails.py) | minimal (19); robust_parse (1) | Expired |
| C33 | LLMs comply with illogical drug-switch requests | Sycophancy, deception & strategic behavior | [`c109_drug_equivalence_compliance.py`](../code/shelflife/experiments/c109_drug_equivalence_compliance.py) | minimal (19); robust (1) | Expired |
| C34 | Hidden prompt injection steers LLM peer reviews | Science, research ideas & peer review | [`c013_peerreview_injection.py`](../code/shelflife/experiments/c013_peerreview_injection.py) | reviewcritique_text_substitute (12); robust (8) | Weakened |
| C35 | Chat LLMs say "strawberry" has 2 r's, not 3 | Reasoning | [`c101_strawberry.py`](../code/shelflife/experiments/c101_strawberry.py) | toy (18); robust (2) | Expired |
| C36 | Reasoning models overthink easy math | Reasoning | [`c055_overthinking.py`](../code/shelflife/experiments/c055_overthinking.py) | gsm8k (19); robust (1) | Expired |
| C37 | Spin in trial abstracts inflates LLM benefit ratings | Science, research ideas & peer review | [`c129_medlitspin.py`](../code/shelflife/experiments/c129_medlitspin.py) | benefit_baseline_nologprobs (12); benefit_baseline (7); robust (1) | Persists |
| C38 | Models state falsehoods under pressure despite knowing truth | Sycophancy, deception & strategic behavior | [`c140_mask.py`](../code/shelflife/experiments/c140_mask.py) | minimal_system_roles (19); robust (1) | Weakened |
| C39 | LLMs get lost when instructions arrive over many turns | Long context, agents & tool use | [`c089_lost_conversation.py`](../code/shelflife/experiments/c089_lost_conversation.py) | paper_math (19); robust (1) | Weakened |
| C40 | LLMs side with posters the crowd judged at fault (AITA) | Sycophancy, deception & strategic behavior | [`c026_elephant.py`](../code/shelflife/experiments/c026_elephant.py) | aita_binary_v1 (19); robust (1) | Weakened |
| C41 | LLMs fail to abstain on unanswerable questions | Factuality, hallucination & calibration | [`c068_abstentionbench.py`](../code/shelflife/experiments/c068_abstentionbench.py) | paper_subset330 (19); robust (1) | Persists |
| C42 | AI affirms users' actions more often than humans do | Sycophancy, deception & strategic behavior | [`c142_sycophantic_ai_science.py`](../code/shelflife/experiments/c142_sycophantic_ai_science.py) | robust (20) | Persists |
| C43 | AI endorses problematic actions users describe | Sycophancy, deception & strategic behavior | [`c142_sycophantic_ai_science.py`](../code/shelflife/experiments/c142_sycophantic_ai_science.py) | pas_endorse_robust (20) | Persists |
| C44 | Frontier models blackmail an executive to avoid replacement | Sycophancy, deception & strategic behavior | [`c028_agentic_misalignment.py`](../code/shelflife/experiments/c028_agentic_misalignment.py) | detector_gpt6 (16); robust (3); earlier pipeline outputs rescored (1) | Expired |
| C45 | LLMs give near-identical answers to open-ended prompts | Creativity, output diversity & homogeneity | [`c143_artificial_hivemind.py`](../code/shelflife/experiments/c143_artificial_hivemind.py) | top_p_r20 (15); top_p_r20_panel_completion (5) | Persists |
| C46 | LLMs leak information marked private when summarizing meetings | Jailbreaks & safety robustness | [`c161_confaide.py`](../code/shelflife/experiments/c161_confaide.py) | robust (20) | Expired |
| C47 | LLMs give contradictory answers to paired before/after timeline questions | Reasoning | [`c162_temporal_grounding.py`](../code/shelflife/experiments/c162_temporal_grounding.py) | robust (18); zero_shot_panel (2) | Weakened |
| C48 | LLMs show implicit stereotype bias in IAT-style word association | Social bias, culture & simulated populations | [`c079_implicit_bias_iat.py`](../code/shelflife/experiments/c079_implicit_bias_iat.py) | gpt4_replication (14); robust (5); earlier pipeline outputs rescored (1) | Weakened |
| C49 | LLMs prefer AI-written research abstracts over human-written ones | LLM judges & evaluation validity | [`c157_ai_ai_bias.py`](../code/shelflife/experiments/c157_ai_ai_bias.py) | robust (20) | Expired |
| C50 | LLMs give more nonanswers to political-biography requests in Arabic than in English | Social bias, culture & simulated populations | [`c166_language_censorship.py`](../code/shelflife/experiments/c166_language_censorship.py) | robust (20) | Expired |

[`claims.csv`](claims.csv) adds the metric, its null value, the paper value and our reproduction
anchor, the paper's models and the citation.
[`panel_values.csv`](panel_values.csv) has all 1,000 claim × model values with CIs, *n*, a
reliability flag and the run each value came from.
