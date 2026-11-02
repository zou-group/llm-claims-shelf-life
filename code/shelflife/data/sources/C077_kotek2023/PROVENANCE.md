# C077 source snapshot: Kotek, Dockum & Sun (2023), "Gender bias and stereotypes in Large Language Models"

- URL: https://arxiv.org/pdf/2308.14921 (CI '23 version), retrieved 2026-10-03 with `pdftotext`. No code or data
  release exists.
- `printed_sentences.txt`: the 15 sentence schemas printed in Sec. 3.3, one per line, in the paper's form (male
  occupation subject, female occupation object, pronoun "she"). Line 1 keeps the carrier and question, which the paper
  prints only for #1. PDF line breaks are joined with single spaces. Typographic apostrophes (’) are kept as printed.
  Schema 14 is printed "consulted with florist" (article missing).
- `followup.txt`: the invariant turn-2 follow-up exactly as printed in Sec. 3.3, with the paper's curly quotes and the
  {"he", "she"} slot.
- Why it is needed: the paper's prompts exist only in the PDF. The registry derives all 60 sentences, questions and the
  follow-up from these lines. `tests/claims/test_c077.py` compares them with the first-pass reconstruction
  (`tasks/C077/build_cases.py`).
