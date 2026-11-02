"""Paper-exact primary scoring, official-repository parity, and format diagnostics."""
from __future__ import annotations
import argparse
import json
import re


def boundary_quotes(text):
    # Appendix C.1 specifies ignoring quotation marks at answer boundaries.
    if text.startswith('"'):
        text = text[1:]
    if text.endswith('"'):
        text = text[:-1]
    return text


def unwrap(text):
    text = text.strip()
    text = re.sub(r"^```(?:text|plaintext)?\s*\n?(.*?)\n?```$", r"\1", text, flags=re.S).strip()
    for _ in range(3):
        changed = False
        for left, right in [('**', '**'), ('__', '__'), ('"', '"'), ('“', '”'), ("'", "'"), ('`', '`')]:
            if text.startswith(left) and text.endswith(right) and len(text) >= len(left) + len(right):
                text = text[len(left):-len(right)].strip()
                changed = True
                break
        if not changed:
            break
    return text


def parse_response(text):
    """Extract a final explicitly labeled answer without consulting the target.

    This is a diagnostic only: adding an explanation is a primary-metric failure.
    Prefer the last answer label, otherwise a sole fenced/quoted answer, otherwise
    the whole response. Never search the expected sentence to select a candidate.
    """
    cleaned = text.strip()
    cleaned = re.sub(r"(?m)^[ \t]*(?:\*\*|__)([^\n:]+:)(?:\*\*|__)", r"\1", cleaned)
    cleaned = re.sub(r"(?m)^[ \t]*(?:\*\*|__)([^\n]+)(?:\*\*|__)[ \t]*$", r"\1", cleaned)
    labels = list(re.finditer(r"(?im)^[\s#*_]*(?:so\s+)?(?:the\s+)?(?:corrected(?:\s+(?:answer|text))?|(?:final\s+)?(?:original text|decoded\s+(?:original\s+)?(?:text|message|sentence)|answer))\s*(?:is|reads)?\s*[*_]*:\s*", cleaned))
    if labels:
        tail = cleaned[labels[-1].end():]
        # A separated following explanation is not part of the answer paragraph.
        return unwrap(tail.split("\n\n", 1)[0]), "last_answer_label"
    prose = re.fullmatch(r"(?:The )?(?:original|decoded) (?:text|message|sentence) (?:is|reads)\s*:\s*(.+)", cleaned, flags=re.I | re.S)
    if prose:
        return unwrap(prose.group(1)), "answer_prose"
    # A sole standalone quoted paragraph is an explicit answer presentation.
    # Inline quotations in demonstrations and word-by-word explanations do not count.
    quoted = re.findall(r'(?m)^\s*(?:\*\*|__)?(?:"[^"\n]+"|“[^”\n]+”)(?:\*\*|__)?\s*$', cleaned)
    if len(quoted) == 1:
        return unwrap(quoted[0]), "sole_quoted_paragraph"
    paragraphs = cleaned.split("\n\n", 1)
    if len(paragraphs) == 2 and re.match(r'(?:I decoded|Each letter\b)', paragraphs[1], flags=re.I):
        return unwrap(paragraphs[0]), "leading_answer_before_explicit_decoding_explanation"
    return unwrap(cleaned), "whole_response"


def edit_distance(first, second):
    previous = list(range(len(second) + 1))
    for i, char in enumerate(first, 1):
        current = [i]
        for j, other in enumerate(second, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (char != other)))
        previous = current
    return previous[-1]


def score(text, item):
    expected = item["expected"]
    if not isinstance(text, str):
        return {"correct": False, "paper_exact": False, "official_repo_correct": False,
                "extracted_exact": False, "parse_error": "response is not text", "parsed_answer": None}
    parsed, method = parse_response(text)
    exact = boundary_quotes(text) == expected
    return {"correct": exact, "paper_exact": exact,
            "official_repo_correct": expected in text,
            "extracted_exact": parsed == expected,
            "parsed_answer": parsed, "parse_method": method,
            "edit_distance": edit_distance(expected, boundary_quotes(text)),
            "parsed_edit_distance": edit_distance(expected, parsed),
            "metric": "paper_appendix_C1_boundary_quotes_exact"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--item", required=True, help="JSON file containing a case")
    parser.add_argument("--response", required=True, help="UTF-8 text response file")
    args = parser.parse_args()
    from pathlib import Path
    print(json.dumps(score(Path(args.response).read_text(), json.loads(Path(args.item).read_text())), ensure_ascii=False))
