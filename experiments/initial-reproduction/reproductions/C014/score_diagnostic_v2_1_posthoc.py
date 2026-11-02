"""Posthoc format sensitivity: one explicit bold score under Recommendation.

Created after the main run exposed this format. Never replaces frozen strict/v1.
Only a full Markdown heading followed by a single bold numeric-and-label row is
normalized; ratings elsewhere and rubric blocks remain unavailable.
"""
import re
from score_diagnostic import score as v1_score


def score(text, item):
    text = text if isinstance(text, str) else ''
    lines = text.splitlines()
    normalized = list(lines)
    recovered = []
    heading = re.compile(r'^\s*#{1,6}\s+(?:\*\*)?Recommendation(?:\s+(?:Score|Rating))?(?:\*\*)?\s*:?\s*$', re.I)
    colon_row = re.compile(r'^\s*\*\*(\d+(?:\.\d+)?)(?:\s*/\s*10)?\s*:\s*([^\n]+?)\*\*\s*$')
    dash_row = re.compile(r'^\s*\*\*(\d+(?:\.\d+)?)(?:\s*/\s*10)?\*\*\s*[-–—]\s*([^\n]+?)\s*$')
    numeric_row = re.compile(r'^\s*(?:[-*+]\s+)?(?:\*\*)?\d+(?:\.\d+)?(?:\s*/\s*10)?\s*(?:[:.)]|\*\*|$)')
    for i, line in enumerate(lines):
        if not heading.fullmatch(line):
            continue
        end = next((j for j in range(i+1, len(lines)) if re.match(r'^\s*#{1,6}\s', lines[j])), len(lines))
        nonempty = [j for j in range(i+1, end) if lines[j].strip()]
        if not nonempty:
            continue
        j = nonempty[0]
        match = colon_row.fullmatch(lines[j]) or dash_row.fullmatch(lines[j])
        # Reject multiple choice/rubric lines in the same recommendation section.
        if not match or sum(bool(numeric_row.match(lines[k])) for k in nonempty) != 1:
            continue
        normalized[i] = 'Recommendation: ' + match[1] + ': ' + match[2]
        normalized[j] = ''
        recovered.append({'heading': line, 'rating_row': lines[j]})
    if not recovered:
        return {**v1_score(text, item), 'diagnostic_version': '2.1_posthoc',
                'diagnostic_method_v2': 'unchanged_v1', 'posthoc': True}
    result = v1_score('\n'.join(normalized), item)
    return {**result, 'diagnostic_version': '2.1_posthoc', 'posthoc': True,
            'diagnostic_method_v2': 'explicit_heading_single_bold_numeric_label_row',
            'posthoc_format_evidence': recovered}
