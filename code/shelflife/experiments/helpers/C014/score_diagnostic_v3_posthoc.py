"""Post-hoc v3 explicit-field parser (panel-30 extension, 2026-10-01).

Created after panel outputs showed explicit numeric recommendations in heading/value layouts
missed by strict/v1/v2.1 (e.g. `### Recommendation` then `8: accept, good paper`,
`**8 (Accept, good paper)**`, `**Accept (8/10)**`). Never replaces strict/v1/v2.1; reported
alongside them.

Rule: if v2.1 parses, use it. Otherwise find every line naming the Recommendation field
(optionally Markdown heading, bold, numbering, bullet, `Final`/`Overall` prefix, `Score`/`Rating`
suffix). Its value is the same-line remainder after the field name, or else the first non-empty
line of the section. A value is a 1-10 integer in an explicit rating position: leading number
(`8`, `8:`, `8 -`, `8 (label)`, `8/10`), or a parenthesised / slash-10 number (`(8)`, `(8/10)`,
`8/10`). The value line must yield exactly one distinct number, and the section must not be a
rubric echo (two or more lines starting with `N:`). All Recommendation fields must agree.
Prose such as "I recommend acceptance" is never converted into a number.
"""
import re
from score_diagnostic_v2_1_posthoc import score as v21_score
from score import ALLOWED_ANCHORS

FIELD = re.compile(r'^\s*(?:[>*+-]\s*)?(?:#{1,6}\s*)?(?:\d+[.)]\s*)?(?:Final\s+|Overall\s+|Initial\s+)?Recommendation(?:\s+(?:Score|Rating))?\s*(?:\(1\s*-\s*10\))?(?:\s*[:：–—-]\s*(.*)|\s*\(\s*(\d[^)]*)\)\s*[:：]?\s*|\s*)$', re.I)
TABLE_FIELD = re.compile(r'^\s*(?:Final\s+|Overall\s+)?Recommendation(?:\s+(?:Score|Rating))?\s*$', re.I)
PROSE = re.compile(r'(?:\b(?:rating|score)\s*(?:of\s+|[:：]\s*|\s)|\bas\s+(?:an?\s+)?|\b(?:accept|reject|threshold|paper)\s*\()(\d+(?:\.\d+)?)(?:\s*/\s*10)?(?!\d)(?!\.\d)', re.I)
HEADINGISH = re.compile(r'^\s*(?:#{1,6}\s|[A-Z][A-Za-z /&-]{2,40}:?\s*$)')
RUBRIC_ROW = re.compile(r'^\s*(?:[-*+]\s*)?\d+\s*[:.)\-–—]\s*(?:strong|reject|accept|marginal|trivial|ok but|good paper)', re.I)
SUBLABEL = re.compile(r'(?:Rating|Score|Option(?:\s+chosen)?|Recommendation)\s*[:：]\s*', re.I)
LEAD = re.compile(r'^\s*(?:[-*+>]\s*)?(?:[^\w\s(]\s*)?(\d+(?:\.\d+)?)(?=\s*(?:$|[:.)\-–—(/]|\s+[*_]?[A-Za-z(]))')
PAREN = re.compile(r'\(\s*(\d+(?:\.\d+)?)\s*(?:/\s*10)?\s*\)')
SLASH = re.compile(r'\b(\d+(?:\.\d+)?)\s*/\s*10\b')


def _clean(line):
    line = line.replace('**', '').replace('__', '').replace('`', '').rstrip()
    # Markdown table row: | Recommendation | 8 - accept | -> Recommendation: 8 - accept
    if line.lstrip().startswith('|'):
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        for k, c in enumerate(cells[:-1]):
            if TABLE_FIELD.match(c):
                return 'Recommendation: ' + cells[k + 1]
    return line


def _prose_values(line):
    # Fallback for a value written as a sentence directly under the field, e.g. "I recommend acceptance
    # with a rating of 8": only an explicit number ("rating/score of N", "as N", "(N)") in the first sentence.
    first = re.split(r'(?<=[.!?])\s', line, maxsplit=1)[0][:300]
    return {float(x) for x in PROSE.findall(first)}


def _values(line):
    # Only the rating position (start of the value) is read; later enumerations like "(1) ... (2)" are ignored.
    line = line[:40]
    vals = set()
    # A sub-label inside the value ("Rating: 8", "Score: 8 (accept)", "REJECT (Rating: 5 ...)") points at the value.
    sub = SUBLABEL.search(line)
    if sub:
        line = line[sub.end():]
    m = LEAD.match(line)
    if m:
        vals.add(float(m[1]))
    # Parenthesised and /10 values only inside the first clause ("Reject. Reasons: (1) ..." is not a value).
    clause = re.split(r'[.:;]\s', line, maxsplit=1)[0]
    vals.update(float(x) for x in PAREN.findall(clause))
    vals.update(float(x) for x in SLASH.findall(clause))
    if vals:
        # Any other standalone integer near the start (e.g. "6 or 8") makes the value ambiguous.
        head = re.sub(r'/\s*10\b', '', line[:25])
        vals.update(float(x) for x in re.findall(r'(?<![\d.])(\d+)(?![\d.])', head))
    return vals


def score(text, item):
    base = v21_score(text, item)
    text = text if isinstance(text, str) else ''
    raw = text.splitlines()
    lines = [_clean(x) for x in raw]
    bold_heading = re.compile(r'^\s*(?:#{1,6}\s|\*\*[A-Za-z][^*\d]{1,59}\*\*\s*:?\s*$)')
    section_rating = re.compile(r'^\s*(?:[-*+]\s*)?(?:Rating|Score)\s*[:：]\s*(.+)$', re.I)
    # Inline field after a sentence end ("... processing. Recommendation: 10.").
    inline = [float(x) for x in re.findall(r'(?<=[.!?])\s+Recommendation\s*[:：]\s*(\d+(?:\.\d+)?)(?![\d.]?\d)', '\n'.join(lines))]
    found, evidence, rubric = [], [], False
    for i, line in enumerate(lines):
        m = FIELD.match(line)
        if not m:
            continue
        rest = (m[1] or m[2] or '').strip()
        section = []
        for j in range(i + 1, len(lines)):
            if FIELD.match(lines[j]) or bold_heading.match(raw[j]):
                break
            section.append(lines[j])
        if rest:
            value_line = rest
        else:
            nonempty = [x for x in section if x.strip() and not re.fullmatch(r'\s*[-=_*]{3,}\s*', x)]
            if not nonempty:
                continue
            value_line = nonempty[0]
            if sum(bool(RUBRIC_ROW.match(x)) for x in nonempty[:12]) >= 2:
                rubric = True
                continue
        vals, kind = _values(value_line), 'field_value'
        if not vals:
            vals, kind = _prose_values(value_line), 'prose_explicit_number'
        if not vals and not rest:
            # A single labelled "Rating: N" / "Score: N" line inside the Recommendation section.
            rows = [section_rating.match(x) for x in section]
            rows = [r for r in rows if r]
            if len(rows) == 1:
                vals, kind = _values(rows[0][1]), 'section_rating_line'
        if len(vals) == 1:
            found.append(vals.pop()); evidence.append({'field': line.strip(), 'value_line': value_line.strip(), 'kind': kind})
        elif len(vals) > 1:
            found.append(None); evidence.append({'field': line.strip(), 'value_line': value_line.strip(), 'ambiguous': True})
    for v in inline:
        found.append(v); evidence.append({'inline_field': v, 'kind': 'inline_after_sentence'})
    if base['parsed'] is not None:
        # Keep the earlier parser's value unless another explicit Recommendation field disagrees.
        if any(v is None or v != base['parsed'] for v in found):
            return {**base, 'parsed': None, 'recommendation': None, 'parse_status': 'v3_conflicts_with_v2_1_value',
                    'matches_reconstructed_allowed_anchor': None, 'recommends_accept_at_threshold6': None,
                    'diagnostic_version': '3_posthoc', 'diagnostic_method_v3': 'conflict', 'v3_evidence': evidence}
        return {**base, 'diagnostic_version': '3_posthoc', 'diagnostic_method_v3': 'unchanged_v2_1'}
    nums = {v for v in found if v is not None}
    ok = bool(found) and None not in found and len(nums) == 1 and next(iter(nums)).is_integer() and 1 <= next(iter(nums)) <= 10
    if not ok:
        status = ('v3_conflicting_or_ambiguous' if found else ('v3_rubric_echo_only' if rubric else base['parse_status']))
        return {**base, 'parse_status': status, 'diagnostic_version': '3_posthoc', 'diagnostic_method_v3': 'unrecovered',
                'v3_evidence': evidence}
    v = int(nums.pop())
    return {**base, 'parsed': v, 'recommendation': v, 'parse_status': 'v3_posthoc_explicit_field',
            'matches_reconstructed_allowed_anchor': v in ALLOWED_ANCHORS, 'recommends_accept_at_threshold6': v >= 6,
            'diagnostic_version': '3_posthoc', 'diagnostic_method_v3': 'heading_or_inline_explicit_value', 'v3_evidence': evidence}
