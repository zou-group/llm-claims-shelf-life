"""Automated extraction of the FIRST completed bibliography and its entries (replaces the
prior manual boundary review; validated against those approved boundaries).

Rule (paper: evaluate only the initial bibliography, exclude anything after it):
start = first line that is a bibliography heading (References / Bibliography / Works Cited /
Reference List / Sources, optionally markdown-decorated); end = the first subsequent
markdown heading, horizontal rule followed by prose, or a non-entry prose paragraph after at
least one entry. Entries are non-empty lines (bullets/numbering stripped); a line that looks
like a continuation of the previous entry (starts lowercase / with a digit / with a URL / DOI)
is joined to it.
"""
import re

HEAD = re.compile(r'(?im)^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*|__)?[ \t]*(?:(?:\d+|[IVXLC]+)\.?\s*)?\(?(?:bibliography|references|works cited|reference list|references cited|sources cited|selected bibliography)\)?[ \t]*(?:\*\*|__)?[ \t]*(?:\([^)\n]{0,40}\))?[ \t]*:?[ \t]*(?:\*\*|__)?[ \t]*$')
NEXT_HEAD = re.compile(r'^\s*(#{1,6}\s|\*\*[^*]{2,80}\*\*\s*:?\s*$|---+\s*$|___+\s*$|\*\*\*+\s*$)')
YEARISH = re.compile(r'\b(1[5-9]\d\d|20[0-3]\d|n\.d\.)\b')


INTEXT_END = re.compile(r'\([A-Z][^()]{1,80}?,?\s(1[5-9]\d\d|20\d\d)[a-z]?(?:,\s*p+\.\s*[\d-]+)?\)\.?\s*$')
PROSE = re.compile(r"\b(please|verify|I have|I've|note that|here is|here are|you will|you should|you may|would add|a stronger|consider adding|continued\)?:|word count|words\)|all urls|accessed \d)\b", re.I)


def looks_entry(line):
    s = line.strip()
    bold_author = re.match(r"^\*\*[^*]{0,40}?[A-Z][\w'\u00C0-\u017F-]+,\s*[A-Z]", s) is not None
    if re.match(r'^(doi:|https?://|10\.\d)', s, re.I):
        return False
    if INTEXT_END.search(s) or len(s.split('. ')[0].split()) > 30:
        return False  # body prose (ends with an in-text citation / long first sentence), not an entry
    if len(s) < 25 or (s.startswith('**') and not bold_author) or s.startswith('>') or PROSE.search(s[:120]):
        return False
    return YEARISH.search(s) is not None or re.search(r'\b(Press|University|Publisher|Journal|Routledge)\b', s) is not None


def _key(e):
    return re.sub(r'[^a-z0-9]+', '', e.lower())


def _merge(entries, new):
    """Restarted / continued lists after a truncated turn: a fragment that is a prefix of a
    later entry (or the same entry repeated) is replaced, never double-counted."""
    k = _key(new)
    for j, old in enumerate(entries):
        ko = _key(old)
        if k.startswith(ko[:max(30, len(ko) - 5)]) or ko.startswith(k) or (len(k) > 60 and k[:60] == ko[:60]):
            if len(new) > len(old):
                entries[j] = new
            return
    entries.append(new)


def extract(text):
    m = HEAD.search(text)
    if not m:
        return {'has_bibliography': False, 'entries': [], 'start': None, 'end': None}
    start = m.end()
    lines = text[start:].split('\n')
    offs, pos = [], start
    for line in lines:
        offs.append(pos)
        pos += len(line) + 1
    clean = [re.sub(r'^(?:[-*\u2022]|\d{1,3}[.)]|\[\d{1,3}\])\s+', '', l.strip()) for l in lines]
    entries, end, seen = [], None, False
    for i, body in enumerate(clean):
        if not body:
            continue
        if looks_entry(body) and not NEXT_HEAD.match(body):
            _merge(entries, body) if seen else entries.append(body)
            seen = True
            continue
        if not seen:
            continue  # decoration / sub-heading before the first entry
        if re.match(r'^(https?://|doi:|10\.\d|[a-z(\d]|\*[^*]{3,80}\*?,?\s*\d)', body, re.I) and not re.match(r'^\(?\*?\(?word', body, re.I) and not NEXT_HEAD.match(body) and len(body) < 200:
            entries[-1] += ' ' + body  # wrapped continuation of previous entry
            continue
        nxt = [b for b in clean[i + 1:] if b][:3]
        if any(looks_entry(b) and not NEXT_HEAD.match(b) for b in nxt):
            continue  # sub-heading / "References (continued)" / restart remark inside the list
        end = offs[i]
        break
    return {'has_bibliography': bool(entries), 'entries': entries, 'start': start,
            'end': end if end is not None else len(text)}
