"""C010 extraction v2. Wraps the v1 first-bibliography extractor (extract.py,
unchanged) and post-processes its entries:

1. split merged double entries: a v1 entry that contains a second complete entry start
   ("... Verso. Shakespeare's Word. (n.d.). ..." or "... Press. Smith, J. (2010). ...") is
   split at that boundary;
2. drop body-text prose that v1 kept as an entry (paragraphs written after / inside the
   bibliography): no author-style opening AND a long first sentence, or a sentence containing an
   in-text citation "Author (2015) found ...".

`extract_v2(text)` returns the v1 dict plus 'entries' (cleaned), 'dropped_prose' and
'split_from' (list of (v1_index, n_parts)) for per-model accounting.
"""
import re

from extract import extract as extract_v1

YEAR_PAREN = r'\((?:1[5-9]\d\d|20[0-3]\d|n\.d\.)[a-z]?(?:,[^)]{0,30})?\)\.?'
# a new entry starting after a sentence end: "Surname, X. ... (2010)." / "Org Name. (n.d.)." /
# "Surname, First. 2010." ; looked for only after position 40 in an entry
NEW_ENTRY = re.compile(
    r'(?<=[.!?*)\]])\s+(?=\**(?:'
    r"[A-Z][\w'’À-ſ-]+,\s(?:[A-Z]\.|[A-Z][a-z]+)[^()]{0,160}?" + YEAR_PAREN +
    r"|[A-Z][^().:]{1,80}?\.\s" + YEAR_PAREN +
    r'))')
AUTHOR_START = re.compile(
    r"^\**\s*(?:"
    r"[A-ZÀ-ſ][\w'’À-ſ-]+(?:\s(?:[A-Z][\w'’-]+|de|van|von|der|da|del|la|le))*,\s"  # Surname, ...
    r"|[A-Z][\w'’.-]+(?:\s[A-Z][\w'’.-]+){0,3}\.?\s\((?:1[5-9]\d\d|20[0-3]\d|n\.d\.)"  # Name (2010)
    r"|[A-Z][^.]{0,120}?\((?:1[5-9]\d\d|20[0-3]\d|n\.d\.)[a-z]?\)"  # anything (2010) early
    r"|[A-Z][^.]{0,120}?\.\s(?:1[5-9]\d\d|20[0-3]\d)[a-z]?\.\s"  # Org. 2010.
    r"|[A-Z][\w.&'’-]*(?:\s[A-Z&][\w.&'’-]*){0,10}\.\s[\"“*_A-Z]"  # Org/author. Title
    r")")
INTEXT_SENT = re.compile(r"\b[A-Z][a-z]+(?:\s(?:et al\.|and [A-Z][a-z]+))?\s\((?:19|20)\d\d[a-z]?\)\s(?:[a-z]+\s){0,2}(?:found|show|showed|shows|argue|argued|argues|propose|proposed|proposes|suggest|suggested|suggests|report|reported|reports|estimate|estimated|estimates|document|documented|demonstrate|demonstrated|note|noted|observe|observed)\b")


ABBREV = {'al', 'eds', 'ed', 'vol', 'no', 'pp', 'p', 'st', 'dr', 'jr', 'sr', 'inc', 'co', 'ltd', 'vs', 'v', 'trans', 'rev', 'ch'}
PROSE_MARK = re.compile(r'(?<=[A-Za-z.]\s)\((?:1[5-9]|20)\d\d[a-z]?\)(?:\s*[,;]|\s[a-z])')  # "(2018) show" / "(2018)," = in-text use


def first_sentence(s):
    """Text up to the first sentence end, not counting initials ('J.'), 'et al.', 'U.S.' etc."""
    for m in re.finditer(r'[.!?](?=\s|$)', s):
        tok = re.findall(r"[\w.'\u2019-]+$", s[:m.start()])
        t = tok[0] if tok else ''
        if m.group(0) == '.' and (len(t) == 1 or t.lower() in ABBREV or '.' in t):
            continue
        return s[:m.end()]
    return s


def is_prose(e):
    s = re.sub(r'^\**', '', e.strip())
    st = re.search(r'\((?:1[5-9]|20)\d\d[a-z]?\)\.|\.\s(?:1[5-9]|20)\d\d[a-z]?\.\s', s[:300])
    pm = PROSE_MARK.search(s)
    if st and (not pm or st.start() < pm.start()):
        return False  # APA / author-date entry start before any in-text citation
    if len(s.split()) < 20:
        return False
    if PROSE_MARK.search(s) or INTEXT_SENT.search(s):
        return True  # uses in-text citations ("Krueger (2015) propose", "(2018),")
    return len(first_sentence(s).split()) >= 12 and not AUTHOR_START.match(s)


def split_merged(e):
    parts, last = [], 0
    for m in NEW_ENTRY.finditer(e):
        left = e[last:m.start()].strip()
        # the left part must itself be a complete entry: >= 8 words, contains a year / n.d.,
        # and is not an author list cut at an ellipsis or 'et al.'
        if (m.start() - last < 40 or len(left.split()) < 8 or not re.search(r'\b(1[5-9]\d\d|20[0-3]\d|n\.d\.)', left)
                or re.search(r'(\.\.\.|\u2026|et al\.)$', left)):
            continue
        parts.append(e[last:m.start()].strip())
        last = m.end()
    parts.append(e[last:].strip())
    return [p for p in parts if p]


def extract_v2(text):
    r = extract_v1(text)
    out, dropped, split_from = [], [], []
    for i, e in enumerate(r['entries']):
        parts = split_merged(e)
        if len(parts) > 1:
            split_from.append((i, len(parts)))
        for p in parts:
            if is_prose(p):
                dropped.append(p)
            else:
                out.append(p)
    return {**r, 'entries': out, 'has_bibliography': bool(out), 'dropped_prose': dropped, 'split_from': split_from,
            'v1_entries': r['entries']}
