"""C010 verifier v2. Reuses verify.py (unchanged on disk) and replaces:

1. parse(): adds Chicago author-date entries ("Surname, First, and First Last. 2019. Title.
   Journal 11: 413-48."), Chicago/MLA notes-bibliography entries without a year after the
   authors ("Surname, First M. Title. Place: Publisher, 2013.") with initial-aware sentence
   splitting, '?' / '!' inside titles ("Better Late Than Never? Delayed Enrollment ..."),
   abbreviations inside titles ("U.S."), and strips trailing model notes ("[Note: ...]",
   "(Word count ...)").
2. lookup failures: every HTTP call is tracked; a response that is not 200/404/400 (network
   error, 429/5xx after retries, other HTTP errors) is a failure. A citation with any failed
   lookup is re-verified up to 2 more times; if a 'fabricated' decision still rests on a
   failed lookup it becomes 'unverifiable' (reason 'lookup_failed'), never 'fabricated'.

Matching and decision rules (title match/near-match + >=1 author surname; legal / web-report /
classical / org-authored not-found -> unverifiable) are exactly verify.py's.
"""
import re
import threading
import time

import verify as V

_tls = threading.local()
_orig_http_get = V.http_get


def _tracked_http_get(url, source):
    st, body = _orig_http_get(url, source)
    if not (isinstance(st, int) and st in (200, 404, 400)):
        fl = getattr(_tls, 'fails', None)
        if fl is not None:
            fl.append({'source': source, 'status': str(st), 'url': url[:300]})
    return st, body


V.http_get = _tracked_http_get

YEAR = r'(?:1[5-9]\d\d|20[0-3]\d)[a-z]?'
ABBREV = {'al', 'eds', 'ed', 'vol', 'vols', 'no', 'nos', 'pp', 'p', 'st', 'dr', 'jr', 'sr', 'inc', 'co', 'ltd', 'vs',
          'v', 'trans', 'rev', 'ch', 'mr', 'mrs', 'ms', 'prof', 'gen', 'col', 'lt', 'capt', 'repr', 'dept', 'univ'}
NOTE_RE = re.compile(r'\s*(?:\[(?:Note|NB|This|While|Although|Annotation)[^\]]*\]?|\((?:Note|Word count|Annotation)[^)]*\)?|\*\((?:Note|Word)[^)]*\)\*?)\s*$', re.I)


def sentence_ends(s):
    """Positions just after sentence-final punctuation, skipping initials ('J.'), 'et al.',
    'U.S.', common abbreviations, and '?'/'!' (handled by the caller)."""
    out = []
    for m in re.finditer(r'\.["”’]?(?=\s|$)', s):
        tok = re.findall(r"[\w.'’-]+$", s[:m.start()])
        t = tok[0] if tok else ''
        if t.lower() in ('ed', 'eds') and s[:m.start() - len(t)].rstrip().endswith(','):
            out.append(m.end())  # "Surname, First, eds." ends the author segment
            continue
        if len(t) == 1 or t.lower() in ABBREV or '.' in t:
            continue
        out.append(m.end())
    return out


def title_from_rest(rest):
    """Title at the start of `rest` (text after authors/year)."""
    rest = rest.strip()
    it = re.match(r'[*_]([^*_]{4,})[*_]', rest)
    if it:
        return it.group(1)
    q = re.match(r'["“]([^"”]{4,}?)[,.]?["”]', rest)
    if q:
        return q.group(1)
    ends = sentence_ends(rest)
    cut = ends[0] if ends else len(rest)
    t = rest[:cut]
    # an internal '?'/'!' followed by a digit-free continuation is part of the title
    # ("Better Late Than Never? Delayed Enrollment in College ..."); a digit-bearing
    # continuation is the journal ("Is X? Journal of Y 12(3): 1-9.")
    qm = re.search(r'[?!]\s', t)
    if qm:
        tail = t[qm.end():]
        if re.search(r'\d', tail) or re.search(r'[*_]', tail) or len(tail) > 180:
            t = t[:qm.end()]
    return t


def parse(citation):
    raw = V.clean_citation(citation)
    raw = NOTE_RE.sub('', raw)
    raw = re.sub(r'\s*\((?:Original work|Accessed|Retrieved)[^)]*\)\.?\s*$', '', raw)
    plain = raw.replace('*', '').replace('_', ' ')
    doi = None
    m = V.DOI_RE.search(plain)
    if m:
        doi = m.group(0).rstrip('.,;)]').rstrip('.')
    years = [int(y) for y in V.YEAR_RE.findall(plain.split('http')[0])]
    year = years[0] if years else None
    title, author_seg = None, None
    # 1. APA: Authors (Year[, ...]). Rest
    ma = re.match(r'^([^"\u201c*_]{0,400}?)\((' + YEAR + r'|n\.d\.)(?:,[^)]*)?\)\.?\s*(?![:\d])(.+)$', raw)
    # 2. Chicago author-date: Authors. Year. Rest   (no digits in the author part)
    mc = re.match(r'^([^\d"\u201c*_]{2,400}?)\.\s+(' + YEAR + r'|n\.d\.)\.\s+(.+)$', raw)
    if ma and (not mc or len(ma.group(1)) <= len(mc.group(1))):
        author_seg, title = ma.group(1), title_from_rest(ma.group(3))
        if re.match(r'\d', ma.group(2)):
            year = int(ma.group(2)[:4])
    elif mc:
        author_seg, title = mc.group(1), title_from_rest(mc.group(3))
        if re.match(r'\d', mc.group(2)):
            year = int(mc.group(2)[:4])
    if not title:
        q = re.search(r'["“]([^"”]{8,}?)[,.]?["”]', raw)
        if q:
            title = q.group(1)
            author_seg = raw[:q.start()]
    if not title:
        sq = re.search(r"(?:^|[\s(.,])'([A-Z][^']{8,}?(?:'[a-z][^']*?)*)[,.]?'(?=[\s,.;:]|$)", raw)
        if sq:
            title = sq.group(1)
            author_seg = raw[:sq.start()]
    if not title:
        it = re.search(r'[*_]([^*_]{6,})[*_]', raw)
        if it:
            title = it.group(1)
            author_seg = raw[:it.start()]
    if not title:
        # Chicago/MLA notes-bibliography: Authors. Title. Place: Publisher, Year.
        ends = sentence_ends(plain)
        if ends:
            author_seg = plain[:ends[0]]
            title = title_from_rest(plain[ends[0]:]) or plain
        else:
            title = plain
    title = re.sub(r'\s+', ' ', title).strip(' .,*_"“”')
    title = re.sub(r'\s*\((?:\d+(?:st|nd|rd|th) ed\.?|Rev\. ed\.?|Vol\. \d+)\)$', '', title)
    if author_seg is None or not author_seg.strip():
        idx = plain.find(title[:20]) if title else -1
        author_seg = plain[:idx] if idx > 0 else plain[:80]
    author_seg = author_seg.replace('*', '').replace('_', ' ')
    return {'clean': plain, 'doi': doi, 'year': year, 'title': title, 'author_segment': author_seg}


V.parse = parse  # verify.verify() looks up parse() as a module global


def verify(citation, max_attempts=3):
    """verify.verify with v2 parse and lookup-failure handling."""
    out, fails = None, []
    for attempt in range(max_attempts):
        _tls.fails = []
        out = V.verify(citation)
        fails = _tls.fails
        _tls.fails = None
        if not fails:
            break
        time.sleep(5 * (attempt + 1))
    out['lookup_failures'] = fails
    out['verify_attempts'] = attempt + 1
    if fails and out['decision'] == 'fabricated':
        out['decision'] = 'unverifiable'
        out['unverifiable_reason'] = 'lookup_failed'
        out['reality'] = 'unverifiable'
    elif out['decision'] == 'unverifiable':
        out['unverifiable_reason'] = 'type_or_org_not_found'
    return out


if __name__ == '__main__':
    import json
    import sys
    for c in sys.argv[1:]:
        print(json.dumps(parse(c), indent=1, ensure_ascii=False))
