"""Automated bibliographic verifier for C010 (replaces the paper's manual verification).

For each bibliography entry: search Crossref (bibliographic query + any cited DOI), OpenAlex,
Open Library (books) and Semantic Scholar (title match, fallback), then apply the paper's
reality rule: a work is REAL if a catalogue record matches its title (match or near-match)
AND its authors (at least one matching surname); otherwise it is NOT FOUND (fabricated
candidate). Legal authorities, statutes, web pages and organisational documents are
poorly covered by these catalogues; a not-found entry of that kind is labelled
'unverifiable' (reported as an unknown bound, never silently counted as fabricated).

Among REAL works, a coarse automated error check flags date / journal / volume-pages /
author mismatches against the matched record ('exists_with_errors').

Every HTTP response is cached under cache/ (keyed by URL hash) so reruns are free and
auditable. Free public APIs only; polite per-host rate limits; no user e-mail is sent.
"""
import hashlib
import json
import os
import re
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / 'cache'
UA = 'C010-citation-verifier/1.0 (academic reproduction study; low-rate)'
INTERVAL = {'api.crossref.org': 0.25, 'api.openalex.org': 0.15, 'openlibrary.org': 1.0,
            'api.semanticscholar.org': 1.2}
_locks = {h: threading.Lock() for h in INTERVAL}
_last = {h: 0.0 for h in INTERVAL}


def http_get(url, source):
    """GET with on-disk cache, per-host spacing and bounded retries. Returns (status, json|None)."""
    key = hashlib.sha256(url.encode()).hexdigest()
    path = CACHE / source / key[:2] / f'{key}.json'
    if path.exists():
        rec = json.loads(path.read_text())
        return rec['status'], rec['body']
    host = urllib.parse.urlparse(url).hostname
    status, body = None, None
    for attempt in range(6):
        with _locks[host]:
            wait = _last[host] + INTERVAL[host] - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            _last[host] = time.monotonic()
        try:
            headers = {'User-Agent': UA, 'Accept': 'application/json'}
            if host == 'api.semanticscholar.org' and os.environ.get('SEMANTIC_SCHOLAR_API_KEY'):
                headers['x-api-key'] = os.environ['SEMANTIC_SCHOLAR_API_KEY']
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=40) as r:
                status, body = r.status, json.loads(r.read().decode('utf-8', 'replace'))
            break
        except urllib.error.HTTPError as e:
            status = e.code
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(min(60, 3 * 2 ** attempt))
                continue
            body = None
            break
        except Exception as e:  # network error: retry, then give up without caching
            status = f'error:{type(e).__name__}'
            time.sleep(min(60, 3 * 2 ** attempt))
    if isinstance(status, int) and status in (200, 404, 400):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'url': url, 'status': status, 'retrieved': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'body': body}))
    return status, body


# ---------------- text normalisation and parsing ----------------
def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))


def norm(s):
    s = strip_accents(s or '').lower().replace('&', ' and ')
    s = re.sub(r'<[^>]+>', ' ', s)  # crossref jats tags
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def clean_citation(c):
    c = re.sub(r'\[([^\]]*)\]\((https?://[^)]+)\)', r'\1 \2', c)  # markdown links
    c = c.replace('\\*', '*').replace('**', '')
    c = re.sub(r'^\s*(?:[-*•]|\d{1,3}[.)]|\[\d{1,3}\])\s+', '', c)
    c = c.replace('–', '-').replace('—', '-').replace('’', "'").replace('“', '"').replace('”', '"')
    return re.sub(r'\s+', ' ', c).strip()


DOI_RE = re.compile(r'10\.\d{4,9}/[^\s"<>]+', re.I)
YEAR_RE = re.compile(r'\b(1[5-9]\d\d|20[0-3]\d)[a-z]?\b')


def parse(citation):
    raw = clean_citation(citation)
    plain = raw.replace('*', '').replace('_', ' ')
    doi = None
    m = DOI_RE.search(plain)
    if m:
        doi = m.group(0).rstrip('.,;)]').rstrip('.')
    years = [int(y) for y in YEAR_RE.findall(plain.split('http')[0])]
    year = years[0] if years else None
    title = None
    q = re.search(r'["“]([^"”]{8,}?)[,.]?["”]', raw)
    if q:
        title = q.group(1)
    if not title:
        sq = re.search(r"(?:^|[\s(.,])'([A-Z][^']{8,}?(?:'[a-z][^']*?)*)[,.]?'(?=[\s,.;:]|$)", raw)
        if sq:
            title = sq.group(1)
    if not title:
        # APA: Authors (Year). Title. Rest  -- title may end with ? or !
        m = re.search(r'\((?:\d{4}[a-z]?|n\.d\.)(?:,[^)]*)?\)\.?\s*(.+)$', raw)
        if m:
            rest = m.group(1)
            it = re.match(r'\*([^*]{6,})\*', rest)
            if it:
                title = it.group(1)
            else:
                t = re.match(r'(.+?[.?!])(?:\s|$)', rest)
                title = (t.group(1) if t else rest)
    if not title:
        it = re.search(r'\*([^*]{6,})\*', raw)
        if it:
            title = it.group(1)
    if not title:
        # Chicago/MLA books: Author. Title. Place: Publisher, Year.
        parts = re.split(r'(?<=[a-z0-9)])\.\s+', plain)
        title = parts[1] if len(parts) > 1 else plain
    title = title.strip(' .*_"')
    # author segment: text before the year parenthesis or before the title
    idx = plain.find('(' + str(year)) if year else -1
    if idx < 0 and title:
        idx = plain.find(title[:20])
    author_seg = plain[:idx] if idx > 0 else plain[:80]
    return {'clean': plain, 'doi': doi, 'year': year, 'title': title, 'author_segment': author_seg}


def ratio(a, b):
    return SequenceMatcher(None, a, b).ratio() if a and b else 0.0


def title_score(cand_title, parsed):
    """(strong, main) similarity. strong: full candidate title vs cited title (or verbatim
    containment in the citation). main: main title (before ':') comparisons, which tolerate
    missing/variant subtitles and therefore need corroboration (year) in evaluate()."""
    ct = norm(cand_title)
    if len(ct) < 4:
        return 0.0, 0.0
    cmain = norm(re.split(r'[:?.]\s', cand_title + ' ')[0])
    pt = norm(parsed['title'])
    pmain = norm(re.split(r'[:?.]\s', parsed['title'] + ' ')[0])
    full = ' ' + norm(parsed['clean']) + ' '
    strong = ratio(ct, pt)
    if ((len(ct) >= 18 and len(ct) >= 0.7 * len(pt)) or len(ct) >= 25) and (' ' + ct + ' ') in full:
        strong = 1.0
    main = max(ratio(cmain, pmain), ratio(ct, pmain)) if len(pmain) >= 4 else 0.0
    return strong, main


def author_match(cand_authors, parsed):
    """(n candidate surnames found in the citation's author segment, n candidate surnames)."""
    seg = ' ' + norm(parsed['author_segment'] or parsed['clean'][:100]) + ' '
    fams = [norm(a) for a in cand_authors if a and norm(a)]
    hits = 0
    for f in fams:
        toks = [t for t in f.split() if len(t) > 1]
        if toks and any((' ' + t + ' ') in seg for t in toks[-2:]):
            hits += 1
        elif toks and len(toks[-1]) >= 5 and any(len(w) >= 5 and ratio(toks[-1], w) >= 0.85 for w in seg.split()):
            hits += 1  # transliteration / typo variant of the surname
    return hits, len(fams)


# ---------------- catalogue adapters -> candidate dicts ----------------
def cand_crossref(item):
    return {'source': 'crossref', 'title': ' '.join(item.get('title') or []),
            'subtitle': ' '.join(item.get('subtitle') or []),
            'authors': [a.get('family') or a.get('name') or '' for a in item.get('author', []) + item.get('editor', [])],
            'given': [a.get('given') or '' for a in item.get('author', [])],
            'year': ((item.get('issued') or {}).get('date-parts') or [[None]])[0][0],
            'container': ' '.join(item.get('container-title') or []), 'volume': item.get('volume'),
            'page': item.get('page'), 'doi': item.get('DOI'), 'type': item.get('type')}


def cand_openalex(w):
    loc = (w.get('primary_location') or {}).get('source') or {}
    return {'source': 'openalex', 'title': w.get('display_name') or '', 'subtitle': '',
            'authors': [(a.get('author') or {}).get('display_name', '').split(' ')[-1] for a in w.get('authorships', [])],
            'authors_full': [(a.get('author') or {}).get('display_name', '') for a in w.get('authorships', [])],
            'year': w.get('publication_year'), 'container': loc.get('display_name') or '',
            'volume': (w.get('biblio') or {}).get('volume'), 'page': (w.get('biblio') or {}).get('first_page'),
            'doi': (w.get('doi') or '').replace('https://doi.org/', '') or None, 'type': w.get('type')}


def cand_openlibrary(d):
    return {'source': 'openlibrary', 'title': d.get('title') or '', 'subtitle': d.get('subtitle') or '',
            'authors': d.get('author_name') or [], 'year': d.get('first_publish_year'), 'container': '',
            'volume': None, 'page': None, 'doi': None, 'type': 'book', 'publisher': (d.get('publisher') or [])[:5]}


def cand_s2(p):
    return {'source': 'semanticscholar', 'title': p.get('title') or '', 'subtitle': '',
            'authors': [(a.get('name') or '').split(' ')[-1] for a in p.get('authors') or []],
            'year': p.get('year'), 'container': p.get('venue') or '', 'volume': None, 'page': None,
            'doi': (p.get('externalIds') or {}).get('DOI'), 'type': None}


def q(s):
    return urllib.parse.quote(s, safe='')


def search_crossref(parsed):
    out = []
    st, body = http_get(f"https://api.crossref.org/works?query.bibliographic={q(parsed['clean'][:400])}&rows=6", 'crossref')
    if st == 200 and body:
        out += [cand_crossref(i) for i in body['message']['items']]
    return out


def search_crossref_title(parsed):
    a = first_surname(parsed)
    url = f"https://api.crossref.org/works?query.title={q(parsed['title'][:250])}&rows=6"
    if a:
        url += f'&query.author={q(a)}'
    st, body = http_get(url, 'crossref')
    return [cand_crossref(i) for i in body['message']['items']] if st == 200 and body else []


def lookup_doi(doi):
    st, body = http_get(f'https://api.crossref.org/works/{q(doi)}', 'crossref')
    if st == 200 and body:
        return [cand_crossref(body['message'])]
    return []


def search_openalex(parsed):
    t = re.sub(r'[,:;?!|()\[\]{}"*]', ' ', parsed['title'])[:250]
    st, body = http_get(f"https://api.openalex.org/works?search={q(t)}&per-page=8&select=display_name,authorships,publication_year,primary_location,biblio,doi,type", 'openalex')
    out = []
    if st == 200 and body:
        out += [cand_openalex(w) for w in body.get('results', [])]
    return out


def first_surname(parsed):
    seg = parsed['author_segment'] or ''
    m = re.match(r"\s*([A-Z][A-Za-z'\-À-ſ]+(?:\s(?:de|van|von|der|la|le)\s?[A-Z][A-Za-z'\-]+)?)", strip_accents(seg))
    return m.group(1) if m else ''


def _ol(url):
    st, body = http_get(url, 'openlibrary')
    return [cand_openlibrary(d) for d in body.get('docs', [])] if st == 200 and body else []


OL_FIELDS = '&limit=8&fields=title,subtitle,author_name,first_publish_year,publisher'


def openlibrary_queries(parsed):
    """Ordered Open Library queries (run lazily until a match is found): title+author, then
    free-text main title + surname, then title only."""
    t = re.split(r'[:?]', parsed['title'])[0][:150]
    a = first_surname(parsed)
    qs = []
    if a:
        qs.append(f"https://openlibrary.org/search.json?title={q(t)}&author={q(a)}" + OL_FIELDS)
        qs.append(f"https://openlibrary.org/search.json?q={q(t + ' ' + a)}" + OL_FIELDS)
    qs.append(f"https://openlibrary.org/search.json?title={q(t)}" + OL_FIELDS)
    return qs


def search_s2(parsed):
    st, body = http_get(f"https://api.semanticscholar.org/graph/v1/paper/search/match?query={q(parsed['title'][:250])}&fields=title,authors,year,venue,externalIds", 's2')
    if st == 200 and body:
        return [cand_s2(p) for p in body.get('data', [])]
    return []


# ---------------- type heuristics ----------------
LEGAL = re.compile(r'\b(v\.|vs\.?|Act\s(?:of\s)?\d{4}|Act \d{4}|\bAct\b,?\s\d{4}|U\.S\.C\.|\bC\.F\.R\b|Stat\.|Regulation \(|Directive|Treaty|Convention on|AC \d|QB \d|WLR|All ER|EWCA|UKSC|UKHL|F\.\dd|S\. ?Ct\.|U\.S\. \d|Code\b|Bill\b|Hansard|Ch \d)', re.I)
WEB = re.compile(r'https?://|www\.|Retrieved from|Accessed|\[Online\]|\.gov\b|\.org/', re.I)
ARTICLE = re.compile(r'\b\d+\s*\(\d+[^)]*\)\s*,?\s*(?:pp?\.?\s*)?\d+|\bvol\.?\s*\d+|\b\d+\s*:\s*\d+\s*-\s*\d+|\b\d+,\s*\d+\s*-\s*\d+', re.I)


def guess_type(parsed):
    c = parsed['clean']
    if re.search(r'\b(century|BCE|B\.C\.|c\. \d{3}\b)', c) and not parsed['year']:
        return 'classical'
    if parsed['year'] and parsed['year'] < 1700:
        return 'classical'
    if re.search(r'\b(Translated by|trans\.|Loeb Classical)', c, re.I):
        return 'classical'  # translations / editions of older works: catalogue titles vary too much
    if LEGAL.search(c):
        return 'legal'
    if re.search(r'\bIn\s+[A-Z][^.]*\(Eds?\.?\)|\(Eds?\.\)|\bEd(?:s|itors?)?\.?,', c):
        return 'chapter'
    if ARTICLE.search(c):
        return 'article'
    if WEB.search(c) and not parsed['doi']:
        return 'web_or_report'
    if re.search(r'\b(Report|Working Paper|Briefing|Policy Brief|Factsheet|Fact sheet|Statistics|Survey|Guidance|White Paper|Discussion Paper|Study \d{4}|Outlook|Inventory|Yearbook|Review of)\b', c):
        return 'web_or_report'
    return 'book'


ORG_AUTHOR = re.compile(r'\b(Organization|Organisation|Association|Institute|Agency|Department|Ministry|Office|Commission|Council|Committee|Bank|Fund|Agency|Centers?|Society|Foundation|Government|Parliament|Nations|Union|Service|Authority|Board|Bureau|Court|Programme|Program|Forum|Panel|Group)\b')


# ---------------- verification ----------------
def evaluate(cands, parsed, ctype='book'):
    """Best candidate under the paper's title+author (near-)match rule."""
    best = None
    seg = (parsed['author_segment'] or '').strip()
    org = bool(ORG_AUTHOR.search(seg)) or bool(re.match(r'^[A-Z]{2,6}\b', seg)) or (
        bool(seg) and not re.search(r'\b[A-Z]\.', seg) and len(seg.split()) <= 8 and not re.search(r',', seg))
    pmain_len = len(norm(re.split(r'[:?.]\s', parsed['title'] + ' ')[0]))
    for c in cands:
        title = c['title'] + ((': ' + c['subtitle']) if c.get('subtitle') else '')
        s1, m1 = title_score(title, parsed)
        s2, m2 = title_score(c['title'], parsed)
        strong, main = max(s1, s2), max(m1, m2)
        hits, _ = author_match(c['authors'] + c.get('given', []), parsed)
        n = len([a for a in c['authors'] if a])
        hits = min(hits, n)
        ydiff = abs(int(c['year']) - parsed['year']) if c.get('year') and parsed.get('year') else None
        year_close = ydiff is None or ydiff <= 3
        auth_ok = hits > 0 or (org and (n == 0 or strong >= 0.9 or (main >= 0.95 and year_close)))
        if auth_ok and strong >= 0.90:
            level = 2
        elif auth_ok and strong >= 0.80 and year_close:
            level = 2  # near-match title (wording variant), same period
        elif auth_ok and main >= 0.95 and ((pmain_len >= 15 and ctype != 'article') or year_close):
            level = 2  # main title matches, subtitle missing/variant in catalogue or citation
        elif strong >= 0.95 and n == 0 and ydiff is not None and ydiff <= 2:
            level = 1  # exact title, catalogue lacks authors, same period
        else:
            level = 0
        ts = max(strong, main)
        score = (level, ts, hits)
        if best is None or score > best[0]:
            best = (score, c, {'title_score': round(strong, 3), 'main_title_score': round(main, 3), 'author_hits': hits,
                               'cand_authors': n, 'year_diff': ydiff})
    return best


def error_flags(c, parsed, ctype):
    """Coarse automated substantive-error flags against the matched record (secondary metric)."""
    flags = {}
    if c.get('year') and parsed.get('year'):
        tol = 0 if ctype == 'article' else 1
        flags['date'] = abs(int(c['year']) - parsed['year']) > tol if ctype == 'article' else None
    if ctype == 'article' and c.get('container'):
        cont = norm(c['container'])
        flags['journal'] = not (cont and (' ' + cont + ' ') in (' ' + norm(parsed['clean']) + ' ')) and ratio(cont, norm(parsed['clean'])) < 0.9
    if ctype == 'article' and c.get('volume'):
        v = str(c['volume']).strip()
        flags['volume'] = not re.search(r'\b' + re.escape(v) + r'\b', parsed['clean'])
    if c['source'] == 'crossref' and c['authors']:
        hits, n = author_match(c['authors'], parsed)
        flags['authorship'] = hits < min(n, 3)
    return flags


def verify(citation):
    parsed = parse(citation)
    ctype = guess_type(parsed)
    cands, trail = [], []
    if parsed['doi']:
        cands += lookup_doi(parsed['doi']); trail.append('doi')
    cands += search_crossref(parsed); trail.append('crossref')
    best = evaluate(cands, parsed, ctype)
    if not best or best[0][0] < 2:
        cands += search_crossref_title(parsed); trail.append('crossref_title')
        best = evaluate(cands, parsed, ctype)
    if (not best or best[0][0] < 2) and ctype in ('book', 'chapter', 'web_or_report', 'legal'):
        for url in openlibrary_queries(parsed):
            cands += _ol(url); trail.append('openlibrary')
            best = evaluate(cands, parsed, ctype)
            if best and best[0][0] >= 2:
                break
    if not best or best[0][0] < 2:
        cands += search_s2(parsed); trail.append('s2')
        best = evaluate(cands, parsed, ctype)
    level = best[0][0] if best else 0
    out = {'parsed': parsed, 'type_guess': ctype, 'sources_tried': trail, 'n_candidates': len(cands)}
    if level >= 1:
        c = best[1]
        flags = error_flags(c, parsed, ctype)
        out.update(decision='exists_with_errors' if any(v for v in flags.values() if v) else 'exists',
                   match_level=level, match=c, scores=best[2], error_flags=flags)
        if parsed['doi'] and c.get('doi') and c['doi'].lower() != parsed['doi'].lower():
            out['doi_mismatch'] = True
    else:
        seg = (parsed['author_segment'] or '').strip()
        org = bool(ORG_AUTHOR.search(seg)) or bool(re.match(r'^[A-Z]{2,6}\b', seg))
        out['org_author'] = org
        out.update(decision='unverifiable' if (ctype in ('legal', 'web_or_report', 'classical') or org) else 'fabricated',
                   match_level=0, best_near=best[1] if best else None, scores=best[2] if best else None)
    out['reality'] = {'exists': 'real', 'exists_with_errors': 'real'}.get(out['decision'], out['decision'])
    return out


if __name__ == '__main__':
    import sys
    for c in sys.argv[1:]:
        print(json.dumps(verify(c), indent=1, default=str)[:3000])
