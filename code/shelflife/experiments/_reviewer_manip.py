"""Shared machinery for the LLM-reviewer manipulation pilot claims (C144-C160, 2026-10-12).

Source: the claim records. Each claim module declares its verbatim
prompts (code/shelflife/data/sources/CNNN_*/prompts.json), its conditions and its score parser; this module provides

* the shared manuscript pool: the 100 ReviewCritique papers already used by C013 (50 accepted / 50 rejected
  ICLR/NeurIPS submissions with extracted body text), restricted to the 97 whose abstract and introduction can be
  located, interleaved accepted/rejected in file order so that any prefix is balanced; bodies truncated to a fixed
  cl100k token budget (default 4,000) to fit the pilot budget;
* injection placement on extracted text (PDF white text becomes ordinary text after PDF-to-text conversion, which is
  what a text-input LLM reviewer sees): 'start' (before the title / top of page 1), 'first_page' (first line
  break or sentence end after ~3,000 characters), 'end' (after the truncated body, standing in for the last page);
* a paired aggregate: per condition means of each parsed metric, and per treatment the mean paired difference vs
  the control condition with a paper-cluster bootstrap CI, counting missing and unparsed rows.

Common deviations for every pilot claim (listed again in each variant):
  * ReviewCritique papers replace each paper's own manuscript set (ICLR 2024/2025, medical, economics...);
  * text truncated to the first N cl100k tokens; injected text inserted into the extracted text instead of a PDF;
  * one review per paper and condition; output cap 4,096 tokens (16,384 for reasoning models) instead of the
    papers' settings; provider-default sampling where the paper used a web UI or did not state one.
"""
import functools
import hashlib
import json
import re
from statistics import mean

import tiktoken

from .. import stats
from ..paths import DATA, INITIAL
from ..program import Program
from ..prompts import PromptTemplate, Source
from ..registry import Experiment, Variant
from . import _robust

REVIEW_DATA = INITIAL / 'claims/C014_llm-reviews-nondiscriminating/code/ReviewCritique/data/ReviewCritique.jsonl'
SOURCES = DATA / 'sources'
REVIEW_CAP = 4096
MIN_CAP_REASONING = 16384
COMMON_DEVIATIONS = (
    'ReviewCritique papers (C013 pool, ICLR/NeurIPS submissions) replace the paper\'s own manuscript set',
    'manuscript body truncated to the first N cl100k tokens; injection inserted into extracted text, not a PDF',
    'one review per paper and condition (pilot, 30 papers by default)',
    'output cap 4096 tokens, 16384 for reasoning models; provider-default sampling unless the paper states otherwise',
)

_ABS = re.compile(r'A\s?BSTRACT|Abstract')
_INTRO = re.compile(r'\n\s*1\s+I\s?NTRODUCTION|\n\s*1\s+Introduction|\n\s*I\s?NTRODUCTION\b')
_SEC2 = re.compile(r'\n\s*2\s+[A-Z][A-Z ]{2,}|\n\s*2\s+[A-Z][a-z]')


@functools.lru_cache(None)
def encoder():
    return tiktoken.get_encoding('cl100k_base')


def truncate(text, n_tokens):
    tokens = encoder().encode(text, disallowed_special=())
    return encoder().decode(tokens[:n_tokens])


def split_sections(body):
    """(abstract, introduction) located by headings, or None if either is missing."""
    a = _ABS.search(body)
    i = _INTRO.search(body)
    if not (a and i and a.end() < i.start()):
        return None
    s2 = _SEC2.search(body, i.end())
    if not s2:
        return None
    return body[a.end():i.start()].strip(), body[i.start():s2.start()].strip()


@functools.lru_cache(None)
def pool():
    """The 97 parsable ReviewCritique papers, interleaved accepted/rejected in file order."""
    rows = [json.loads(x) for x in REVIEW_DATA.read_text().splitlines() if x.strip()]
    assert len(rows) == 100
    papers = {'accepted': [], 'rejected': []}
    for r in rows:
        sec = split_sections(r['body_text'])
        if sec is None:
            continue
        papers[r['decision']].append({
            'base_id': hashlib.sha256(r['title'].encode()).hexdigest()[:16], 'title': r['title'].strip(),
            'decision': r['decision'], 'body': r['body_text'], 'abstract': sec[0], 'introduction': sec[1]})
    out = []
    for acc, rej in zip(papers['accepted'], papers['rejected']):
        out += [acc, rej]
    return tuple(out)


def place(text, injection, where):
    """Insert ``injection`` into ``text`` at 'start', 'first_page' or 'end' (separated by blank lines)."""
    if injection is None:
        return text
    if where == 'start':
        return injection + '\n\n' + text
    if where == 'end':
        return text + '\n\n' + injection
    if where == 'first_page':
        # extracted text has long unbroken lines: first line break or sentence end after ~3,000 chars (~1 page)
        ends = [i for i in (text.find('\n', 3000), text.find('. ', 3000)) if i >= 0]
        cut = min(ends) + 1 if ends else len(text)
        return text[:cut] + '\n\n' + injection + '\n\n' + text[cut:]
    raise ValueError(where)


def load_prompts(folder):
    return json.loads((SOURCES / folder / 'prompts.json').read_text())


def template(folder, key, text=None, style='literal', locator=None, note=None):
    """PromptTemplate for prompts.json key (reviewer_system / reviewer_user / manipulations.<name>)."""
    data = load_prompts(folder)
    if text is None:
        text = data
        for part in key.split('.'):
            text = text[part]
    loc = locator or (data.get('locators') or {}).get(key.split('.')[-1]) or key
    recon = any(key.split('.')[-1] in str(r) for r in data.get('reconstructed') or [])
    return PromptTemplate(key.replace('.', '_'), text, style=style, source=Source(
        repo=str(SOURCES / folder), path='prompts.json', locator=str(loc)[:200],
        note=note or ('RECONSTRUCTED (not published verbatim)' if recon else 'verbatim, see PROVENANCE.md')))


def fill(text, **slots):
    """Literal placeholder substitution ({paper}, {title}, ...) that leaves other braces (JSON examples) alone."""
    for k, v in slots.items():
        text = text.replace('{' + k + '}', v)
    return text


_NUM = r'(\d+(?:\.\d+)?)'


def labelled_number(text, labels, lo=None, hi=None):
    """Last number following one of ``labels`` (e.g. 'Rating', 'Overall score'), allowing markdown, ':' and '/10'.
    Returns float or None; out-of-range values are None."""
    t = _robust.normalize(text)
    pat = re.compile(r'(?:' + '|'.join(labels) + r')\s*(?:\([^)]*\))?(?:\s*[|:=\-])*\s*(?:is\s*)?' + _NUM, re.I)
    found = [float(m.group(1)) for m in pat.finditer(t)]
    if not found:
        return None
    v = found[-1]
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        return None
    return v


def json_field(text, field):
    """Value of ``field`` from the last JSON object in the text that contains it, else None."""
    t = re.sub(r'```(?:json)?', '', text or '')
    for m in reversed(list(re.finditer(r'\{.*\}', t, flags=re.S))):
        try:
            obj = json.loads(m.group(0))
        except ValueError:
            continue
        if isinstance(obj, dict) and field in obj:
            return obj[field]
    m = re.search(r'"' + re.escape(field) + r'"\s*:\s*"?' + _NUM, t)
    return float(m.group(1)) if m else None


class ReviewProgram(Program):
    """Default: one review call per item (step 'review'); claim modules override ``Experiment.run`` for multi-step
    protocols (rewrite first, judge after, rebuttal turn)."""
    name = 'reviewer_manipulation'

    def __init__(self, exp, variant):
        self.exp, self.variant = exp, variant
        self.auxiliary_models = tuple(exp.auxiliary(variant))
        self.case_extra = {'min_cap_reasoning': MIN_CAP_REASONING}

    def preview(self, item):
        return self.exp.build(item, self.variant) if not self.auxiliary_models else None

    def forward(self, item, lm):
        return self.exp.run(item, lm, self.variant)


class ReviewManipulation(Experiment):
    """Base class (not registered). Subclasses set conditions, control, primary and implement build/parse."""
    maturity = 'draft'
    conditions = ('clean', 'injected')
    control = 'clean'
    primary = ('rating', 'injected', +1)     # (metric, treatment, direction: +1 = claim predicts an increase)
    metrics = ('rating',)
    n_papers = 30
    body_tokens = 4000
    params = {'max_tokens': REVIEW_CAP}

    # --- hooks -------------------------------------------------------------------------------------------------
    def auxiliary(self, variant):
        return ()

    def make_item(self, paper, condition, variant):
        """Fields needed by build(); default: truncated body plus injection placement from variant options."""
        body = truncate(paper['body'], variant.options.get('body_tokens', self.body_tokens))
        inj = variant.prompts.get(f'manipulations_{condition}')
        return {'title': paper['title'], 'text': place(body, inj.text if inj else None,
                                                        variant.options.get('placement', 'end'))}

    def build(self, item, variant):
        raise NotImplementedError

    def parse(self, text, item, variant):
        raise NotImplementedError

    def run(self, item, lm, variant):
        c = lm(self.build(item, variant), variant.params, step='review')
        return {'content': c.content, 'finish_reason': c.finish_reason}

    # --- Experiment API ----------------------------------------------------------------------------------------
    def items(self, variant, limit=None):
        n = variant.options.get('n_papers', self.n_papers)
        rows = []
        for paper in pool()[:n]:
            for cond in variant.options.get('conditions', self.conditions):
                rows.append({'id': f"{paper['base_id']}:{cond}", 'base_id': paper['base_id'], 'condition': cond,
                             'decision': paper['decision'], **self.make_item(paper, cond, variant)})
        return rows[:limit] if limit is not None else rows

    def program(self, variant):
        return ReviewProgram(self, variant)

    def record_output(self, record):
        out = super().record_output(record)
        if out is not None and 'trace' in record and 'finish_reasons' not in out:
            out = {**out, 'finish_reasons': {t.get('step'): t.get('finish_reason') for t in record['trace']}}
        return out

    def score(self, item, output, variant):
        text = output.get('content') or ''
        s = self.parse(text, item, variant)
        s['empty'] = not text.strip()
        s['truncated'] = output.get('finish_reason') in ('length', 'max_tokens')
        s['unparsed'] = all(s.get(m) is None for m in self.metrics)
        return s

    def aggregate(self, rows, variant):
        conds = variant.options.get('conditions', self.conditions)
        res = {'n_rows': len(rows), 'missing': sum(r.get('score') is None for r in rows),
               'unparsed': sum(bool((r.get('score') or {}).get('unparsed')) for r in rows),
               'truncated': sum(bool((r.get('score') or {}).get('truncated')) for r in rows), 'conditions': {},
               'paired': {}}
        by = {}
        for r in rows:
            by.setdefault(r['item']['base_id'], {})[r['item']['condition']] = r.get('score') or {}
        for c in conds:
            sc = [r['score'] for r in rows if r['item']['condition'] == c and r.get('score')]
            res['conditions'][c] = {'n': sum(r['item']['condition'] == c for r in rows), 'scored': len(sc),
                                    **{m: _mean([s.get(m) for s in sc]) for m in self.metrics}}
        for c in conds:
            if c == self.control:
                continue
            for m in self.metrics:
                pairs = [{'base_id': b, 'd': float(v[c][m]) - float(v[self.control][m])} for b, v in by.items()
                         if (v.get(c) or {}).get(m) is not None and (v.get(self.control) or {}).get(m) is not None]
                est = ci = None
                if pairs:
                    est, lo, hi = stats.cluster_bootstrap(pairs, lambda r: r['base_id'], lambda rs: mean(x['d'] for x in rs))
                    ci = [lo, hi]
                res['paired'][f'{m}:{c}-{self.control}'] = {'diff': est, 'ci95': ci, 'pairs': len(pairs)}
        m, t, d = self.primary
        h = res['paired'].get(f'{m}:{t}-{self.control}', {})
        res['shelflife'] = {'metric': f'{m} ({t} minus {self.control})', 'value': h.get('diff'), 'ci': h.get('ci95'),
                           'n': h.get('pairs'), 'direction': d, 'null_value': 0.0,
                           'headline_reliable': _robust.reliable(len(rows), res['unparsed'], res['missing'])}
        return res


def _mean(values):
    v = [float(x) for x in values if x is not None]
    return mean(v) if v else None
