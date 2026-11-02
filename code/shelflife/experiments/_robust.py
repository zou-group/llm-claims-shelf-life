"""Shared robustness helpers for modern-model outputs (added 2026-10-10 for panel runs).

Modern chat and reasoning models answer in formats the papers' parsers never saw: markdown emphasis (**B**), LaTeX
(\\boxed{12}, \\(x\\)), a reasoning preamble before the answer, <think> blocks, "Final answer:" trailers, unicode minus
signs, thousands separators. Claim modules use these helpers ONLY as a fallback: the paper's parser runs first and its
result stands whenever it succeeds; the robust extraction is tried only when the paper parser returns nothing. That
keeps every faithful variant's scores (and prior-run parity) unchanged and makes the fallback auditable: modules
report how many answers each path produced.

Conventions for claim modules :
  * scoring option ``parse``: 'paper' (default, the released scorer) or 'robust' (paper first, then these helpers);
  * variant ``robust_parse``: the panel minimal variant with options {'parse': 'robust'} and identical requests,
    used to rescore saved outputs offline;
  * variant ``robust``: ``robust_parse`` plus an output cap large enough that the answer is not cut off
    (``max_tokens`` floor), for new runs on large panels;
  * reliability: ``reliable(n, unparsed, missing)`` tolerates a small share of failures (default 10%) instead of
    flagging a whole model for one unparsed answer.
"""
import re

ROBUST_TOLERANCE = 0.10      # max share of unparsed + missing rows for a reliable headline under parse='robust'
MIN_ROBUST_TOKENS = 1024     # default output-cap floor for 'robust' variants on non-reasoning models

_THINK = re.compile(r'<think>.*?</think>|<thinking>.*?</thinking>|<reasoning>.*?</reasoning>', re.S | re.I)
_OPEN_THINK = re.compile(r'<think>.*\Z|<thinking>.*\Z', re.S | re.I)   # unterminated (truncated) think block
_BOXED = re.compile(r'\\boxed\s*\{((?:[^{}]|\{[^{}]*\})*)\}')
_FINAL = re.compile(r'(?:final\s+answer|the\s+answer\s+is|answer\s*:|answer\s+is|(?:correct|best|right)\s+(?:answer|option|choice)\s+is|my\s+(?:answer|choice)\s+is|therefore[,:]?|thus[,:]?|so[,:]?)'
                    r'\s*(?:is\s*)?:?\s*', re.I)
_NUM = re.compile(r'(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w])')
_FRAC = re.compile(r'(?<![\w.])[-+]?\d+/\d+(?![\w/])')
_CONNECTIVES = {'or', 'and', 'is', 'because', 'since', 'as', 'seems', 'would', 'fits', 'matches', 'nor'}
_NEG = re.compile(r"(?:\bnot|n't|\bno|\brather\s+than|\binstead\s+of|\bexcept)\s+(?:option\s+|choice\s+)?\(?$", re.I)


def normalize(text):
    """Strip reasoning blocks, markdown emphasis/code, LaTeX wrappers; unify minus signs and whitespace."""
    if text is None:
        return ''
    t = _THINK.sub(' ', str(text))
    t = _OPEN_THINK.sub(' ', t)
    t = _BOXED.sub(lambda m: f' {m.group(1)} ', t)
    t = re.sub(r'\\text\{([^{}]*)\}', r'\1', t)
    t = re.sub(r'\\\(|\\\)|\\\[|\\\]|\$\$?', ' ', t)
    t = t.replace('\\times', '×').replace('\\cdot', '·').replace('\\,', '').replace('\\!', '')
    t = re.sub(r'(\*\*|__|\*|`+)', '', t)            # markdown emphasis and code ticks
    t = re.sub(r'^\s{0,3}#{1,6}\s*', '', t, flags=re.M)  # headings
    t = t.replace('\u2212', '-').replace('\u2013', '-').replace('\u2014', '-').replace('\u00a0', ' ')
    return re.sub(r'[ \t]+', ' ', t).strip()


def answer_segment(text):
    """The part of the response after the LAST explicit answer marker ('final answer', 'the answer is', 'answer:'),
    or the whole normalized text if there is none."""
    t = normalize(text)
    last = None
    for m in _FINAL.finditer(t):
        if re.match(r'(?:final\s+answer|the\s+answer\s+is|answer\s*:|answer\s+is|(?:correct|best|right)\s+(?:answer|option|choice)\s+is|my\s+(?:answer|choice)\s+is)', m.group(0), re.I):
            last = m
    return t[last.end():].strip() if last else t


def _to_number(s):
    s = s.replace(',', '')
    if '/' in s:
        a, b = s.split('/', 1)
        return float(a) / float(b) if float(b) else None
    return float(s)


def _numbers(t, fractions):
    if fractions:
        spans = [(m.start(), m.group(0)) for m in _FRAC.finditer(t)]
        covered = {i for m in _FRAC.finditer(t) for i in range(m.start(), m.end())}
        spans += [(m.start(), m.group(0)) for m in _NUM.finditer(t) if m.start() not in covered]
        return [x for _, x in sorted(spans)]
    return _NUM.findall(_FRAC.sub(lambda m: m.group(0).split('/')[0], t))


def extract_number(text, where='marker_or_sole', fractions=False):
    """A numeric answer, or None.

    where='marker_or_sole' (default, conservative): the first number after the LAST explicit answer marker
    ('final answer', 'the answer is', 'answer:'); else the number if the whole response contains exactly one.
    where='marker_then_last': as above, else the last number anywhere (opt-in: only for tasks whose answer is the
    final number of a worked solution). where='last' / 'first': last / first number anywhere.
    fractions=False reads '6/10' as 6 (a rating); fractions=True reads it as 0.6."""
    t = normalize(text)
    if where in ('marker_or_sole', 'marker_then_last'):
        seg = answer_segment(text)
        if seg != t:
            nums = _numbers(seg, fractions)
            if nums:
                return _to_number(nums[0])
        if where == 'marker_or_sole':
            nums = _numbers(t, fractions)
            return _to_number(nums[0]) if len(nums) == 1 else None
    nums = _numbers(t, fractions)
    if not nums:
        return None
    return _to_number(nums[0] if where == 'first' else nums[-1])


def extract_choice(text, options, where='marker_then_unique'):
    """One option label from ``options`` (e.g. 'ABCD', ['1', '2'], ['Yes', 'No']), case-insensitive for words.

    Order: (1) after the LAST answer marker, the first clause must name exactly one option (non-negated);
    (2) a response that opens with a single option ('B', '(B) …', 'B. …', 'Yes, …') and names no other option in its
    first sentence; (3) the unique non-negated option mentioned in the whole response as a standalone label
    ('(B)', 'B)', 'B.', 'B:' or a line on its own for single letters; any whole-word mention for word options).
    Mentions after a negation ('not A', 'rather than B') are ignored. A single capital letter used as a word
    ('A refusal is …', 'I think …') is not a label. Returns None if ambiguous or absent."""
    opts = [str(o) for o in options]
    words = any(len(o) > 1 for o in opts)
    canon = {o.lower(): o for o in opts}
    alt = '|'.join(re.escape(o) for o in sorted(opts, key=len, reverse=True))
    if words:
        rx = re.compile(r'\b(' + alt + r')\b', re.I)
        strict = rx
    else:
        # bare token: not glued to letters/digits; strict label: wrapped or followed by ) . : , or end of line
        rx = re.compile(r'(?<![A-Za-z0-9])\(?(' + alt + r')\)?(?![A-Za-z0-9])')
        strict = re.compile(r'(?<![A-Za-z0-9])(?:\((' + alt + r')\)|(' + alt + r')(?=\s*(?:[).:,;\]]|$)))', re.M)

    def mentions(s, pattern):
        out = []
        for m in pattern.finditer(s):
            lab = next(g for g in m.groups() if g)
            if _NEG.search(s[:m.start()][-20:]):
                continue
            nxt = re.match(r'\s+([a-z]+)', s[m.end():])
            if not words and lab in ('A', 'I') and nxt and nxt.group(1) not in _CONNECTIVES:
                continue  # article / pronoun, not an option label
            out.append(canon.get(lab.lower(), lab))
        return out

    t = normalize(text)
    seg = answer_segment(text)
    if seg != t:
        clause = re.split(r'[.;\n]|\bor\b|\band\b|,\s*(?:but|not)\b', seg, maxsplit=1)[0]
        whole = re.split(r'[.;\n]', seg, maxsplit=1)[0]
        found = mentions(clause[:80], rx)
        if found and len(set(mentions(whole[:120], rx))) == 1:
            return found[0]
        if found:
            return None  # 'Answer: A or B' and similar: ambiguous
    opening = re.split(r'(?<=[.!?])\s|\n', t, maxsplit=1)[0]
    head = mentions(t[:12], rx)
    if head and t.lstrip('( ').lower().startswith(head[0].lower()) and set(mentions(opening, rx)) == {head[0]}:
        return head[0]
    found = set(mentions(t, strict))
    return found.pop() if len(found) == 1 else None


def extract_yes_no(text):
    return extract_choice(text, ['Yes', 'No'])


def fallback(paper_value, robust_fn, *args, **kw):
    """Paper parser result if it succeeded (not None), else the robust extraction. Returns (value, path) with path
    in {'paper', 'robust', None} so modules can report how many answers each path produced."""
    if paper_value is not None:
        return paper_value, 'paper'
    v = robust_fn(*args, **kw)
    return v, ('robust' if v is not None else None)


def reliable(n, unparsed=0, missing=0, tol=ROBUST_TOLERANCE):
    """Headline reliability under parse='robust': enough rows and at most ``tol`` of them unparsed or missing."""
    return bool(n) and (unparsed + missing) / n <= tol


def truncation_missing(variant):
    """How a cut-off response (finish_reason 'length') is treated under parse='robust', from the variant option
    ``truncated``: 'missing' (default) = unparsed and excluded, right for ``robust_parse`` rescoring outputs produced
    at the paper's small cap, where the cut is an artefact of the cap; 'score' = scored by the paper rule on whatever
    text exists, right for ``robust`` runs whose cap is generous: a model that spends the whole budget without
    answering has failed the task (C057 the panel: reasoning models looping for 16k tokens), and excluding those rows
    would flag the model unreliable and drop exactly its hardest items. Returns True when cut-off rows are missing."""
    opts = getattr(variant, 'options', variant) or {}
    return opts.get('truncated', 'missing') != 'score'


def cap_floor(params, floor=MIN_ROBUST_TOKENS):
    """Params with max_tokens raised to at least ``floor`` (the policy raises it further for reasoning models)."""
    p = dict(params)
    if p.get('max_tokens') is not None:
        p['max_tokens'] = max(int(p['max_tokens']), floor)
    return p
