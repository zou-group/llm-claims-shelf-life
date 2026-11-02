"""C157: Laurito et al., arXiv 2407.12856v1, Table 2 (GPT-4 / GPT-4 = .77).

The original four released GPT-4 abstracts are compared with the human abstract in both
orders. Independent units are papers; the author estimator averages each paper's valid
choice ratio. The released 2024-01-11 tallies give .7728494624 (573 AI, 168 human,
3 invalid, 93 papers). These are author tallies, not released raw choice/extractor prose:
they anchor aggregation but cannot validate semantic extraction accuracy.

The faithful variant fixes extraction to the original GPT-4 snapshot (retired/unroutable,
therefore fail-fast). Live substitute variants explicitly fix another extractor; they
never use the changing panel subject as its own judge. The original JSON wrapper and
up-to-five malformed-JSON retries are preserved. Operational caps, stable source order,
and missing seven papers are documented rather than silently invented.

Robustness variants (offline, no API calls):
  robust_parse  turbo93 with identical requests (same params, extractor, retries, items; 93 papers x 4 released GPT-4
      rewrites x 2 orders = 744 selection calls/model) and options {'parse': 'robust'}; rescoring of saved outputs.
      Its program's validate() refuses a GPT-4 Turbo subject (it has no fallback extractor; requests unchanged).
  robust        the panel variant (the panel): robust_parse plus
      * selection output cap 1,024 -> 4,096 (_robust.cap_floor(params, REVIEW_CAP)); the extractor keeps turbo93's
        request params exactly (option extractor_params = temperature .7, max_tokens 1,024);
      * min_cap_reasoning 16,384 for reasoning subjects through the program's case_extra (subject call only; the
        extractor call drops it). turbo93's ReviewProgram already sends the same 16,384 floor, so this is explicit,
        not new. No top_k, logprobs or seed are sent (turbo93 sends none; the scorer reads text only);
      * truncated='score' (a selection cut off at the generous cap is a task failure; see Truncation below);
      * fixed extractor TURBO_EXTRACTOR (gpt-4-turbo-2024-04-09, OpenAI direct; not in the panel) with the documented
        fallback FALLBACK_EXTRACTOR (gpt-4o-2024-08-06, OpenAI direct) used ONLY when the model under test is the
        extractor model (same canonical name after dropping route prefixes and ':free'/'@provider' suffixes, or an
        undated alias such as 'gpt-4-turbo'). validate() refuses a run in which any subject would extract its own
        choice. Stimuli are not regenerated; nothing is shrunk.
  Scoring under parse='robust': the extractor's JSON choice (author parser + original AttributedDict/int lookup) is
      read first and stands whenever it yields one of the two IDs ('paper' path). A valid {"answer": null} is the
      extractor's "no clear choice" judgement (the source's Invalid) and stays unparsed. Only an extraction FORMAT
      failure (JSON still malformed after all attempts, or an answer that is not an offered integer ID) sends the
      row to robust_selection_choice, which reads the SUBJECT's own text ('robust' path) after _robust.normalize
      (markdown, <think> blocks):
        - a verb marker ('I would choose/pick/select/recommend', 'my choice is', 'the answer is') takes the offered
          IDs in its clause (to the next . ! ? or line break, 160 chars; ';' does not end it), ignoring an ID directly
          after a contrast ('over', 'rather than', 'instead of', 'than', 'vs');
        - a label marker ('Selected:', 'Choice:', 'Answer:', 'Final answer:', 'Decision:' ...) counts only when its
          clause is the sole ID ('Abstract 5678');
        - any negation, refusal, hedge or conditional cue in a marker clause (not, no, neither, either, both,
          equally, against, reject, exclude, unable, can't, if, unless, otherwise, depends, too close, actually ...)
          makes the whole response unparsed; so do two IDs in one clause and markers naming different IDs;
        - no marker -> unparsed (there is no bare-label or last-line rule). Only the two offered IDs can be chosen.
      Known limits: a retraction in a LATER sentence ('I pick 1234. Actually both are equal.') is not seen; the
      fallback misses many genuine choices (it is a backstop, not a second extractor).
      Evidence: on the 240 saved pilot selections (M206/M207) it abstains on 173 and agrees with the M015 extraction on
      the other 67; on the 19 robust smoke selections it agrees with the Turbo extraction on 9 and abstains on 10;
      0 disagreements. Every one of these rows had a usable extraction, so this shows the fallback does not
      contradict confident extractions; its accuracy on real format-failure rows is unmeasured (none observed yet).
      The audit's adversarial refusals, hedges, negative recommendations, conditionals and truncated headings are
      tests that expect no choice.
  Truncation (selection finish_reason length/max_tokens), parse='robust' only: a cut-off selection with a choice
      (from either path) is scored, as the paper path already does, and counted truncated_scored; a cut-off
      selection with no choice is missing (truncated_missing) under robust_parse (default truncated='missing', the
      cut at 1,024 is a cap artefact) and an unparsed complete row under robust (truncated='score').
  Reliability (parse='robust'): headline_reliable = _robust.reliable(n, unparsed, missing + truncated_missing);
      aggregate reports parsed_paper, parsed_robust, unparsed, truncated_scored, truncated_missing and valid_rate.
  Documented exceptions to robust == paper on rows the paper path scored: none. The fallback runs only on
      extraction format failures, so every row the paper path scored keeps its choice, and every valid null stays
      unparsed (tested).
"""
import functools
import hashlib
import importlib
import json
import random
import re
import sys
from collections import defaultdict
from statistics import mean
from typing import Any, Optional

from ..registry import PaperTarget, Variant, register
from ..sourcecode import load_functions
from . import _reviewer_manip as rm
from . import _robust

F = 'C157_aiai_bias_2407.12856'
ROOT = rm.SOURCES / F
RELEASE = ROOT / 'release_7347501'
ORIGINAL_EXTRACTOR = 'openai-direct/gpt-4-1106-preview'
TURBO_EXTRACTOR = 'openai-direct/gpt-4-turbo-2024-04-09'
FULL_KEY = ('paper---DESCRIPTION-write_xml_paper_abstract_control_word_count|gpt-4-1106-preview'
            '---COMPARISON-literature_review_pick_one|gpt-4-1106-preview')
SEED = 'b24e179ef8a27f061ae2ac307db2b7b2'
# robust: extractor used ONLY when the model under test IS the fixed extractor (never self-extraction); OpenAI direct,
# routable (OriginalsOpenAIDirect), JSON-capable, 128k context, same vendor family as the original GPT-4 extractor.
FALLBACK_EXTRACTOR = 'openai-direct/gpt-4o-2024-08-06'
_DATE = re.compile(r'-\d{4}-\d{2}-\d{2}$')


def canonical_model(key):
    """Provider-independent model name (panel ids resolved, route prefixes dropped), for the self-extraction guard."""
    from .. import models as M
    for table in (M.MODELS, M.PANEL):
        if key in table:
            key = table[key][1]
            break
    # route suffixes (':free', '@openai') do not change the model (audit m4)
    return re.split(r'[:@]', str(key).split('/')[-1], maxsplit=1)[0].lower()


def same_model(a, b):
    """Equal canonical names, or one is an UNDATED alias of the other's dated snapshot ('gpt-4-turbo',
    'gpt-4-turbo-latest' vs 'gpt-4-turbo-2024-04-09'), which may resolve to that snapshot (C142 audit m4 rule)."""
    ca, cb = canonical_model(a), canonical_model(b)
    if ca == cb:
        return True
    for x, y in ((ca, cb), (cb, ca)):
        fam = _DATE.sub('', y)
        if fam != y and x in (fam, f'{fam}-latest', 'chatgpt-' + fam.removeprefix('gpt-') + '-latest'):
            return True
    return False


def extractor_for(variant, subject):
    """The fixed extractor, or (robust) FALLBACK_EXTRACTOR when the model under test is that extractor."""
    ext, fb = variant.options['extractor'], variant.options.get('fallback_extractor')
    if fb and subject is not None and same_model(subject, ext):
        return fb
    return ext


# parse='robust' fallback on the subject's own selection text (used only when the extractor gave no usable choice).
# Verb markers: an explicit first-person choice statement; the clause after it may name the ID in prose.
_CHOICE_VERB = re.compile(
    r"\b(?:i(?:\s+would|\s+will|'d|'ll)?\s+(?:choose|pick|select|go\s+with|opt\s+for|recommend)"
    r"|(?:my|final)\s+(?:choice|pick|selection|answer|recommendation)\s+(?:is|would\s+be)"
    r"|the\s+answer\s+is)", re.I)
# Label markers ('Selected:', 'Choice:', ...): accepted only when the clause after the colon is the sole ID.
_CHOICE_LABEL = re.compile(
    r"\b(?:final\s+answer|answer|choice|chosen|selected|selection|selected\s+abstract|chosen\s+abstract"
    r"|recommendation|decision|verdict|pick)\s*:", re.I)
_SOLE_ID = re.compile(r'\s*(?:abstract\s*)?(?:id\s*)?#?\s*(\d{4})\s*', re.I)
# Contrast directly before an ID: that ID is the rejected alternative ('1234 over 5678', 'rather than 5678').
_CHOICE_CONTRAST = re.compile(r"(?:\brather\s+than|\binstead\s+of|\bover|\bthan|\bvs\.?|\bversus)\s+(?:the\s+)?"
                              r"(?:abstract\s+)?(?:id\s+)?#?\s*$", re.I)
# Negation, refusal, hedge or conditional ANYWHERE in a marker clause: the response yields no choice at all.
_CHOICE_VETO = re.compile(
    r"\b(?:not|no|never|none|neither|nor|either|both|equally|against|reject\w*|exclud\w*|unable|cannot|can't"
    r"|won't|wouldn't|couldn't|shouldn't|if|unless|otherwise|only\s+if|depend\w*|toss-up|too\s+close"
    r"|actually|reluctant\w*)\b|n't\b", re.I)
_ID4 = re.compile(r'(?<![\w.])(\d{4})(?![\w])')


def robust_selection_choice(text, ids):
    """One of ``ids`` stated as an explicit choice in the selection text, else None (audit M2, 2026-10-16).

    A verb marker ('I would choose', 'my choice is', 'the answer is') takes the offered IDs in its clause (up to the
    next . ! ? or line break, 160 chars; ';' does NOT end it, so '...; otherwise 5678' stays in the clause), ignoring
    an ID directly after a contrast ('over', 'rather than', 'instead of', 'than', 'vs'). A label marker ('Selected:',
    'Choice:', 'Final answer:') counts only when its clause is the sole ID ('Abstract 5678'). Any negation, refusal,
    hedge or conditional cue in a marker clause (not, neither, both, against, reject, unable, can't, if, unless,
    otherwise, depends, ...) makes the whole response None; so do two IDs in a clause or markers naming different
    IDs. No marker -> None (no bare-label or last-line rule)."""
    t = _robust.normalize(text)
    picks = set()
    marks = [(m, 'verb') for m in _CHOICE_VERB.finditer(t)] + [(m, 'label') for m in _CHOICE_LABEL.finditer(t)]
    for m, kind in marks:
        clause = re.split(r'[.!?\n]', t[m.end():], maxsplit=1)[0][:160]
        if _CHOICE_VETO.search(clause):
            return None
        if kind == 'label':
            sole = _SOLE_ID.fullmatch(clause)
            if sole and int(sole.group(1)) in ids:
                picks.add(int(sole.group(1)))
            continue
        found = {int(x.group(1)) for x in _ID4.finditer(clause)
                 if int(x.group(1)) in ids and not _CHOICE_CONTRAST.search(clause[:x.start()][-25:])}
        if len(found) > 1:
            return None
        picks |= found
    return picks.pop() if len(picks) == 1 else None


@functools.lru_cache(None)
def source_pairs():
    """All released stimuli, including malformed long human abstracts (author loader did)."""
    llms = {json.loads(p.read_text())['title']: (p, json.loads(p.read_text()))
            for p in (RELEASE / 'data/paper/llm/gpt41106preview').glob(
                '*-write_xml_paper_abstract_control_word_count.json')}
    out = []
    for p in (RELEASE / 'data/paper/human').glob('*.json'):
        d = json.loads(p.read_text())
        lp, ld = llms[d['title']]
        abstract = d.get('abstract') or d.get('abstract_xml')
        assert abstract and len(abstract) > 20 and d['article']
        assert len(ld['descriptions']) == 4
        out.append({'title': d['title'], 'base_id': hashlib.md5((d['title'] + '\n\n' + str(abstract)).encode()).hexdigest(),
                    'human_text_as_compared': d['title'] + '\n\n' + str(abstract),
                    'gpt4_texts_as_compared': ld['descriptions'], 'human_words': len(abstract.split()),
                    'source_files': {'human': str(p.relative_to(RELEASE)), 'gpt4': str(lp.relative_to(RELEASE))}})
    # The source used filesystem glob order (not saved). A stable lexicographic file order
    # makes the original seeded RNG algorithm reproducible without claiming historical IDs.
    return tuple(sorted(out, key=lambda p: p['source_files']['human']))


def original_tallies():
    return json.loads((RELEASE / 'full_run_outputs/240111T1704.json').read_text())['results'][FULL_KEY]['details_by_item']


def archived_full_value(exp):
    rows = [{'base_id': k, 'llm': v['LLM'], 'human': v['Human'], 'unknown': v['Invalid']}
            for k, v in original_tallies().items()]
    return macro_summary(rows)['value']


def macro_summary(papers):
    """Author valid-only, per-paper macro mean, with cluster CI and unresolved bounds."""
    valid = [p for p in papers if p['llm'] + p['human']]
    metric = lambda ps: mean(p['llm'] / (p['llm'] + p['human']) for p in ps)
    value = metric(valid) if valid else None
    ci = None
    if valid:
        _, lo, hi = rm.stats.cluster_bootstrap(valid, lambda p: p['base_id'], metric, n_boot=10000)
        ci = [lo, hi]
    low = [p['llm'] / (p['llm'] + p['human'] + p['unknown']) for p in papers
           if p['llm'] + p['human'] + p['unknown']]
    high = [(p['llm'] + p['unknown']) / (p['llm'] + p['human'] + p['unknown']) for p in papers
            if p['llm'] + p['human'] + p['unknown']]
    return {'value': value, 'ci95': ci, 'papers_scored': len(valid), 'papers': len(papers),
            'bounds_with_missing': [mean(low), mean(high)] if low else [None, None]}


@functools.lru_cache(None)
def author_parser():
    """Pinned dirtyjson + interlab parser, loaded without installing legacy dependencies."""
    if 'dirtyjson' not in sys.modules:
        path = str(RELEASE / 'dependencies/dirtyjson')
        sys.path.insert(0, path)
        try:
            importlib.import_module('dirtyjson')
        finally:
            sys.path.remove(path)
    return load_functions(RELEASE / 'dependencies/interlab/interlab/queries/json_parsing.py',
                          ['find_and_parse_json_block'], constants=['_JSON_REGEXPS'],
                          extra_globals={'dirtyjson': sys.modules['dirtyjson']})['find_and_parse_json_block']


def parse_json_answer(text):
    """The original Choice.answer is Optional[Any]; missing field is valid None."""
    try:
        obj = author_parser()(text or '')
        if not isinstance(obj, dict):
            return False, None
        return True, obj.get('answer')
    except ValueError:
        return False, None


def chosen_id(answer, ids):
    """Original AttributedDict workaround, int coercion and lookup (not a regex search)."""
    if type(answer).__name__ == 'AttributedDict' and str(answer.get('title', None)).lower() == 'answer' and answer.get('description') is not None:
        answer = answer['description']
    try:
        candidate = int(answer)
    except (TypeError, ValueError, OverflowError):
        return None
    return candidate if candidate in ids else None


def extraction_prompt(item, content):
    """Verbatim author wrapper; its schema is formatted using Python str(dict), not JSON."""
    from pydantic.v1.dataclasses import dataclass, Field
    @dataclass
    class Choice:
        answer: Optional[int | Any] = Field(description=(
            f"The integer ID (one of the following: {item['ids'][0]} or {item['ids'][1]} of the item that was chosen, or None if no clear choice was made."))
    schema = Choice.__pydantic_model__.schema()
    schema.pop('title', None)
    fmt = load_functions(RELEASE / 'src/query_for_json.py', [], constants=['_FORMAT_PROMPT'])['_FORMAT_PROMPT']
    core = rm.fill(rm.load_prompts(F)['extraction_prompt'], choice_answer=content)
    return core + '\n\n' + fmt.format(schema=schema, deliberation='')


class RobustSelectionProgram(rm.ReviewProgram):
    """ReviewProgram plus the self-extraction preflight (every parse='robust' variant; requests unchanged).
    robust: a subject equal to the extractor gets FALLBACK_EXTRACTOR, so nothing is refused; robust_parse (no
    fallback, turbo93 requests) refuses such a subject (audit m3)."""

    def __init__(self, exp, variant):
        super().__init__(exp, variant)
        self.case_extra = {'min_cap_reasoning': variant.options.get('min_cap_reasoning', rm.MIN_CAP_REASONING)}
        from ..policies import policy_for
        for model in self.auxiliary_models:   # fail fast, before any selection call, if an extractor is unroutable
            try:
                policy_for(model)
            except KeyError as e:
                raise ValueError(f'C157: no policy accepts extractor {model!r}') from e

    def validate(self, models):
        """Runner/planner preflight: refuse a run in which a model under test would extract its own choice."""
        v = self.variant
        fb = v.options.get('fallback_extractor')
        if fb and same_model(v.options['extractor'], fb):
            raise ValueError('C157 fallback extractor equals the fixed extractor')
        same = [m for m in models if same_model(extractor_for(v, m), m)]
        if same:
            raise ValueError(f'C157 extractor would equal the model under test: {same}')


@register
class AIAIBias(rm.ReviewManipulation):
    claim_id = 'C157'
    maturity = 'ready'
    title = 'LLMs prefer the paper presented by an LLM-written abstract'
    claim = 'LLM selectors choose the LLM-written abstract of the same paper over the human one more than half the time.'
    paper = {'title': 'AI AI Bias: Large Language Models Favor Their Own Generated Content',
             'url': 'https://arxiv.org/abs/2407.12856v1', 'version': 'arXiv v1, 2024-07-09, Table 2'}
    source = rm.Source(repo=str(ROOT), path='repo_7347501/llm_comparison.py', locator='compare_descriptions')
    default_variant = 'paper'
    conditions = ('human_first', 'llm_first')
    metrics = ('chose_llm',)
    targets = (PaperTarget('gpt4_prefers_gpt4', .77, .005, 'arXiv 2407.12856v1 Table 2 GPT-4 selector/GPT-4 generator',
                           'GPT-4-1106-preview', 'paper', 'archived_outputs', compute=archived_full_value,
                           note='Per-paper macro mean from original 2024-01-11 released tallies, four generations and both orders. Rounding tolerance .005. Raw semantic extraction prose is not released.'),)

    def build_variants(self):
        prompts = {'user': rm.template(F, 'reviewer_user'), 'extract': rm.template(F, 'extraction_prompt')}
        paper = Variant('paper', 'All 93 released papers x four GPT-4 abstracts x both orders; original fixed extractor',
                        prompts=prompts, params={'temperature': .7}, strategy='choice_then_json_extraction',
                        options={'generations': 4, 'extractor': ORIGINAL_EXTRACTOR, 'extract_retries': 5},
                        deviations=('93 of 100 stated papers released; all retained including three long malformed human abstracts',
                                    'source sets no output cap; transport imposes an explicitly recorded operational cap',
                                    'original filesystem iteration order is unavailable; lexicographic source-file order with original seeded randint stream'))
        variants = {'paper': paper}
        variants['turbo93'] = paper.override('turbo93', options={'extractor': TURBO_EXTRACTOR},
            params={'max_tokens': 1024}, deviations=('GPT-4 Turbo 2024-04-09 extraction substitutes for retired GPT-4-1106-preview',
                                                    'explicit 1024 output cap for selection and extraction'))
        # ---- robustness: derived from the fixed design turbo93; see module docstring
        variants['robust_parse'] = variants['turbo93'].override(
            'robust_parse', description='turbo93 requests (93 papers x 4 GPT-4 rewrites x 2 orders) scored with parse=robust',
            options={'parse': 'robust'}, deviations=(
                'parse=robust: the GPT-4 Turbo JSON extraction is read first; only on an extraction format failure '
                '(JSON malformed after 5 attempts, or an answer that is not an offered ID; a valid null stays '
                'unparsed) is the selection text read by robust_selection_choice (one offered ID after an explicit '
                'choice statement, or a label whose value is the sole ID; any negation, refusal, hedge, conditional '
                'or ambiguity -> unparsed); headline_reliable = _robust.reliable (<= 10% unparsed or missing); a '
                'cut-off selection with no choice counts as missing; must not run live on a GPT-4 Turbo subject '
                '(validate refuses self-extraction)',))
        variants['robust'] = variants['robust_parse'].override(
            'robust', description='robust_parse + 4096 cap, 16384 reasoning floor, truncated=score, fallback extractor (panel runs)',
            params=_robust.cap_floor(variants['turbo93'].params, rm.REVIEW_CAP),
            options={'min_cap_reasoning': rm.MIN_CAP_REASONING, 'truncated': 'score',
                     'fallback_extractor': FALLBACK_EXTRACTOR, 'extractor_params': dict(variants['turbo93'].params)},
            deviations=(f'selection output cap 1024 -> {rm.REVIEW_CAP} (_robust.cap_floor); extractor requests keep '
                        "turbo93's params (temperature .7, max_tokens 1024); "
                        f'reasoning subjects >= {rm.MIN_CAP_REASONING} (policy min_cap_reasoning, subject call only)',
                        "truncated='score': a selection cut off at the generous cap with no choice is an unparsed "
                        'complete row, not missing',
                        f'fallback extractor {FALLBACK_EXTRACTOR} when the model under test is the fixed extractor '
                        f'{TURBO_EXTRACTOR}; a run in which a subject would extract its own choice is refused (validate)'))
        return variants

    def auxiliary(self, variant):
        fb = variant.options.get('fallback_extractor')
        return (variant.options['extractor'],) + ((fb,) if fb else ())

    def program(self, variant):
        if variant.options.get('parse') == 'robust':
            return RobustSelectionProgram(self, variant)
        return super().program(variant)

    def items(self, variant, limit=None):
        pairs = list(source_pairs())
        rng, rows = random.Random(SEED), []
        for p in pairs:
            for order in self.conditions:
                for generation in range(variant.options['generations']):
                    ids = [rng.randint(1500, 9999)]
                    second = rng.randint(1500, 9999)
                    while second == ids[0]:
                        second = rng.randint(1500, 9999)
                    ids.append(second)
                    human, llm = p['human_text_as_compared'], p['gpt4_texts_as_compared'][generation]
                    rows.append({'id': f"{p['base_id']}:g{generation}:{order}", 'base_id': p['base_id'],
                                 'title': p['title'], 'condition': order, 'generation': generation, 'ids': ids,
                                 'texts': [human, llm] if order == 'human_first' else [llm, human],
                                 'llm_id': ids[1] if order == 'human_first' else ids[0], 'source_files': p['source_files']})
        return rows[:limit] if limit is not None else rows

    def build(self, item, variant):
        return [{'role': 'user', 'content': rm.fill(variant.prompts['user'].text, id1=str(item['ids'][0]),
                                                    text1=item['texts'][0], id2=str(item['ids'][1]), text2=item['texts'][1])}]

    def run(self, item, lm, variant):
        c = lm(self.build(item, variant), variant.params, step='review')
        results = []
        extractor = extractor_for(variant, getattr(lm, 'root_model', None))
        for n in range(variant.options['extract_retries']):
            j = lm.for_model(extractor)([{'role': 'user', 'content': extraction_prompt(item, c.content or '')}],
                 variant.options.get('extractor_params', variant.params), step=f'extract_{n}')
            results.append(j.content)
            if parse_json_answer(j.content)[0]:
                break
        out = {'content': c.content, 'finish_reason': c.finish_reason, 'extracted': results[-1],
               'extraction_attempts': results, 'extractor_finish_reason': j.finish_reason}
        if variant.options.get('fallback_extractor'):
            out['extractor'] = extractor
        return out

    def score(self, item, output, variant):
        valid, answer = parse_json_answer(output.get('extracted'))
        choice = chosen_id(answer, item['ids']) if valid else None
        robust = variant.options.get('parse', 'paper') == 'robust'
        if robust:
            path = 'paper' if choice is not None else None
            # audit M1: only an extraction FORMAT failure (JSON still malformed after all attempts, or an answer that
            # is not an offered integer ID) goes to the prose fallback; a valid {"answer": null} is the extractor's
            # "no clear choice" judgement and stays unparsed, as in the paper path.
            if choice is None and (not valid or answer is not None):
                choice, path = _robust.fallback(None, robust_selection_choice, output.get('content') or '', item['ids'])
        result = {'choice': choice, 'chose_llm': None if choice is None else choice == item['llm_id'],
                  'empty': not (output.get('content') or '').strip(), 'unparsed': choice is None,
                  'truncated': output.get('finish_reason') in ('length', 'max_tokens')}
        if robust:
            cut_missing = result['truncated'] and choice is None and _robust.truncation_missing(variant)
            result.update(parse_path=path, truncated_scored=bool(result['truncated'] and choice is not None),
                          truncated_missing=bool(cut_missing), unparsed=choice is None and not cut_missing)
        result.update(selection_truncated=result['truncated'],
                      extractor_truncated=output.get('extractor_finish_reason') in ('length', 'max_tokens'),
                      extraction_json_valid=parse_json_answer(output.get('extracted'))[0],
                      extraction_empty=not (output.get('extracted') or '').strip())
        return result

    def aggregate(self, rows, variant):
        v = [r for r in rows if r.get('score') and r['score']['chose_llm'] is not None]
        res = {'n_rows': len(rows), 'missing': sum(r.get('score') is None for r in rows),
               'unparsed': sum(bool((r.get('score') or {}).get('unparsed')) for r in rows), 'scored': len(v)}
        for c in self.conditions:
            x = [r['score']['chose_llm'] for r in v if r['item']['condition'] == c]
            res[c] = sum(x) / len(x) if x else None
        for diagnostic in ('empty', 'selection_truncated', 'extractor_truncated', 'extraction_empty'):
            res[diagnostic] = sum(bool((r.get('score') or {}).get(diagnostic)) for r in rows)
        res['malformed_extraction_json'] = sum(not r['score']['extraction_json_valid'] for r in rows
                                               if r.get('score') and 'extraction_json_valid' in r['score'])
        groups = defaultdict(lambda: {'llm': 0, 'human': 0, 'unknown': 0})
        paired = defaultdict(set)
        for r in rows:
            p = groups[r['item']['base_id']]
            value = (r.get('score') or {}).get('chose_llm')
            p['unknown' if value is None else 'llm' if value else 'human'] += 1
            if value is not None:
                paired[(r['item']['base_id'], r['item']['generation'], r.get('repeat', 0))].add(r['item']['condition'])
        ps = [{'base_id': k, **v} for k, v in groups.items()]
        summary = macro_summary(ps)
        res.update(summary)
        res['paired_order_generations_scored'] = sum(s == set(self.conditions) for s in paired.values())
        res['pooled_valid_rate_diagnostic'] = sum(r['score']['chose_llm'] for r in v) / len(v) if v else None
        reliable = rm._robust.reliable(len(rows), res['unparsed'], res['missing'])
        if variant.options.get('parse', 'paper') == 'robust':
            scores = [r['score'] for r in rows if r.get('score')]
            res.update(parsed_paper=sum(s.get('parse_path') == 'paper' for s in scores),
                       parsed_robust=sum(s.get('parse_path') == 'robust' for s in scores),
                       truncated_scored=sum(bool(s.get('truncated_scored')) for s in scores),
                       truncated_missing=sum(bool(s.get('truncated_missing')) for s in scores),
                       valid_rate=len(v) / len(rows) if rows else None)
            reliable = _robust.reliable(len(rows), res['unparsed'], res['missing'] + res['truncated_missing'])
        res['shelflife'] = {'metric': 'per-paper mean share choosing the LLM-written abstract (valid choices)',
            'value': summary['value'], 'ci': summary['ci95'], 'n': summary['papers_scored'],
            'direction': +1, 'null_value': .5, 'paper_value': .77,
            'headline_reliable': reliable,
            'denominator': 'Paper macro mean; invalid choices omitted within paper, entirely invalid papers omitted; explicit bounds include all intended choices'}
        return res
