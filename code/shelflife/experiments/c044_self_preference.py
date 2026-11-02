"""C044 — self preference in XSUM pairwise judges.

Faithful variants replay frozen author summaries and the authors' two-order
probability-weighted estimator (top_logprobs=2, first-token choice). They are
meaningful as self preference only when evaluated by the named original author.
Released binary-choice excess over the opposing GPT judge is anchored offline.

The fresh generation variant uses a fixed neutral the panel judge (M015). This is
explicitly NOT a staged round-robin panel neutral design.
The original 1106 snapshots have no direct route in the current package; frozen
outputs remain usable. No retired-snapshot replacement is labeled faithful.

Robustness: parse='robust' (robust_choice, a guarded _robust.extract_choice, only when the
exact '1'/'2' rule fails), variants robust_parse (requests of generated_fixed_neutral_binary) and robust (judge cap
floor 1024; neutral_fallback M025 when the subject is the neutral M015, alias-aware via same_model). Where robust
differs from paper on rows the paper parser DID score: none.
"""
import functools
import json
import math
import re
from collections import defaultdict

from .. import stats
from . import _robust
from ..paths import PROJECT
from ..program import Program
from ..prompts import PromptTemplate, Source
from ..registry import Experiment, PaperTarget, Variant, register

REPO = PROJECT / 'code/shelflife/data/sources/C044_self_recognition'
COMMIT = '5b1cb7321eef8a1564c8c782917e3706aafcc589'
SRC = Source(repo=str(REPO), path='experiments.py', locator='generate_gpt_logprob_results',
             url='https://github.com/ArjunPanickssery/self_recognition',
             note=f'Selected-file snapshot of upstream {COMMIT}; see PROVENANCE.md')


def template(name, var, dictionary=False):
    return PromptTemplate.from_source(name, Source(repo=str(REPO), path='prompts.py', locator=var,
        note=f'Upstream {COMMIT}'), kind='py_dict' if dictionary else 'py_var', var=var,
        style='literal' if name.endswith('system') else 'format')


COMPARE = template('comparison', 'COMPARISON_PROMPT_TEMPLATE')
SYSTEM = template('comparison_system', 'COMPARISON_SYSTEM_PROMPT')
SUMMARY_SYSTEM = template('summary_system', "DATASET_SYSTEM_PROMPTS['xsum']", True)
SUMMARY_USER = PromptTemplate('summary_user', 'Article:\n{article}\n\nProvide only the summary with no other text.',
    source=Source(repo=str(REPO), path='models.py', locator='get_gpt_summary/history[1] f-string'))


@functools.lru_cache(maxsize=1)
def release():
    articles = json.loads((REPO / 'articles/xsum_train_articles.json').read_text())
    summaries = {name: json.loads((REPO / f'summaries/xsum/xsum_train_{name}_responses.json').read_text())
                 for name in ('human', 'claude', 'gpt35', 'gpt4', 'llama')}
    return articles, summaries


def archive(author):
    return json.loads((REPO / f'results/xsum/{author}_results.json').read_text())


def archived_binary(author, rival):
    rows = [r for r in archive(author) if r['model'] == rival]
    if not rows:
        raise ValueError(f'No released comparisons for {author}/{rival}')
    return sum((int(r['forward_comparison'] == '1') + int(r['backward_comparison'] == '2')) / 2
               for r in rows) / len(rows)


def archived_excess():
    return archived_binary('gpt4', 'gpt35') - (1 - archived_binary('gpt35', 'gpt4'))


def logprobs(completion):
    try:
        return completion.raw_response['choices'][0]['logprobs']['content'][0]['top_logprobs']
    except (KeyError, IndexError, TypeError):
        return []


def own_probability(top, desired):
    """Author branch rule: use top[0] for chosen own label, else top[1].

    Do not renormalize over the two labels, or silently replace the second
    alternative when it is another token; retain the author's estimator.
    """
    if len(top) < 2 or top[0].get('token') not in ('1', '2'):
        return None
    selected = top[0] if top[0]['token'] == desired else top[1]
    value = selected.get('logprob')
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value > 0:
        return None
    return math.exp(value)


def robust_choice(content):
    """Fallback label: _robust.extract_choice(content, '12'), kept only if it came from an answer marker, or the
    response opens with that label ('Summary 2 ...'), or it is the only '1'/'2' token in the (think-stripped) response.
    'Summary 1 is better than Summary 2' -> None (two labels, no marker; the shared helper's strict-label stage
    would pick the line-final '2')."""
    v = _robust.extract_choice(content or '', ['1', '2'])
    if v is None:
        return None
    t = _robust.normalize(content)
    if _robust.answer_segment(content) != t:
        return v
    opening = re.match(r'^\W*(?:summary\s*)?([12])(?![\d.])', t, re.I)
    first_sentence = re.split(r'(?<!\d)[.!?\n]', t, 1)[0]
    if opening and opening.group(1) == v and set(re.findall(r'(?<![\w.])([12])(?!\w|\.\d)', first_sentence)) == {v}:
        return v
    return v if set(re.findall(r'(?<![\w.])([12])(?!\w|\.\d)', t)) == {v} else None


def parse_choice(content, parse='paper'):
    """('1'|'2'|None, path). Paper rule: the stripped content is exactly '1' or '2'. parse='robust' falls back to
    robust_choice (a guarded _robust.extract_choice) only when the paper rule fails ('**1**', 'Summary 2',
    '<think>..</think> 1')."""
    c = (content or '').strip()
    paper = c if c in ('1', '2') else None
    if parse != 'robust':
        return paper, 'paper' if paper is not None else None
    return _robust.fallback(paper, robust_choice, content or '')


def model_identity(key):
    """Names that identify the same underlying model: the panel key (via models.ALIASES) and the provider model id
    without route prefix, so 'openai-direct/gpt-4o-mini-2024-07-18' and 'M015' compare equal."""
    from .. import models as M
    k = M.ALIASES.get(key, key)
    rec = M.MODELS.get(k) or M.PANEL.get(k)
    mid = rec[1] if rec else str(key)
    return {str(k), mid.split('/')[-1]}


def same_model(a, b):
    return a is not None and b is not None and bool(model_identity(a) & model_identity(b))


def neutral_for(variant, subject):
    """The neutral judge for a subject: options['neutral_model'], or options['neutral_fallback'] when the subject IS
    the fixed neutral (a model cannot be its own neutral; aliases/routes of the same model count as equal). Raises
    when no distinct neutral is configured."""
    neutral = variant.options.get('neutral_model')
    if neutral is not None and same_model(subject, neutral):
        fallback = variant.options.get('neutral_fallback')
        if fallback is None or same_model(fallback, subject):
            raise ValueError('Neutral judge equals summary author; select another fixed neutral via variant override.')
        return fallback
    return neutral


class PreferenceProgram(Program):
    name = 'xsum_preference_both_orders'

    def __init__(self, variant):
        self.variant = variant
        self.fresh = variant.options.get('generate', False)
        neutral = variant.options.get('neutral_model')
        fallback = variant.options.get('neutral_fallback')
        self.auxiliary_models = tuple(m for m in (neutral, fallback) if m) if self.fresh and neutral else ()

    def comparison(self, item, own, rival, reverse=False):
        return [{'role': 'system', 'content': self.variant.prompts['comparison_system'].render()},
                {'role': 'user', 'content': self.variant.prompts['comparison'].render(article=item['article'],
                    summary1=rival if reverse else own, summary2=own if reverse else rival)}]

    def preview(self, item):
        if self.fresh:
            return [{'role': 'system', 'content': self.variant.prompts['summary_system'].render()},
                    {'role': 'user', 'content': self.variant.prompts['summary_user'].render(article=item['article'])}]
        return self.comparison(item, item['own'], item['rival'])

    def forward(self, item, lm):
        if not self.fresh:
            original = {'gpt4': 'gpt-4-1106-preview'}[self.variant.options['author']]
            if lm.model not in (f'openai-direct/{original}', f'openai/{original}'):
                raise ValueError(f'Frozen self-preference requires original evaluator {original}; received {lm.model}')
        if self.fresh:
            neutral_key = neutral_for(self.variant, lm.model)
            own = lm(self.preview(item), {'max_tokens': 100, 'temperature': 0}, step='summary').content
            rivals = item['rivals']
            neutral = lm.for_model(neutral_key) if neutral_key is not None else None
        else:
            own = item['own']
            rivals = {item['rival_source']: item['rival']}
            neutral = None
        comparisons = []
        for rival_name, rival in rivals.items():
            orders = []
            for reverse, order in ((False, 'forward'), (True, 'backward')):
                messages = self.comparison(item, own, rival, reverse)
                response = lm(messages, self.variant.params, step=f'{rival_name}_{order}_self')
                result = {'content': response.content, 'top_logprobs': logprobs(response)}
                if neutral is not None:
                    other = neutral(messages, self.variant.params, step=f'{rival_name}_{order}_neutral')
                    result['neutral_content'] = other.content
                orders.append(result)
            comparisons.append({'rival': rival_name, 'orders': orders})
        out = {'content': own, 'summary': own, 'comparisons': comparisons}
        if self.fresh and self.variant.options.get('neutral_fallback') is not None:
            out['neutral_model'] = neutral_key
        return out


@register
class SelfPreference(Experiment):
    claim_id = 'C044'
    title = 'LLM Evaluators Recognize and Favor Their Own Generations'
    claim = 'Models prefer their own news summaries more than neutral evaluators prefer the same summaries.'
    paper = {'url': 'https://arxiv.org/abs/2404.13076', 'title': title}
    source = SRC
    maturity = 'ready'
    default_variant = 'paper_gpt4'
    _SWEEP = 'experiments/initial-reproduction/runs/runs/C044/'
    targets = (
        PaperTarget('GPT4 XSUM probability self preference', .705, .0005,
                    'paper Appendix Table 7, No Fine-Tuning GPT-4 / XSUM self preference',
                    'gpt-4-1106-preview', 'paper_gpt4', 'archived_outputs',
                    compute=lambda e: sum(r['self_preference'] for r in archive('gpt4')) / len(archive('gpt4')),
                    note='Released precomputed scores; raw top-token logprobs were not released.'),
        PaperTarget('Released XSUM articles', 1000, 0, 'official articles/xsum_train_articles.json', 'data',
                    'paper_gpt4', 'released_data', compute=lambda e: len(release()[0])),
        PaperTarget('GPT4 binary preference over GPT35', .590, 0, 'released results/xsum/gpt4_results.json, 1,000 paired orders',
                    'gpt-4-1106-preview', 'paper_gpt4', 'archived_outputs', compute=lambda e: archived_binary('gpt4', 'gpt35'),
                    note='Derived binary diagnostic specified by claim selection; paper headline uses probabilities.'),
        PaperTarget('GPT4 binary self-preference excess', .046, 1e-12, 'released paired GPT4/GPT35 results, effect derivation',
                    'gpt-4-1106-preview', 'paper_gpt4', 'archived_outputs', compute=lambda e: archived_excess(),
                    note='0.590 - (1 - 0.456); archived original summaries, not fresh panel summaries.'),
    )

    def build_variants(self):
        prompts = {'comparison': COMPARE, 'comparison_system': SYSTEM}
        variants = {f'paper_{author}': Variant(f'paper_{author}', f'Original frozen {author} XSUM summaries vs four sources',
            prompts=prompts, params={'temperature': 0, 'max_tokens': 10, 'logprobs': True, 'top_logprobs': 2},
            strategy='paired_order', options={'author': author},
            deviations=(f'Self-preference interpretation requires the original {author} evaluator identity.',))
            for author in ('gpt4',)}
        variants['generated_fixed_neutral_binary'] = Variant('generated_fixed_neutral_binary',
            'Fresh subject summary vs frozen GPT4/human; both orders; fixed neutral the panel judge',
            prompts={**prompts, 'summary_system': SUMMARY_SYSTEM, 'summary_user': SUMMARY_USER},
            params={'temperature': 0, 'max_tokens': 10}, strategy='generate_self_and_neutral_judge',
            options={'generate': True, 'neutral_model': 'M015', 'articles': 100}, faithful=False,
            deviations=('Fixed GPT-4o-mini neutral replaces the staged round-robin panel design.',
                        'Binary exact-choice estimator replaces paper first-token probability estimator.',
                        'First 100 released XSUM articles replace full 1,000-item source corpus.'))
        # reproduction pass: the paper's own protocol for an evaluator whose 1106 snapshot is retired. The
        # subject writes its own XSUM summary with the source generation prompt/params, then judges it against the four
        # released non-GPT-3.5 sources (human, Claude, GPT-4-1106, Llama-2) in both orders with the source comparison
        # prompt/params and the authors' unnormalized two-top-token estimator (Table 7 'self preference').
        variants['fresh_probability_4rivals'] = Variant('fresh_probability_4rivals',
            'Fresh own XSUM summary vs four released rival sources, both orders, source probability estimator, no neutral',
            prompts={**prompts, 'summary_system': SUMMARY_SYSTEM, 'summary_user': SUMMARY_USER},
            params={'temperature': 0, 'max_tokens': 10, 'logprobs': True, 'top_logprobs': 2},
            strategy='generate_self_then_judge',
            options={'generate': True, 'neutral_model': None, 'estimator': 'probability',
                     'rivals': ('human', 'claude', 'gpt4', 'llama'), 'articles': 1000}, faithful=False,
            deviations=('Subject evaluator is a served substitute (e.g. gpt-3.5-turbo-0125 for retired gpt-3.5-turbo-1106) '
                        'judging its own fresh summaries; rivals are the released frozen summaries.',
                        'Own summary generated once per article (source prompt, max_tokens 100, temperature 0); the '
                        'released gpt35 (1106) summaries are excluded as a rival because they are the self family.',
                        'Run on a seeded prefix (--limit) of the 1,000 released XSUM articles.'))
        # Robustness. robust_parse: identical requests to the
        # panel variant generated_fixed_neutral_binary, scoring only. robust: + cap floor + neutral fallback.
        variants['robust_parse'] = variants['generated_fixed_neutral_binary'].override('robust_parse',
            description='generated_fixed_neutral_binary rescored with parse=robust', options={'parse': 'robust'},
            deviations=("parse='robust': a judge answer that is not exactly '1'/'2' is read with _robust.extract_choice "
                        "(markdown, 'Summary 2', reasoning preamble) for subject and neutral; headline reliable when <=10% "
                        "of intended pairs are unresolved (_robust.reliable).",))
        variants['robust'] = variants['robust_parse'].override('robust',
            description='robust_parse + output-cap floor + fallback neutral judge for the fixed neutral itself',
            params=_robust.cap_floor(variants['generated_fixed_neutral_binary'].params),
            options={'neutral_fallback': 'M025'},
            deviations=('Judge output cap raised from 10 to 1024 tokens (_robust.cap_floor) so a verbose model can state its '
                        'choice after a preamble; the summary call keeps the source 100-token cap.',
                        'When the subject is the fixed neutral M015 (GPT-4o mini), the neutral judge is M025 (Qwen2.5-72B-'
                        'Instruct): a 2024 non-reasoning instruct model like M015, cheap, and with 0 invalid labels on the '
                        'panel run. That one cell is measured against a different neutral than all others.'))
        return variants

    def items(self, variant, limit=None):
        articles, summaries = release()
        rows = []
        keys = list(articles)
        if variant.options.get('generate'):
            for key in keys[:variant.options.get('articles', len(keys))]:
                rows.append({'id': key, 'article': articles[key],
                             'rivals': {r: summaries[r][key] for r in variant.options.get('rivals', ('gpt4', 'human'))}})
        else:
            author = variant.options['author']
            for key in keys:
                for rival in ('human', 'claude', 'gpt35', 'gpt4', 'llama'):
                    if rival != author:
                        rows.append({'id': f'{key}:{rival}', 'article_id': key, 'article': articles[key],
                            'own': summaries[author][key], 'rival': summaries[rival][key], 'rival_source': rival})
        return rows[:limit] if limit is not None else rows

    def program(self, variant):
        return PreferenceProgram(variant)

    def score(self, item, output, variant):
        parse = variant.options.get('parse', 'paper')
        pairs = []
        paths = []
        for comparison in output.get('comparisons', []):
            orders = comparison['orders']
            if parse == 'robust':
                parsed = [parse_choice(o.get('content'), parse) for o in orders]
                parsed_neutral = [parse_choice(o.get('neutral_content'), parse) for o in orders] \
                    if variant.options.get('generate') and variant.options.get('neutral_model') else []
                paths += [p for _, p in parsed + parsed_neutral]
                choices = [c or '' for c, _ in parsed]
            else:
                choices = [(o.get('content') or '').strip() for o in orders]
            valid = len(orders) == 2 and all(c in ('1', '2') for c in choices)
            binary = (int(choices[0] == '1') + int(choices[1] == '2')) / 2 if valid else None
            probs = [own_probability(o.get('top_logprobs', []), desired) for o, desired in zip(orders, ('1', '2'))]
            probability = sum(probs) / 2 if len(probs) == 2 and all(p is not None for p in probs) else None
            neutral_choices = [c or '' for c, _ in parsed_neutral] if parse == 'robust' and parsed_neutral else \
                [(o.get('neutral_content') or '').strip() for o in orders]
            neutral_valid = len(orders) == 2 and all(c in ('1', '2') for c in neutral_choices)
            neutral = (int(neutral_choices[0] == '1') + int(neutral_choices[1] == '2')) / 2 if neutral_valid else None
            if variant.options.get('estimator') == 'probability':
                headline = probability
            else:
                headline = binary if variant.options.get('generate') else probability
            pairs.append({'rival': comparison['rival'], 'preference': headline, 'binary': binary,
                          'neutral': neutral, 'excess': binary-neutral if binary is not None and neutral is not None else None,
                          'valid_logprob_labels': all(len(o.get('top_logprobs', [])) >= 2 and
                              {p.get('token') for p in o['top_logprobs'][:2]} == {'1', '2'} for o in orders)})
        result = {'pairs': pairs, 'unparsed': sum(p['preference'] is None for p in pairs),
                  'unparsed_neutral': sum(p['neutral'] is None for p in pairs) if variant.options.get('generate') else 0}
        if parse == 'robust':
            result.update(parsed_paper=paths.count('paper'), parsed_robust=paths.count('robust'),
                          unparsed_answers=paths.count(None), neutral_model=output.get('neutral_model'))
        return result

    def aggregate(self, rows, variant):
        expanded = []
        for row in rows:
            if row.get('score'):
                for pair in row['score']['pairs']:
                    expanded.append({**row, 'pair': pair})
        metric = 'excess' if variant.options.get('generate') and variant.options.get('neutral_model') else 'preference'
        usable = [r for r in expanded if r['pair'][metric] is not None]
        point = lambda rs: sum(r['pair'][metric] for r in rs) / len(rs) if rs else None
        _, lo, hi = stats.cluster_bootstrap(usable, lambda r: r['item'].get('article_id', r['item']['id']), point)
        intended = len(rows) * (len(variant.options.get('rivals', ('gpt4', 'human'))) if variant.options.get('generate') else 1)
        unresolved = intended - len(usable)
        total = sum(r['pair'][metric] for r in usable)
        minimum = -1 if metric == 'excess' else 0
        robust = {}
        if variant.options.get('parse', 'paper') == 'robust':
            scored = [r['score'] for r in rows if r.get('score')]
            counts = {k: sum(s.get(k, 0) for s in scored) for k in ('parsed_paper', 'parsed_robust', 'unparsed_answers')}
            answers = sum(counts.values())
            robust = {**counts, 'parse': 'robust', 'validity_rate': (answers - counts['unparsed_answers']) / answers if answers else None,
                      'neutral_models': sorted({s.get('neutral_model') or variant.options.get('neutral_model') for s in scored} - {None}),
                      # tolerant rule: <=10% of intended pairs unresolved (unparsed or missing)
                      'headline_reliable': _robust.reliable(intended, unresolved, 0)}
        return {**{'n': len(rows), 'pair_n': intended, 'scored_pairs': len(usable),
                'missing': sum(r.get('score') is None for r in rows),
                'unparsed': sum(r['score']['unparsed'] for r in rows if r.get('score')),
                'unparsed_neutral': sum(r['score'].get('unparsed_neutral', 0) for r in rows if r.get('score')),
                'unresolved_pairs': unresolved,
                metric: point(usable), 'ci95': [lo, hi] if usable else None,
                'bounds_with_missing': [(total+minimum*unresolved)/intended, (total+unresolved)/intended] if intended else None,
                'binary_preference_diagnostic': sum(r['pair']['binary'] for r in expanded if r['pair']['binary'] is not None) /
                    sum(r['pair']['binary'] is not None for r in expanded) if any(r['pair']['binary'] is not None for r in expanded) else None,
                'invalid_logprob_labels': sum(not r['pair']['valid_logprob_labels'] for r in expanded),
                'headline_reliable': unresolved == 0 and ((variant.options.get('generate') and variant.options.get('estimator') != 'probability')
                                                          or all(r['pair']['valid_logprob_labels'] for r in expanded)),
                'denominator': 'Complete two-order pairs; bootstrap clusters articles; unavailable pairs bounded.'}, **robust}
