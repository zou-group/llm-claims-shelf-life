"""Premise order: reconstructed 12-rule definite-clause proof experiment.

Original generator, sampled problems and automatic proof checker were not
released. The original variant fails before execution; reconstruction uses a
fixed-seed chained DAG with 1-3 antecedents, source Figure 1 request wording,
and an explicit ordered-assertion checker. This narrower graph distribution
and reconstructed scorer cannot validate Table 6's exact original accuracy.

Robustness:
  scoring option ``parse``: 'paper' (default) or 'robust'. Under 'robust' the checker runs on the raw text first and,
  only when that is unparsed (empty or no affirmative derivation), again on ``_robust.normalize(text)`` (think blocks,
  LaTeX wrappers and markdown removed). Under robust_parse (option truncated='missing', the default) an arm that was
  cut off (finish_reason 'length') is unparsed whatever the checker says (right, wrong or unparsed), so the exclusion never selects on the outcome. An item whose either arm is
  unparsed leaves the paired contrast (counted as ``unparsed_items``; never scored as a wrong proof), and ``headline_reliable`` = ``_robust.reliable`` (<= 10% of items
  unparsed or missing) instead of "no unparsed arm at all".
  Documented exception (rows the paper checker scored that robust treats differently): cut-off arms the paper
  checker parsed (as an incomplete, premature or correct proof) are unparsed under robust (panel: 44 of 123
  cut-off arms).
  robust_parse  reconstructed_logic12_fs with parse='robust' (identical requests; offline rescoring).
                + conclusion_first='accept' (2026-10-14, audit experiments/run_minimal_setups/audits/2026-10-11_results):
                the reconstructed checker read an opening answer ("Yes, Alice is q." before the steps) as a premature
                assertion and failed otherwise valid proofs, confounding answer position with validity. Under 'accept'
                an opening conclusion clause asserts nothing and the proof must still derive q (a bare answer still
                fails). Inherited by robust. The reconstructed_logic12* variants keep 'reject' (prior-run parity).
                The robust aggregate's bounds_with_missing now counts every excluded unparsed/missing pair as
                unresolved over the full item denominator (it previously dropped them).
  robust        robust_parse + min_cap_reasoning 16384 (reasoning models exhausted the 4,096 cap on the panel: M144 39%,
                M183 29% of arms truncated with empty or cut-off proofs) and the _robust.cap_floor (no-op at 4,096).
                + truncated='score' (_robust.truncation_missing): at these caps a cut-off arm is a failed proof, not a cap
                artefact. It is checked like any other arm (paper checker, then the normalized fallback); if no
                derivation was reached before the cut (or the text is empty) it is a wrong proof (the paper checker
                counts it not correct), never unparsed; counted as ``truncated_scored``. So for robust the documented
                exception above does not apply (cut-off arms are scored as in the paper), and a second one does:
                cut-off arms with no affirmative derivation (paper checker: not correct, flagged unparsed) are wrong,
                not unparsed. robust_parse keeps 'missing'.
"""
import random
import re
from collections import Counter

from .. import stats
from . import _robust
from ..paths import DATA
from ..program import Program
from ..prompts import PromptTemplate, Source
from ..registry import Experiment, PaperTarget, Variant, register

REPO = DATA / 'sources/C051_arxiv'
SRC = Source(repo=str(REPO), path='02-benchmark.tex', locator='Section 2.1 definite-clause protocol',
             url='https://arxiv.org/abs/2402.08939')
PROMPT_SRC = Source(repo=str(REPO), path='img/figure1.png', locator='Rules/Facts/Query request, inspected visually')
PROMPT = PromptTemplate('user', 'Rules: {rules}\n\nFacts: {facts}\n\nQuery: Is Alice {query}? '
                        'In each step, indicate whether you are using a fact, or a rule.', source=PROMPT_SRC)
WORDS = 'cote kone duve ag jag tess riff feg vide dax rabe thoxe hepe nire hume sene wozz nup zeck rebe ene heff'.split()
CONDITIONS = ('forward', 'shuffled', 'backward')


class MissingOriginalProtocol(RuntimeError):
    pass


def conditions_of(variant):
    return tuple(variant.options.get('conditions', CONDITIONS))


def inversions(order):
    return sum(order[i] > order[j] for i in range(len(order)) for j in range(i + 1, len(order)))


def generate_problem(index, count=12, seed=51):
    rng = random.Random(seed + index)
    symbols = rng.sample(WORDS, count + 5)
    facts, conclusions = symbols[:5], symbols[5:]
    rules = []
    for step, conclusion in enumerate(conclusions):
        mandatory = conclusions[step - 1] if step else rng.choice(facts)
        available = facts + conclusions[:step]
        other = [word for word in available if word != mandatory]
        arity = rng.randint(1, 3)
        premises = [mandatory] + rng.sample(other, arity - 1)
        rng.shuffle(premises)
        rules.append({'premises': premises, 'conclusion': conclusion})
    # 12 rules have 66 pairwise comparisons; exactly 33 inversions gives tau=0.
    # Uniform rejection sample conditioned on that inversion count, bounded.
    order = list(range(count))
    target = count * (count - 1) // 4
    for _ in range(10000):
        rng.shuffle(order)
        if inversions(order) == target:
            break
    else:
        raise RuntimeError('Unable to sample tau-zero permutation within fixed bound')
    return {'id': f'logic12-{index:04d}', 'facts': facts, 'rules': rules, 'query': conclusions[-1],
            'orders': {'forward': list(range(count)), 'backward': list(reversed(range(count))), 'shuffled': order}}


SYMBOL = re.compile(r'[a-z]+')
RULE_TEXT = re.compile(r'\bif\s+((?:alice\s+is\s+)?[a-z]+(?:\s*(?:,|and|&)\s*(?:alice\s+is\s+)?[a-z]+){0,4}),?\s+then\s+'
                       r'(?:alice\s+is\s+)?([a-z]+)')
GOAL = re.compile(r"\?|\b(?:whether|check|checking|need|needs|needed|require|requires|required|want|goal|query|can we|could|"
                  r"would|if|unknown|not|isn't|cannot|can't|assume|suppose|let's|lets|follows from|follow from|depends|"
                  r"to (?:prove|show|determine|derive|establish|get|conclude|find|obtain|verify|confirm|see|check))\b")
REFUTE = re.compile(r"cannot|can't|can not|not possible|unable|insufficient|not enough|undetermined|not (?:be )?(?:proved|proven|derived|"
                    r"concluded|determined|established)|\bno\b[,.]|is not {q}\b|isn't {q}\b|not {q}\b")


WINDOW = 8  # a derived fact's rule premises must be named within the last 8 non-empty clauses (the shown step) ...
CITED = 30  # ... or its rule quoted within the last 30 rule quotations (nested backward-chaining proofs)


APPLIED = re.compile(r'\b(?:rule|step|using|use|by|apply|applying|applied)\b')


def check_proof(item, text, apply_quoted=True, conclusion_first='reject'):
    """Reconstructed full-proof checker (reproduction pass rewrite; the paper's grader was never released).

    Paper criterion (Section 2.1): correct only if the generated proof is completely valid: no hallucinated rules or
    facts, query derived. Implementation: markdown is stripped and the text split into clauses (newline . ; ! ?).
    Quoted rules ("If a and b, then c", optionally with "Alice is") must exist in the problem (else
    hallucinated_rule) and are removed from the clause. Clauses with goal/conditional/negation language (whether,
    check, need, if, to prove, ?, not, ...) assert nothing. In every other clause each problem-vocabulary symbol is
    asserted to hold; the clause's symbols are closed under the rules from what earlier clauses established (so
    "Alice is c because Alice is b" is fine when b is known), and a rule only licenses a new fact when its premises
    are named within the last WINDOW clauses or the rule itself was quoted among the last CITED rule quotations (the
    step must be shown: a dropped step is an error); a symbol still not
    derivable is a premature or hallucinated fact. Correct = no such error, every one of the 12 required conclusions
    (incl. the query) established, and the last verdict sentence (mentioning yes/no/the query) in the final 600
    characters does not refute the query. A quoted rule in an assertive clause marked as applied (rule / step /
    using / by / apply) whose premises are already established also establishes its conclusion, so rule-list proofs
    ("1. Alice is a (fact) 2. If a, then b (rule) ... Therefore, Alice is q.") are valid in order; bare restatements of
    the rule list (no marker) establish nothing (apply_quoted=False gives the stricter assertion-only sensitivity
    scorer). Text with no derived assertion and no
    refutation is unparsed.
    conclusion_first='reject' (default; reconstructed scorer of every run before 2026-10-14) reads an opening answer
    ("Yes, Alice is q." before any step) as a premature assertion of q, so a valid proof that states its answer first
    fails. conclusion_first='accept' (robust variants, 2026-10-14 audit fix) treats an opening conclusion clause
    (nothing derived yet, no rule quoted in the clause, and the query the only not-yet-known symbol it names) as the
    statement to be proved: it asserts nothing, and the response is correct only if the query is then derived by the
    proof; a stated conclusion the proof never derives is still a premature fact (so a bare "Yes, Alice is q." fails
    exactly as under 'reject'). Limitations: implicit derivations written without a vocabulary symbol in an assertive
    clause are missed (incomplete), and assertive clauses that merely restate a goal without goal words count as
    assertions (conservative).
    """
    if not text or not text.strip():
        return {'correct': False, 'unparsed': True, 'error': 'empty'}
    clean = re.sub(r'[*_`#|>]', ' ', text).lower()
    rules = item['rules']
    rule_set = {(frozenset(r['premises']), r['conclusion']) for r in rules}
    problem = set(item['facts']) | {r['conclusion'] for r in rules} | {w for r in rules for w in r['premises']}
    vocab = set(item.get('vocabulary', WORDS)) | problem
    query = item['query']
    known, established = set(item['facts']), set()
    error = None
    window, cited = [], []
    stated = False   # conclusion_first='accept': an opening conclusion awaiting its derivation
    for clause in re.split(r'\n|(?<=[.;!?])\s', clean):
        if not clause.strip():
            continue
        window = (window + [set(SYMBOL.findall(clause)) & vocab])[-WINDOW:]
        shown = set().union(*window)
        cited = cited[-(CITED - 1):]
        quoted = []
        def rule_check(m):
            nonlocal error
            premises = [w for w in re.split(r'\s*(?:,|and|&)\s*', re.sub(r'alice\s+is\s+', '', m.group(1))) if w]
            if all(w in vocab for w in premises) and m.group(2) in vocab:
                if (frozenset(premises), m.group(2)) not in rule_set and error is None:
                    error = 'hallucinated_rule'
                quoted.append((set(premises), m.group(2)))
                cited.append(m.group(2))
                return ' '
            return m.group(0)
        clause = RULE_TEXT.sub(rule_check, clause)
        if GOAL.search(clause):
            continue
        if apply_quoted and APPLIED.search(clause):
            # An explicitly applied rule ("Using rule: If a, then b", "If a, then b (rule)") whose premises are all
            # established licenses its conclusion; with unmet premises it licenses nothing (no error by itself).
            for premises, conclusion in quoted:
                if premises <= known and conclusion not in known and (frozenset(premises), conclusion) in rule_set:
                    known.add(conclusion); established.add(conclusion)
        mentioned = [w for w in SYMBOL.findall(clause) if w in vocab]
        pending = [w for w in dict.fromkeys(mentioned) if w not in known]
        changed = True
        while changed and pending:
            changed = False
            for w in list(pending):
                if any(r['conclusion'] == w and set(r['premises']) <= known and
                       (set(r['premises']) <= shown or w in cited) for r in rules):
                    known.add(w); established.add(w); pending.remove(w); changed = True
        if (conclusion_first == 'accept' and pending == [query] and not established and not quoted
                and error is None):
            stated = True
            continue
        if pending and error is None:
            error = 'hallucinated_or_premature_fact'
    verdicts = [c for c in re.split(r'\n|(?<=[.;!?])\s', clean[-600:])
                if re.search(rf'\b(?:yes|no|{re.escape(query)})\b', c) and not c.strip().endswith('?')]
    refuted = bool(verdicts) and bool(re.search(REFUTE.pattern.format(q=re.escape(query)), verdicts[-1]))
    if refuted:
        return {'correct': False, 'unparsed': False, 'error': 'refutation'}
    if stated and query not in established and error is None:
        error = 'hallucinated_or_premature_fact'   # the stated conclusion is never derived: same as 'reject'
    if error:
        return {'correct': False, 'unparsed': False, 'error': error}
    if not established:
        return {'correct': False, 'unparsed': True, 'error': 'no_affirmative_derivation'}
    needed = {r['conclusion'] for r in rules}
    correct = query in established and needed <= established
    return {'correct': correct, 'unparsed': False, 'error': None if correct else 'incomplete_derivation'}


def robust_check(item, text, call=None, cut_missing=True, conclusion_first='reject'):
    """parse='robust': check_proof on the raw text; only if that is unparsed, on _robust.normalize(text).
    cut_missing=True (variant option truncated='missing', robust_parse): an arm that was cut off (finish_reason
    'length' / status 'truncated') is unparsed whatever the checker says, so excluding it never depends on the outcome.
    cut_missing=False (truncated='score', robust): a cut-off arm is checked like any other; if the checker finds no
    affirmative derivation (or the text is empty) it is a wrong proof, as the paper checker counts it (the model spent
    the whole generous budget without completing the proof), and it is flagged ``truncated_scored``.
    conclusion_first is passed to every check_proof call (robust variants use 'accept')."""
    check = lambda t, **k: check_proof(item, t, conclusion_first=conclusion_first, **k)
    call = call or {}
    cut = call.get('finish_reason') == 'length' or call.get('status') == 'truncated'
    if cut and cut_missing:
        first = check(text)
        return {**first, 'unparsed': True, 'cut_off': True, 'paper_unparsed': first['unparsed'],
                'correct_strict': check(text, apply_quoted=False)['correct'], 'parse_path': None}
    result = check(text)
    used, path = text, (None if result['unparsed'] else 'paper')
    if result['unparsed']:
        norm = _robust.normalize(text)
        if norm != text:
            second = check(norm)
            if not second['unparsed']:
                result, used, path = second, norm, 'robust'
    extra = {}
    if cut:
        extra = {'cut_off': True, 'truncated_scored': True, 'paper_unparsed': result['unparsed']}
        if result['unparsed']:      # no derivation before the cut: an incomplete proof (paper checker: not correct)
            result, path = {**result, 'correct': False, 'unparsed': False}, 'paper'
    return {**result, 'correct_strict': check(used, apply_quoted=False)['correct'], 'parse_path': path,
            **extra}


class PremiseOrders(Program):
    name = 'paired_premise_orders'

    def __init__(self, variant):
        self.variant = variant
        cap = variant.options.get('min_cap_reasoning')
        self.case_extra = {'min_cap_reasoning': cap} if cap else {}

    def messages(self, item, condition):
        rules = ' '.join('If ' + ' and '.join(item['rules'][i]['premises']) + ', then ' +
                         item['rules'][i]['conclusion'] + '.' for i in item['orders'][condition])
        facts = ' '.join(f'Alice is {word}.' for word in item['facts'])
        return [{'role': 'user', 'content': self.variant.prompts['user'].render(rules=rules, facts=facts, query=item['query'])}]

    def preview(self, item):
        return self.messages(item, 'forward')

    def forward(self, item, lm):
        if self.variant.options['original']:
            raise MissingOriginalProtocol('Original 200-problem generator/data/checker unavailable; no paid call allowed.')
        return {condition: lm(self.messages(item, condition), self.variant.params, step=condition).summary()
                for condition in conditions_of(self.variant)}


@register
class C051(Experiment):
    claim_id = 'C051'
    title = 'Premise order matters in propositional proof problems'
    claim = 'Shuffled and backward rule order lowers valid proof accuracy relative to forward order.'
    paper = {'url': 'https://arxiv.org/abs/2402.08939', 'title': title}
    source = SRC
    maturity = 'ready'
    default_variant = 'paper_original'

    def build_variants(self):
        original = Variant('paper_original', 'Original 12-rule no-distractor benchmark; unavailable artifact guard',
                           prompts={'user': PROMPT}, params={'temperature': 0}, strategy='paired_premise_orders',
                           options={'original': True, 'count': 200},
                           deviations=('Exact source generator, item set and original proof grader unreleased.',))
        reconstructed = original.override('reconstructed_logic12', params={'temperature': 0, 'max_tokens': 4096},
                                          options={'original': False, 'count': 200},
                                          description='Source-constrained fixed-seed reconstruction, 200 paired problems',
                                          deviations=('Chained DAG generator is narrower than unspecified original distribution.',
                                                      'Conservative ordered-assertion proof checker reconstructed, not author scorer.',
                                                      '4096 cap and newline request serialization reconstructed.'))
        two_arm = reconstructed.override('reconstructed_logic12_fs', options={'original': False, 'count': 200,
                                                                              'conditions': ('forward', 'shuffled')},
                                         description='Reconstruction, forward and tau=0 arms only (headline contrast; '
                                                     'cost cut for the $10/$30 GPT-4-turbo snapshot)')
        robust_parse = two_arm.override(
            'robust_parse', options={'parse': 'robust', 'conclusion_first': 'accept'},
            description='reconstructed_logic12_fs scored with the robust fallback (identical requests)',
            deviations=('parse=robust: checker re-run on _robust.normalize(text) only when the raw text is unparsed; '
                        'items with an unparsed arm leave the contrast; reliable when <= 10% of items unparsed/missing',
                        "conclusion_first='accept' (2026-10-14 audit fix): an opening answer before the proof is the "
                        'statement to prove, not a premature fact; the query must still be derived by the proof'))
        robust = robust_parse.override(
            'robust', params=_robust.cap_floor(two_arm.params), options={'min_cap_reasoning': 16384, 'truncated': 'score'},
            description='robust_parse with min_cap_reasoning 16384 for new panel runs; cut-off arms are scored',
            deviations=('reasoning models get a 16,384-token output cap (4,096 truncated M144/M183 proofs on the panel)',
                        "truncated='score': at the 4,096 / 16,384 caps a cut-off arm is a failed proof, checked like any "
                        'other arm (no derivation before the cut = wrong), not unparsed'))
        return {v.name: v for v in (original, reconstructed, two_arm, robust_parse, robust)}

    def items(self, variant, limit=None):
        if variant.options['original']:
            raise MissingOriginalProtocol('Original premise-order dataset/generator missing; use explicitly non-faithful reconstruction.')
        count = variant.options['count'] if limit is None else min(variant.options['count'], limit)
        return [generate_problem(i) for i in range(count)]

    def program(self, variant):
        return PremiseOrders(variant)

    def score(self, item, output, variant):
        if variant.options['original']:
            raise MissingOriginalProtocol('Original proof checker unavailable; reconstructed parser is a separate protocol.')
        texts = {c: ((output.get(c) or {}).get('content') or '') for c in conditions_of(variant)}
        if variant.options.get('parse', 'paper') != 'robust':
            return {c: {**check_proof(item, t), 'correct_strict': check_proof(item, t, apply_quoted=False)['correct']}
                    for c, t in texts.items()}
        cut_missing = _robust.truncation_missing(variant)
        first = variant.options.get('conclusion_first', 'reject')
        return {c: robust_check(item, t, output.get(c) or {}, cut_missing, first) for c, t in texts.items()}

    def aggregate(self, rows, variant):
        if variant.options.get('parse', 'paper') == 'robust':
            return self._aggregate_robust(rows, variant)
        return self._aggregate(rows, variant)

    def _aggregate_robust(self, rows, variant):
        """Paper aggregate over items whose arms all parsed (paper or fallback); unparsed items counted, not scored."""
        arms = conditions_of(variant)
        unparsed = [r for r in rows if r.get('score') is not None and any(r['score'][c]['unparsed'] for c in arms)]
        kept = [r for r in rows if r.get('score') is None or not any(r['score'][c]['unparsed'] for c in arms)]
        out = self._aggregate(kept, variant)
        paths = Counter(r['score'][c].get('parse_path') for r in rows if r.get('score') is not None for c in arms)
        missing = sum(r.get('score') is None for r in rows)
        # 2026-10-14 audit fix: bounds over all intended items; each excluded unparsed or missing pair is unresolved
        # and can move the paired difference by +-1 (previously _aggregate(kept) dropped them, e.g. 9 observed zero
        # deltas + 1 unparsed pair gave [0, 0] instead of [-0.1, +0.1]).
        known = sum(int(r['score']['shuffled']['correct']) - int(r['score']['forward']['correct'])
                    for r in kept if r.get('score') is not None)
        unresolved = len(unparsed) + missing
        out.update(n=len(rows), parse='robust', unparsed_items=len(unparsed),
                   conclusion_first=variant.options.get('conclusion_first', 'reject'),
                   bounds_with_missing=([(known - unresolved) / len(rows), (known + unresolved) / len(rows)]
                                        if rows else None),
                   parsed_paper=paths.get('paper', 0), parsed_robust=paths.get('robust', 0), unparsed=paths.get(None, 0),
                   truncated_scored=sum(bool(r['score'][c].get('truncated_scored')) for r in rows
                                        if r.get('score') is not None for c in arms),
                   conditions_all_items=self._aggregate(rows, variant)['conditions'])
        out['shelflife']['headline_reliable'] = out['headline_reliable'] = _robust.reliable(len(rows), len(unparsed), missing)
        out['shelflife']['validity_rate'] = (len(rows) - len(unparsed) - missing) / len(rows) if rows else None
        return out

    def _aggregate(self, rows, variant):
        scored = [r for r in rows if r.get('score') is not None]
        arms = conditions_of(variant)
        conditions = {c: stats.rate_summary(rows, lambda r: r['score'][c]['correct']) for c in arms}
        for c in arms:
            conditions[c]['unparsed'] = sum(r['score'][c]['unparsed'] for r in scored)
        gap = lambda rs: sum(int(r['score']['shuffled']['correct']) - int(r['score']['forward']['correct'])
                             for r in rs) / len(rs) if rs else None
        point, lo, hi = stats.cluster_bootstrap(scored, lambda r: r['item']['id'], gap)
        missing = len(rows) - len(scored)
        known = sum(int(r['score']['shuffled']['correct']) - int(r['score']['forward']['correct']) for r in scored)
        strict = lambda rs: sum(int(r['score']['shuffled'].get('correct_strict', False)) -
                                int(r['score']['forward'].get('correct_strict', False)) for r in rs) / len(rs) if rs else None
        s_point, s_lo, s_hi = stats.cluster_bootstrap(scored, lambda r: r['item']['id'], strict)
        for c in arms:
            conditions[c]['strict_rate'] = (sum(bool(r['score'][c].get('correct_strict')) for r in scored) / len(scored)
                                            if scored else None)
        return {'n': len(rows), 'missing': missing, 'conditions': conditions, 'shuffled_minus_forward': point,
                'strict_assertion_only': {'shuffled_minus_forward': s_point, 'ci95': [s_lo, s_hi] if scored else None},
                'ci95': [lo, hi] if scored else None,
                'bounds_with_missing': [(known - missing) / len(rows), (known + missing) / len(rows)] if rows else None,
                'shelflife': {'metric': 'reconstructed_proof_accuracy_shuffled_minus_forward', 'value': point,
                             'direction': -1, 'n': len(scored), 'ci': [lo, hi] if scored else None,
                             'headline_reliable': bool(scored) and not any(r['score'][c]['unparsed'] for r in scored for c in arms)}}

    targets = (
        PaperTarget('gpt4_turbo_forward', .965, .0005, 'arXiv 2402.08939 Table 6(a), 12 rules Forward 96.5%',
                    'gpt-4-turbo (unpinned)', 'paper_original', 'unverifiable', note='Original generator/data/grader unavailable.'),
        PaperTarget('gpt4_turbo_shuffled', .808, .0005, 'arXiv 2402.08939 Table 6(a), 12 rules Shuffled 80.8%',
                    'gpt-4-turbo (unpinned)', 'paper_original', 'unverifiable',
                    note='Shuffled table pools tau .5/0/-.5; tau-zero minimal reconstruction is not this estimator.'),
        PaperTarget('gpt4_turbo_backward', .84, .0005, 'arXiv 2402.08939 Table 6(a), 12 rules Backward 84.0%',
                    'gpt-4-turbo (unpinned)', 'paper_original', 'unverifiable'),
    )
