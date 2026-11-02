import pytest

from shelflife.experiments.c051_premise_order import C051, MissingOriginalProtocol, generate_problem, check_proof, inversions

exp = C051()


def test_original_fail_fast_and_reconstruction_structure():
    with pytest.raises(MissingOriginalProtocol, match='dataset/generator'):
        exp.items(exp.variant())
    variant = exp.variant('reconstructed_logic12')
    rows = exp.items(variant)
    assert len(rows) == 200
    assert not variant.faithful
    assert exp.items(variant, 0) == []
    for row in rows:
        assert len(row['rules']) == 12
        assert inversions(row['orders']['shuffled']) == 33
        assert row['orders']['backward'] == list(reversed(range(12)))
        assert all(1 <= len(rule['premises']) <= 3 for rule in row['rules'])
        # Every rule is essential: deleting any one makes query unprovable.
        for removed in range(12):
            known = set(row['facts'])
            for i, rule in enumerate(row['rules']):
                if i != removed and set(rule['premises']) <= known:
                    known.add(rule['conclusion'])
            assert row['query'] not in known
    assert rows == exp.items(variant)


def test_ordered_proof_checker_literal_fixtures():
    item = {'facts': ['a'], 'rules': [{'premises': ['a'], 'conclusion': 'b'},
                                     {'premises': ['b'], 'conclusion': 'c'}], 'query': 'c'}
    assert check_proof(item, 'Alice is a (fact). Alice is b by rule If a, then b. Alice is c by rule If b, then c.')['correct']
    assert not check_proof(item, 'Alice is c.')['correct']
    assert check_proof(item, 'Alice is d. Alice is b. Alice is c.')['error'] == 'hallucinated_or_premature_fact'
    assert check_proof(item, 'If a, then c. Alice is c.')['error'] == 'hallucinated_rule'
    assert check_proof(item, '**Yes**')['unparsed']
    assert check_proof(item, 'We cannot conclude that Alice is c.')['error'] == 'refutation'
    assert check_proof(item, '')['unparsed']


def test_prompt_order_only_changes_rules():
    row = generate_problem(0)
    variant = exp.variant('reconstructed_logic12')
    program = exp.program(variant)
    messages = [program.messages(row, c)[0]['content'] for c in ('forward', 'backward', 'shuffled')]
    for text in messages[1:]:
        assert text.split('Facts: ')[1] == messages[0].split('Facts: ')[1]
        rules = lambda t: {s.strip() for s in t.split('\n\nFacts:')[0].removeprefix('Rules: ').split('.') if s.strip()}
        assert rules(text) == rules(messages[0])


def test_reconstructed_checker_scope_labels_and_paraphrases():
    item = {'facts': ['a'], 'rules': [{'premises': ['a'], 'conclusion': 'b'}], 'query': 'b'}
    # This checker validates recognized ordered assertions, not fact/rule labels.
    assert check_proof(item, 'Alice is a (fact). Alice is b (fact).')['correct']
    # Unsupported proof-language paraphrases are not silently treated as proofs.
    assert check_proof(item, 'Given a holds, b follows from the implication.')['unparsed']


def test_missing_pairs_bounds():
    score = {c: {'correct': c == 'forward', 'unparsed': False} for c in ('forward', 'shuffled', 'backward')}
    result = exp.aggregate([{'item': {'id': 'a'}, 'score': score}, {'item': {'id': 'b'}, 'score': None}],
                           exp.variant('reconstructed_logic12'))
    assert result['shuffled_minus_forward'] == -1
    assert result['bounds_with_missing'] == [-1, 0]


CHAIN = {'facts': ['cote', 'kone'], 'rules': [{'premises': ['cote'], 'conclusion': 'duve'},
                                             {'premises': ['duve', 'kone'], 'conclusion': 'jag'}], 'query': 'jag'}


def test_goal_statements_are_not_assertions_regression_sweep_10_07():
    """Reproduction pass bug: the earlier checker read 'whether Alice is **jag**' as asserting the query."""
    text = ("Let's analyze step-by-step whether Alice is **jag**.\n### Step 1: Check if Alice is **duve**\n"
            "- Alice is cote (Fact)\n- Rule: If cote, then duve.\n- Therefore, Alice is **duve**.\n"
            "### Step 2: Is Alice jag?\n- Alice is duve (Step 1) and kone (Fact)\n- Rule: If duve and kone, then jag.\n"
            "- Therefore, Alice is **jag**.\n\n**Yes, Alice is jag.**")
    assert check_proof(CHAIN, text) == {'correct': True, 'unparsed': False, 'error': None}


def test_rule_list_proofs_and_strict_sensitivity():
    text = '1. Alice is cote (fact)\n2. If cote, then duve (rule)\n3. If duve and kone, then jag (rule)\n\nTherefore, Alice is jag.'
    assert check_proof(CHAIN, text)['correct']
    assert check_proof(CHAIN, text, apply_quoted=False)['error'] == 'hallucinated_or_premature_fact'
    # A bare restatement of the rules (no application marker) licenses nothing.
    bare = 'Rules:\nIf cote, then duve.\nIf duve and kone, then jag.\nTherefore, Alice is jag.'
    assert check_proof(CHAIN, bare)['error'] == 'hallucinated_or_premature_fact'
    # An applied rule with unmet premises licenses nothing.
    unmet = '1. Using rule: If duve and kone, then jag.\nTherefore, Alice is jag.'
    assert check_proof(CHAIN, unmet)['error'] == 'hallucinated_or_premature_fact'


def test_premature_hallucinated_and_refutation_cases():
    assert check_proof(CHAIN, 'Alice is cote. Alice is jag. Alice is duve.')['error'] == 'hallucinated_or_premature_fact'
    assert check_proof(CHAIN, 'Alice is cote. Alice is wozz.')['error'] == 'hallucinated_or_premature_fact'
    assert check_proof(CHAIN, 'Using rule: If kone, then jag. Alice is jag.')['error'] == 'hallucinated_rule'
    valid = 'Alice is cote. By rule If cote, then duve, Alice is duve. By rule If duve and kone, then jag, Alice is jag.'
    assert check_proof(CHAIN, valid)['correct']
    assert check_proof(CHAIN, valid + '\nHowever, we cannot conclude that Alice is jag.')['error'] == 'refutation'
    assert check_proof(CHAIN, 'Alice is cote (Fact). Alice is not duve (Fact). Therefore, Alice is not jag.')['error'] == 'refutation'
    assert check_proof(CHAIN, valid.replace('By rule If duve and kone, then jag, Alice is jag.', ''))['error'] == 'incomplete_derivation'


def test_two_arm_variant_conditions_and_aggregate():
    variant = exp.variant('reconstructed_logic12_fs')
    assert not variant.faithful
    calls = []
    class LM:
        def __call__(self, messages, params, step):
            calls.append(step)
            class C:
                def summary(self):
                    return {'content': ''}
            return C()
    out = exp.program(variant).forward(generate_problem(0), LM())
    assert calls == ['forward', 'shuffled'] and set(out) == {'forward', 'shuffled'}
    score = exp.score(generate_problem(0), out, variant)
    assert set(score) == {'forward', 'shuffled'} and score['forward']['unparsed']
    agg = exp.aggregate([{'item': {'id': 'x'}, 'score': score}], variant)
    assert set(agg['conditions']) == {'forward', 'shuffled'} and agg['shuffled_minus_forward'] == 0


# ------------------------------------------------------------------ robustness
from pathlib import Path

from shelflife.experiments.c051_premise_order import robust_check

RUN = Path(__file__).resolve().parents[3] / 'experiments/run_minimal_setups/runs/C051-reconstructed_logic12_fs-L80'


def test_robust_variants_params():
    fs, rp, rb = (exp.variant(v) for v in ('reconstructed_logic12_fs', 'robust_parse', 'robust'))
    assert rp.params == fs.params and rp.prompts == fs.prompts and rp.strategy == fs.strategy
    assert rp.options == {**fs.options, 'parse': 'robust', 'conclusion_first': 'accept'} and not rp.faithful
    assert rb.options['conclusion_first'] == 'accept' and 'conclusion_first' not in fs.options
    assert rb.params == {**fs.params, 'max_tokens': 4096} and rb.options['min_cap_reasoning'] == 16384
    assert exp.program(rb).case_extra == {'min_cap_reasoning': 16384}
    assert exp.program(rp).case_extra == {} and exp.program(fs).case_extra == {}
    # the faithful guard and the generation variant are unchanged
    assert exp.variant('paper_original').faithful and 'parse' not in fs.options


def test_robust_check_modern_formats():
    valid = ('**Step 1:** Alice is **cote** (fact).\n**Step 2:** By rule If cote, then duve, Alice is **duve**.\n'
             '**Step 3:** By rule If duve and kone, then jag, Alice is **jag**.\n\n**Final answer:** Yes, Alice is jag.')
    assert robust_check(CHAIN, valid)['correct'] and robust_check(CHAIN, valid)['parse_path'] == 'paper'
    # a reasoning block that never reaches an assertion, then a proof: identical verdict to the paper checker
    think = '<think>Need jag? check duve first</think>\n' + valid
    assert robust_check(CHAIN, think)['correct'] == check_proof(CHAIN, think)['correct']
    # truncated mid-proof before any derivation, empty, and a refusal stay unparsed/refuted, never correct
    assert robust_check(CHAIN, 'Here is the step-by-step logical derivation:\n\nStep 1: Alice is')['parse_path'] is None
    assert robust_check(CHAIN, '')['unparsed'] and robust_check(CHAIN, '')['parse_path'] is None
    refusal = robust_check(CHAIN, 'We cannot conclude that Alice is jag.')
    assert refusal['error'] == 'refutation' and not refusal['correct'] and refusal['parse_path'] == 'paper'
    # LaTeX-wrapped symbols: the fallback path only runs when the raw text is unparsed
    latex = 'Alice is \\(\\text{cote}\\). By rule If cote, then duve, Alice is \\(\\text{duve}\\). By rule If duve and kone, then jag, Alice is \\(\\text{jag}\\).'
    assert robust_check(CHAIN, latex)['correct']


def test_cut_off_arms_are_unparsed_whatever_the_checker_says():
    valid = 'Alice is cote. By rule If cote, then duve, Alice is duve. By rule If duve and kone, then jag, Alice is jag.'
    cut = {'finish_reason': 'length', 'status': 'truncated'}
    for text in (valid, 'Alice is cote. Alice is jag.', ''):
        r = robust_check(CHAIN, text, cut)
        assert r['unparsed'] and r['parse_path'] is None and r['cut_off']
    assert robust_check(CHAIN, valid, {'finish_reason': 'stop'})['correct']


def test_robust_aggregate_excludes_unparsed_items_and_is_tolerant():
    rp = exp.variant('robust_parse')
    ok = {'correct': True, 'unparsed': False, 'correct_strict': True, 'parse_path': 'paper'}
    bad = {'correct': False, 'unparsed': False, 'correct_strict': False, 'parse_path': 'paper'}
    unp = {'correct': False, 'unparsed': True, 'correct_strict': False, 'parse_path': None}
    rows = [{'item': {'id': f'i{i}'}, 'score': {'forward': ok, 'shuffled': bad}} for i in range(19)]
    rows.append({'item': {'id': 'trunc'}, 'score': {'forward': ok, 'shuffled': unp}})
    agg = exp.aggregate(rows, rp)
    assert agg['shelflife']['value'] == -1 and agg['shelflife']['n'] == 19          # truncated item not scored wrong
    assert agg['unparsed_items'] == 1 and agg['unparsed'] == 1 and agg['parsed_paper'] == 39
    assert agg['shelflife']['headline_reliable'] is True                         # 1/20 = 5% <= 10%
    paper = exp.aggregate(rows, exp.variant('reconstructed_logic12_fs'))
    assert paper['shelflife']['headline_reliable'] is False and paper['shelflife']['n'] == 20
    rows += [{'item': {'id': f't{i}'}, 'score': {'forward': unp, 'shuffled': unp}} for i in range(2)]
    assert exp.aggregate(rows, rp)['shelflife']['headline_reliable'] is False     # 3/22 > 10%


def test_truncated_arm_scored_under_robust_unparsed_under_robust_parse():
    """truncated='score' (robust, generous cap): a cut-off arm is checked like any other (no derivation = wrong);
    robust_parse (truncated='missing'): it is unparsed whatever it contains."""
    rp, rb = exp.variant('robust_parse'), exp.variant('robust')
    assert rb.options['truncated'] == 'score' and 'truncated' not in rp.options
    item = generate_problem(0)
    cut = {'content': 'Here is the step-by-step logical derivation:\n\nStep 1: Alice is', 'finish_reason': 'length',
           'status': 'truncated'}
    out = {'forward': cut, 'shuffled': cut}
    s_rb, s_rp = exp.score(item, out, rb), exp.score(item, out, rp)
    for arm in ('forward', 'shuffled'):
        assert s_rp[arm]['unparsed'] and s_rp[arm]['parse_path'] is None
        assert not s_rb[arm]['unparsed'] and s_rb[arm]['correct'] is False and s_rb[arm]['truncated_scored']
    # an empty cut-off arm (whole budget spent on hidden reasoning) is a wrong proof too
    empty = exp.score(item, {'forward': {'content': '', 'finish_reason': 'length'}, 'shuffled': cut}, rb)
    assert not empty['forward']['unparsed'] and not empty['forward']['correct']
    agg = exp.aggregate([{'item': {'id': 'x'}, 'score': s_rb}], rb)
    assert agg['unparsed_items'] == 0 and agg['truncated_scored'] == 2 and agg['shelflife']['n'] == 1
    assert exp.aggregate([{'item': {'id': 'x'}, 'score': s_rp}], rp)['unparsed_items'] == 1
    # a complete proof cut off afterwards keeps the checker's verdict under robust
    valid = 'Alice is cote. By rule If cote, then duve, Alice is duve. By rule If duve and kone, then jag, Alice is jag.'
    assert robust_check(CHAIN, valid, {'finish_reason': 'length'}, cut_missing=False)['correct']


# ------------------------------------------- scoring revision
VALID = 'Alice is cote. By rule If cote, then duve, Alice is duve. By rule If duve and kone, then jag, Alice is jag.'


@pytest.mark.parametrize('text', [
    '**Yes, Alice is jag.** Here is the step-by-step derivation:\n' + VALID,           # conclusion first (M206 style)
    'Yes. Alice is **jag**.\n\n' + VALID,                                              # "Yes." then the answer
    'Answer: Yes, Alice is jag.\n' + VALID,
    'Alice is jag, by the following steps:\n' + VALID,                                  # M204 style
    'Yes, Alice is jag.\n' + VALID + '\nTherefore, yes, Alice is jag.',                # first and last
    VALID + '\n\n**Yes, Alice is jag.**',                                              # last only
])
def test_conclusion_first_valid_proofs_accepted_by_robust_option(text):
    """Audit MAJOR: a valid proof is correct whether its conclusion is stated first, last or both (robust option);
    the reconstructed checker of the prior runs keeps rejecting an opening answer as a premature fact."""
    accept = check_proof(CHAIN, text, conclusion_first='accept')
    assert accept == {'correct': True, 'unparsed': False, 'error': None}
    assert check_proof(CHAIN, text, apply_quoted=False, conclusion_first='accept')['correct']
    opens_with_answer = not text.startswith('Alice is cote')
    assert check_proof(CHAIN, text)['correct'] is (not opens_with_answer)   # default 'reject' unchanged
    for name in ('robust_parse', 'robust'):
        out = exp.score(CHAIN, {'forward': {'content': text, 'finish_reason': 'stop'},
                                'shuffled': {'content': VALID, 'finish_reason': 'stop'}}, exp.variant(name))
        assert out['forward']['correct'] and out['shuffled']['correct']
    paper = exp.score(CHAIN, {'forward': {'content': text}, 'shuffled': {'content': VALID}},
                      exp.variant('reconstructed_logic12_fs'))
    assert paper['forward']['correct'] is (not opens_with_answer)


@pytest.mark.parametrize('text, error', [
    ('Yes, Alice is jag.', 'hallucinated_or_premature_fact'),                         # bare answer, no derivation
    ('**Yes, Alice is jag.** Here is the step-by-step derivation:', 'hallucinated_or_premature_fact'),
    ('Yes, Alice is jag. Alice is cote. Alice is jag.', 'hallucinated_or_premature_fact'),   # never derived
    ('Yes, Alice is jag. Alice is cote. By rule If cote, then duve, Alice is duve.', 'hallucinated_or_premature_fact'),
    ('Yes, Alice is jag. Alice is cote. Alice is jag. Alice is duve.', 'hallucinated_or_premature_fact'),  # step skipped
    ('Yes, Alice is jag. Alice is wozz.', 'hallucinated_or_premature_fact'),             # hallucinated fact
    ('Yes, Alice is jag. Using rule: If kone, then jag. Alice is jag.', 'hallucinated_rule'),
    ('Yes, Alice is duve and jag.\n' + VALID, 'hallucinated_or_premature_fact'),        # opening asserts a non-query step
    ('Yes, Alice is jag (rule If duve and kone, then jag).\n' + VALID, 'hallucinated_or_premature_fact'),  # premature rule use
    ('Yes, Alice is jag.\n' + VALID + '\nHowever, we cannot conclude that Alice is jag.', 'refutation'),
])
def test_conclusion_first_invalid_proofs_still_fail(text, error):
    """The opening conclusion asserts nothing under 'accept': the proof must still derive the query validly, so
    a bare restatement (which the reconstructed checker scores as a wrong proof, not unparsed) still fails."""
    for strict in (True, False):
        accept = check_proof(CHAIN, text, apply_quoted=strict, conclusion_first='accept')
        assert not accept['correct'] and not accept['unparsed'], (text, accept)
    assert check_proof(CHAIN, text, conclusion_first='accept')['error'] == error
    assert not check_proof(CHAIN, text)['correct']


def test_conclusion_first_never_rescues_what_the_proof_body_does_not_prove():
    """'accept' = scoring the text with its opening answer clause removed (the answer itself licenses nothing)."""
    body_ok, body_bad = VALID, 'Alice is duve. By rule If duve and kone, then jag, Alice is jag.'  # dropped step
    for body in (body_ok, body_bad):
        assert check_proof(CHAIN, 'Yes, Alice is jag.\n' + body, conclusion_first='accept') == check_proof(CHAIN, body)
    assert not check_proof(CHAIN, body_bad)['correct']


def test_robust_bounds_count_unparsed_pairs_9_plus_1_fixture():
    """Audit MODERATE: 9 observed zero-delta pairs + 1 unresolved pair gave bounds [0, 0]; conservative
    full-denominator bounds are [-0.1, +0.1]. Missing (no score) pairs are unresolved too."""
    rp = exp.variant('robust_parse')
    ok = {'correct': True, 'unparsed': False, 'correct_strict': True, 'parse_path': 'paper'}
    unp = {'correct': False, 'unparsed': True, 'correct_strict': False, 'parse_path': None}
    rows = [{'item': {'id': f'i{i}'}, 'score': {'forward': ok, 'shuffled': ok}} for i in range(9)]
    agg = exp.aggregate(rows + [{'item': {'id': 'u'}, 'score': {'forward': ok, 'shuffled': unp}}], rp)
    assert agg['n'] == 10 and agg['unparsed_items'] == 1 and agg['shelflife']['n'] == 9
    assert agg['shelflife']['value'] == 0 and agg['shelflife']['headline_reliable'] is True
    assert agg['bounds_with_missing'] == pytest.approx([-0.1, 0.1])
    agg = exp.aggregate(rows + [{'item': {'id': 'm'}, 'score': None}], rp)
    assert agg['bounds_with_missing'] == pytest.approx([-0.1, 0.1]) and agg['n'] == 10
    # observed deltas still enter: 8 zero + 1 drop (-1) + 1 unparsed -> [-0.2, 0.0]
    bad = {'correct': False, 'unparsed': False, 'correct_strict': False, 'parse_path': 'paper'}
    rows[0] = {'item': {'id': 'i0'}, 'score': {'forward': ok, 'shuffled': bad}}
    agg = exp.aggregate(rows + [{'item': {'id': 'u'}, 'score': {'forward': unp, 'shuffled': unp}}], rp)
    assert agg['bounds_with_missing'] == pytest.approx([-0.2, 0.0])
    assert agg['conclusion_first'] == 'accept'
