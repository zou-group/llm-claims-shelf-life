"""C093: documents, prompts and judge requests equal the authors' code byte for byte; label and Rej* rules; the
Program end to end on a mock transport (offline)."""
import os
import random

import pytest

from shelflife import registry
from shelflife.experiments import c093_rgb_negative_rejection as M
from shelflife.ledger import MockTransport
from shelflife.runner import run
from shelflife.sourcecode import load_functions

exp = registry.get('C093')
pytestmark = pytest.mark.skipif(exp.maturity == 'draft' and os.environ.get('SHELFLIFE_INCLUDE_DRAFTS') != '1',
                                reason='draft claim: run `python -m shelflife.cli check C093`')

REJECTION = 'I can not answer the question because of the insufficient information in documents.'


def raw_instances(lang):
    import json
    return [json.loads(l) for l in (M.REPO / f'data/{lang}.json').read_text().splitlines() if l.strip()]


def openai_default_system():
    ns = load_functions(M.REPO / 'models/models.py', ['OpenAIAPIModel'])
    defaults = ns['OpenAIAPIModel'].generate.__defaults__
    assert defaults[0] == 0.7 and defaults[2] == 1     # temperature default (overridden by --temp), top_p 1
    return defaults[1]


class StubModel:
    """Stands in for OpenAIAPIModel in the authors' predict(): records (text, temperature, system)."""

    def __init__(self, reply=''):
        self.reply, self.calls = reply, []

    def generate(self, text, temperature=0.7, system=None):
        self.calls.append((text, temperature, system if system is not None else openai_default_system()))
        return self.reply


def test_items_load():
    assert len(exp.items(exp.variant('paper_en'))) == 300
    items = exp.items(exp.variant('minimal'), limit=M.MINIMAL_LIMIT)
    assert len(items) == 200 and len({x['id'] for x in items}) == 200
    assert [x['condition'] for x in items[:4]] == ['negative', 'closed_book', 'negative', 'closed_book']
    assert len({x['qid'] for x in items}) == 100
    assert all(items[i]['qid'] == items[i + 1]['qid'] for i in range(0, 200, 2))


def test_score_rules():
    v = exp.variant('minimal')
    neg, cb = exp.items(v, 2)
    s = exp.score(neg, {'content': REJECTION, 'status': 'ok', 'judge_content': 'No, the question is not addressed by '
                        'the documents.', 'judge_status': 'ok'}, v)
    assert s['rej'] is True and s['rej_star'] is True
    s = exp.score(neg, {'content': 'Sam Mikulak', 'status': 'ok', 'judge_content': 'Yes, the question is addressed by '
                        'the documents.', 'judge_status': 'ok'}, v)
    assert s['rej'] is False and s['rej_star'] is False
    s = exp.score(neg, {'content': 'x', 'status': 'ok', 'judge_content': '', 'judge_status': 'error'}, v)
    assert s['rej_star'] is None
    s = exp.score(cb, {'content': ' '.join(a if isinstance(a, str) else a[0] for a in cb['answer']), 'status': 'ok'}, v)
    assert s['closed_correct'] is True


def responder(payload):
    text = payload['messages'][-1]['content']
    if text.startswith('I will give you a question and an answer generated through document retrieval'):
        return 'No, the question is not addressed by the documents.' if REJECTION in text else \
            'Yes, the question is addressed by the documents.'
    if text.startswith('Document:\n \n\n'):        # closed book
        return 'I am not sure.'
    return REJECTION if len(text) % 2 else 'Some confident answer.'


def test_program_end_to_end_on_mock_transport(tmp_path):
    mock = MockTransport(responder)
    out = run(exp, 'minimal', ['meta-llama/llama-3.1-8b-instruct'], tmp_path / 'run', ledger=tmp_path / 'l.sqlite3',
              ceiling_usd=5, limit=20, transport=mock, workers=2, progress_every=0)
    rows = out['rows']
    assert len(rows) == 20 and all(r['status'] == 'complete' for r in rows)
    judge_payloads = [p for p in mock.payloads if p['model'] == 'openai/gpt-3.5-turbo-0613']
    assert len(judge_payloads) == 10 and len(mock.payloads) == 30
    m = out['metrics']['meta-llama/llama-3.1-8b-instruct']
    h = m['shelflife']
    assert h['n'] == 10 and m['judged'] == 10 and m['n_closed_book'] == 10 and m['closed_book_accuracy'] == 0
    assert h['value'] == pytest.approx(1 - m['rej_star']) and m['rej_star'] == m['rej_exact']
    assert set(h) >= {'metric', 'value', 'ci', 'n', 'direction', 'null_value', 'paper_value', 'definition'}
    assert h['direction'] == 1 and h['paper_value'] == 0.55


def test_aggregate_keeps_failed_calls_in_bounds():
    v = exp.variant('minimal')
    items = exp.items(v, 6)
    rows = []
    for i, x in enumerate(items):
        if i == 0:
            rows.append({'item': x, 'output': None, 'score': None, 'status': 'call_error', 'repeat': 0, 'model': 'm'})
            continue
        out = {'content': REJECTION, 'status': 'ok', 'judge_content': 'not addressed', 'judge_status': 'ok'}
        rows.append({'item': x, 'output': out, 'score': exp.score(x, out, v), 'status': 'complete', 'repeat': 0,
                     'model': 'm'})
    m = exp.aggregate(rows, v)
    assert m['n_negative'] == 3 and m['judged'] == 2 and m['unscored_subject'] == 1
    assert m['rej_star'] == 1.0 and m['rej_star_bounds_all_requests'] == [2 / 3, 1.0]


def test_judges_routable_and_variants():
    from shelflife.policies import policy_for
    for v in exp.variants.values():
        prog = exp.program(v)
        for jm in prog.auxiliary_models:
            policy_for(jm)
    assert exp.variant('minimal').options['judge_model'] == 'openai/gpt-3.5-turbo-0613'
    assert M.RUN_VARIANT in exp.variants and exp.claim_set_key == 'C093'
    for m in M.REPRO_MODELS:
        policy_for(m)
    bound, n = M.judge_cost_bound()
    assert n == 100 and bound > 0


# --- scoring deviations from the earlier pipeline validation (experiments/run_minimal_setups/runs/earlier/validation/batch1.md) ----------
def test_blank_answer_is_missing_not_rejection():
    """M144 returned empty answers that the judge called '文档信息不足… not addressed'; they must be missing."""
    v = exp.variant('minimal')
    neg = exp.items(v, 1)[0]
    for content, status in (('', 'empty'), ('   \n', 'ok')):
        s = exp.score(neg, {'content': content, 'status': status, 'judge_content': '文档信息不足，not addressed',
                            'judge_status': 'ok'}, v)
        assert s['blank'] is True and s['labels'] is None and s['rej'] is None and s['rej_star'] is None
    items = exp.items(v, 4)
    rows = []
    for i, x in enumerate(items):
        out = {'content': '' if i == 0 else REJECTION, 'status': 'empty' if i == 0 else 'ok',
               'judge_content': 'not addressed', 'judge_status': 'ok'}
        rows.append({'item': x, 'output': out, 'score': exp.score(x, out, v), 'status': 'complete', 'repeat': 0,
                     'model': 'm'})
    m = exp.aggregate(rows, v)
    assert m['n_negative'] == 2 and m['judged'] == 1 and m['blank_answer'] == 1 and m['unscored_subject'] == 1
    assert m['shelflife']['n'] == 1 and m['shelflife']['value'] == 0.0
    assert m['shelflife']['scoring_deviations'] == list(M.SCORING_DEVIATIONS)


def test_leak_screen():
    # surname, alias and date forms, from the validators' examples
    assert M.leak_hits(['Rebecka Peterson'], ['As the 2023 National Teacher of the Year, Peterson said...'])
    assert M.leak_hits(['Gayle King'], ['Gayle King accepted the Walter Cronkite Award...'])
    assert M.leak_hits([['May 7', 'May. 7', '7 May']], ['Loyola University Maryland (Championship Game) 5/7/2023 | 12:00 PM'])
    assert M.leak_hits(['Pry'], ['Brent Pry will lead the Hokies'])
    assert not M.leak_hits(['Pry'], ['Virginia Tech hired a new coach (pryor notes).'])
    assert not M.leak_hits(['12'], ['On 12 occasions the game was delayed.'])        # bare short numbers skipped
    assert not M.leak_hits(['defensive coordinator'], ['the coordinator said'])       # no surname for lowercase answers
    v = exp.variant('minimal')
    neg = [x for x in exp.items(v, 200) if x['condition'] == 'negative']
    flagged = [x['qid'] for x in neg if M.leak_hits(x['answer'], x['docs'])]
    assert 15 <= len(flagged) <= 30 and 270 in flagged          # Rebecka Peterson (validator example)


def test_leak_sensitivity_reported_and_headline_unchanged():
    v = exp.variant('minimal')
    items = [x for x in exp.items(v, 200) if x['condition'] == 'negative']
    rows = []
    for x in items:
        out = {'content': 'Some answer', 'status': 'ok', 'judge_content': 'Yes, the question is addressed by the documents.',
               'judge_status': 'ok'}
        if M.leak_hits(x['answer'], x['docs']):
            out['judge_content'] = 'No, not addressed'
        rows.append({'item': x, 'output': out, 'score': exp.score(x, out, v), 'status': 'complete', 'repeat': 0,
                     'model': 'm'})
    m = exp.aggregate(rows, v)
    ls = m['leak_sensitivity']
    assert ls['n_leak_questions'] == ls['n_leak_judged'] > 0
    assert ls['non_rejection_excluding_leak'] == 1.0 and ls['non_rejection_on_leak'] == 0.0
    assert m['shelflife']['value'] == pytest.approx(1 - ls['n_leak_judged'] / 100)
    assert m['shelflife']['leak_sensitivity_value'] == 1.0


# ------------------------------------------------------------------ robustness (parse='robust', 2026-10-10)
def test_robust_variants_requests():
    mini, rp, rb = exp.variant('minimal'), exp.variant('robust_parse'), exp.variant('robust')
    assert rp.params == mini.params and rp.prompts == mini.prompts and rp.strategy == mini.strategy
    assert rp.options['judge_model'] == mini.options['judge_model'] == M.JUDGE_MODEL
    assert rp.options['parse'] == 'robust' and not rp.faithful and not rb.faithful and rb.deviations
    x = exp.items(mini, 1)[0]
    assert exp.program(rp).case(x) == exp.program(mini).case(x)
    assert exp.program(rb).case(x) == {**exp.program(mini).case(x), 'min_cap_reasoning': M.ROBUST_MIN_CAP_REASONING}
    assert exp.program(rb).case_extra == {'min_cap_reasoning': M.ROBUST_MIN_CAP_REASONING}
    assert exp.program(mini).case_extra == {}


@pytest.mark.parametrize('answer,judge,rej_star,unparsed', [
    (REJECTION, 'No, the question is not addressed by the documents.', True, False),
    ('**Sam Mikulak**', 'Yes, the question is addressed by the documents.', False, False),
    ('Sam Mikulak', 'No, the question is not fully addressed by the documents.', False, False),   # paper rule kept
    ('Sam Mikulak', '**No**, the question is **not addressed**.', True, False),
    ('Sam Mikulak', 'I am unable to evaluate this.', None, True),       # no verdict: unparsed, not "not rejected"
    ('Sam Mikulak', '', None, True),
    ('', 'No, the question is not addressed by the documents.', None, True),   # blank answer (reasoning ate the cap)
    ('<think>hmm</think>\n' + REJECTION, 'No, the question is not addressed by the documents.', True, False),
])
def test_robust_rej_star_modern_formats(answer, judge, rej_star, unparsed):
    neg = exp.items(exp.variant('robust_parse'), 1)[0]
    out = {'content': answer, 'status': 'ok', 'judge_content': judge, 'judge_status': 'ok'}
    r = exp.score(neg, out, exp.variant('robust_parse'))
    assert (r['rej_star'], r['unparsed']) == (rej_star, unparsed)
    p = exp.score(neg, out, exp.variant('minimal'))
    if r['parse_path'] == 'paper':
        assert p['rej_star'] is r['rej_star']


def test_robust_aggregate_reliability_and_counts():
    v = exp.variant('robust_parse')
    items = [x for x in exp.items(v, 40) if x['condition'] == 'negative']
    rows = []
    for i, x in enumerate(items):
        if i == 0:
            rows.append({'item': x, 'output': None, 'score': None, 'status': 'call_error'})
            continue
        out = {'content': '' if i < 3 else 'Answer', 'status': 'ok', 'judge_status': 'ok',
               'judge_content': 'No, the question is not addressed by the documents.' if i % 2 else 'Yes, it is addressed.'}
        rows.append({'item': x, 'output': out, 'score': exp.score(x, out, v), 'status': 'complete'})
    m = exp.aggregate(rows, v)
    assert m['parse'] == {'parsed_paper': 17, 'parsed_robust': 0, 'unparsed': 2, 'missing': 1}
    assert m['headline_reliable'] is m['shelflife']['headline_reliable'] is (3 / 20 <= 0.10)
    assert m['shelflife']['n'] == 17
    assert 'parse' not in exp.aggregate(rows, exp.variant('minimal'))


