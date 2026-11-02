"""C093 — RGB negative rejection (Chen, Lin, Han, Sun, "Benchmarking Large Language Models in Retrieval-Augmented
Generation", AAAI 2024, arXiv 2309.01431).

Claim: in retrieval-augmented QA where none of the 5 retrieved documents contains the answer, models rarely give the
instructed rejection ("I can not answer the question because of the insufficient information in documents."). The best
English rejection rate was 45% (ChatGPT, Rej*, judged by ChatGPT); exact-match Rej 24.67% (Table 3).

Protocol (official repo chen700564/RGB @ 65ec39e4; evalue.py with --noise_rate 1.0 --passage_num 5, then
reject_evalue.py)
  documents  ``processdata`` at noise_rate 1: the first 5 negative (relevant but answerless) documents, shuffled after
             ``random.seed(2333)``; joined with '\\n'.
  subject    system = config/instruction.yaml ``en.system``; user = ``en.instruction`` "Document:\\n{DOCS} \\n\\nQuestion:
             \\n{QUERY}"; OpenAIAPIModel: temperature 0.2 (README command for ChatGPT; the paper gives none), top_p 1, no
             output cap.
  Rej        exact rule from ``predict``: 'insufficient information' (English) is a substring of the response -> label -1.
  Rej*       reject_evalue.py ``check``: the ChatGPT judge prompt (verbatim, with the trailing newline and 4 spaces),
             one user message to gpt-3.5-turbo with no temperature or cap; rejection iff 'not addressed' appears in the
             judge output.
  closed-book probe (minimal design) = the authors' own passage_num=0 path of evalue.py: ``instruction.format(QUERY=q,
             DOCS='')`` sent with OpenAIAPIModel.generate's default system message ("You are a helpful assistant. You
             can help me by answering my questions. You can also ask me questions."); correct iff ``checkanswer`` labels
             contain no 0 (all answer parts present). It tells "misled by noise" apart from "knows the answer".

Variants
  paper_en          default, faithful: Table 3 English, all 300 questions, subject then Rej* judge.
  minimal           RUN_VARIANT. English negative-rejection item + closed-book probe for each question, questions in
                    ``random.Random(0)`` order, interleaved (rag, closed_book) so ``--limit 200`` = 100 questions.
  robust_parse      minimal requests, parse='robust': paper Rej/Rej*; a judge reply with no
                    "addressed" verdict is unparsed instead of "not rejected"; headline reliable iff <= 10% of negative
                    items are unparsed (blank answer / no verdict) or missing (failed call).
  robust            robust_parse + min_cap_reasoning 16384 (M144 spent the 4096 cap reasoning on half the items).
                    Robust != paper on rows the paper scored: a judge reply containing no "addressed" verdict (paper:
                    "not rejected") is unparsed/missing. 0 such rows on the panel (1,900 judge replies).

Judge model: the paper's "gpt-3.5-turbo" in Sep 2023 resolved to gpt-3.5-turbo-0613, reachable as
``openai/gpt-3.5-turbo-0613`` (OpenRouter -> Azure), so the faithful judge is used. For the original-model reproduction
the subject and judge coincide, exactly as in the paper (ChatGPT judged ChatGPT).

Metrics (per model). Paper denominators: evalue/reject_evalue skip calls that raise, so failed subject or judge calls
leave the denominator; we report them as ``unscored`` with all-request bounds.
  rej_exact, rej_star                         Table 3 Rej / Rej*.
  shelflife                                    non-rejection = 1 - Rej* on English negative-rejection items, item
                                              bootstrap CI; direction +1, null 0, paper_value 0.55 (ChatGPT).
  closed_book_accuracy, non_rejection_by_closed_book  stratification of the headline by whether the model answers the
                                              question correctly without documents (contamination: 2022-23 news).
  rag_answer_contains_gold                    responses that contain the gold answer despite answerless documents.

Deviations
  * Subject temperature 0.2 is the README command, not a paper setting; top_p 1 and no cap are OpenAIAPIModel's
    defaults. Transport policies add an output cap (recorded per call).
  * The judge's demonstration answers include Chinese; kept verbatim.
  * Released predictions/judgments: none, so no target is checked against model outputs (Table 3 is ``live`` for the
    proxy gpt-3.5-turbo-0613 and ``unverifiable`` for the open models).
  * Data: ``data/en.json`` (the paper's data). ``en_refine.json`` (2024-03, corrected documents/answers) is not used.

Scoring deviations (earlier pipeline validation 2026-10-05, experiments/run_minimal_setups/runs/earlier/validation/batch1.md)
  * A blank subject answer (empty or whitespace-only content) on a negative-rejection item is MISSING: no Rej, no Rej*
    (the judge's verdict on an empty answer is ignored), counted in ``unscored_subject`` / ``blank_answer`` and in the
    all-request bounds. The authors' code would label it (Rej 0) and judge it; M144 had 3 empty answers that the judge
    called "not addressed" (= rejection).
  * Leak sensitivity (secondary; the headline is unchanged): ``leak_terms``/``leak_hits`` flag questions whose 5
    "negative" (answerless) documents contain a gold alias (word-bounded, case-insensitive; bare 1-2 digit aliases
    skipped), the surname of a capitalised 2-3 token name answer (case-sensitive), or a numeric M/D(/YYYY) form of a
    date answer. ``leak_sensitivity`` reports non-rejection without the flagged questions and on them. This is a crude
    data-noise screen (~22 of the first 100 minimal questions are flagged), not a correction of en.json.
"""
import functools
import json
import random
import re
from collections import Counter

from . import _robust
from ..policies import policy_for
from ..program import CallFailed, Program
from ..prompts import PromptTemplate, Source
from ..registry import Experiment, PaperTarget, Variant, register
from ..sourcecode import load_functions
from ..paths import PROJECT
from .. import stats

REPO = PROJECT / 'code/shelflife/data/sources/C093_RGB'
COMMIT = '65ec39e40e7dc9abb50e9bf1b4f32be3f6f16615'
URL = 'https://github.com/chen700564/RGB'
_YAML = Source(repo=str(REPO), path='config/instruction.yaml', commit=COMMIT, url=URL)


def _yaml():
    import yaml
    return yaml.load((REPO / 'config/instruction.yaml').read_text(), Loader=yaml.FullLoader)


def _from_yaml(name, lang, key, style):
    return PromptTemplate(name, _yaml()[lang][key], style=style,
                          source=Source(repo=str(REPO), path='config/instruction.yaml', locator=f'{lang}.{key}',
                                        commit=COMMIT, url=URL),
                          note='YAML value loaded with yaml.FullLoader as evalue.py does; byte-tested in test_c093')


SYSTEM_EN = _from_yaml('system', 'en', 'system', 'literal')
INSTRUCTION_EN = _from_yaml('user', 'en', 'instruction', 'format')
JUDGE = PromptTemplate.from_source('judge', Source(repo=str(REPO), path='reject_evalue.py', locator='check: prompt',
                                                   commit=COMMIT, url=URL),
                                   kind='py_var', var='prompt', note="filled with prompt.format(question=, answer=)")
CLOSED_BOOK_SYSTEM = PromptTemplate(
    'closed_book_system', 'You are a helpful assistant. You can help me by answering my questions. You can also ask me '
    'questions.', style='literal',
    source=Source(repo=str(REPO), path='models/models.py', locator='OpenAIAPIModel.generate default system', commit=COMMIT,
                  url=URL),
    note='evalue.predict calls model.generate(text, temperature) without a system when docs are empty (byte-tested)')

SUBJECT_PARAMS = {'temperature': 0.2, 'top_p': 1}
JUDGE_MODEL = 'openai/gpt-3.5-turbo-0613'
JUDGE_PARAMS = {}
JUDGE_0125 = 'openai-direct/gpt-3.5-turbo-0125'   # reproductions identity-checked ChatGPT judge
DOC_SEED = 2333
PASSAGE_NUM = 5

# --- panel run settings (earlier pipeline) ----------------------------------------------------------------------------
RUN_VARIANT = 'minimal'
MINIMAL_LIMIT = 200        # 100 English questions x (negative-rejection RAG call + closed-book probe); 100 judge calls
REPEATS = 1
REPRO_MODELS = ['openai/gpt-3.5-turbo-0613']   # the paper's gpt-3.5-turbo (Sep 2023 alias target)
REPRO_LIMIT = 200
PAPER_VALUE = 0.55          # 1 - Rej* (45.00%), ChatGPT English, Table 3
ROBUST_MIN_CAP_REASONING = 16384   # robust: M144 spent the whole 4096 cap reasoning on 49/99 negative items

SCORING_DEVIATIONS = (
    'blank (empty/whitespace) subject answer on a negative item is missing (no Rej/Rej*), not judged as a rejection',
    'leak_sensitivity (secondary): non-rejection excluding questions whose negative documents contain a gold alias, '
    'gold surname or numeric date form',
)


# ----------------------------------------------------------------------------------------------- authors' code
@functools.lru_cache(maxsize=None)
def official():
    """evalue.py processdata / checkanswer / predict and reject_evalue.py check, compiled without their imports."""
    ns = load_functions(REPO / 'evalue.py', ['processdata', 'checkanswer', 'getevalue', 'predict'])
    ns.update(load_functions(REPO / 'reject_evalue.py', ['check']))
    return ns


def docs_for(instance, passage_num=PASSAGE_NUM):
    """processdata(instance, noise_rate=1, passage_num) after random.seed(2333): negative[:5], shuffled."""
    docs = list(instance['negative'][:passage_num])
    random.Random(DOC_SEED).shuffle(docs)
    return docs


def check_answer(prediction, ground_truth):
    """evalue.py ``checkanswer`` (lowercased substring per answer part; list parts are alternatives)."""
    prediction = prediction.lower()
    if type(ground_truth) is not list:
        ground_truth = [ground_truth]
    labels = []
    for instance in ground_truth:
        flag = True
        if type(instance) == list:
            flag = False
            instance = [i.lower() for i in instance]
            for i in instance:
                if i in prediction:
                    flag = True
                    break
        else:
            instance = instance.lower()
            if instance not in prediction:
                flag = False
        labels.append(int(flag))
    return labels


def labels_for(prediction, answer):
    """The label rule of evalue.predict (English): -1 if the rejection phrase is present, else checkanswer."""
    if '信息不足' in prediction or 'insufficient information' in prediction:
        return [-1]
    return check_answer(prediction, answer)


# ----------------------------------------------------------------------------------------------- leak screen
_MONTHS = {m: i + 1 for i, m in enumerate(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct',
                                           'nov', 'dec'])}


def _alternatives(answer):
    for part in (answer if isinstance(answer, list) else [answer]):
        for a in (part if isinstance(part, list) else [part]):
            yield a.strip()


def leak_terms(answer):
    """(aliases lowercased, dates (month, day, year or None), surnames) derived from a gold answer."""
    terms, dates, surnames = set(), set(), set()
    for a in _alternatives(answer):
        md = re.match(r'^([A-Za-z]{3,})\.?\s+(\d{1,2}),?\s*(\d{4})?$', a)
        dm = re.match(r'^(\d{1,2})\s+([A-Za-z]{3,})\.?,?\s*(\d{4})?$', a)
        if md and md.group(1)[:3].lower() in _MONTHS:
            dates.add((_MONTHS[md.group(1)[:3].lower()], int(md.group(2)), md.group(3)))
        elif dm and dm.group(2)[:3].lower() in _MONTHS:
            dates.add((_MONTHS[dm.group(2)[:3].lower()], int(dm.group(1)), dm.group(3)))
        if not a or re.fullmatch(r'\d{1,2}', a):
            continue
        terms.add(a.lower())
        toks = a.split()
        if 2 <= len(toks) <= 3 and all(re.fullmatch(r"[A-Z][\w'.-]*", t) for t in toks) and len(toks[-1]) >= 4 \
                and toks[-1][:3].lower() not in _MONTHS:
            surnames.add(toks[-1])
    return terms, dates, surnames


def leak_hits(answer, docs):
    """Gold-answer evidence found in the (supposedly answerless) documents; [] if none."""
    raw = '\n'.join(docs)
    low = raw.lower()
    terms, dates, surnames = leak_terms(answer)
    hits = [t for t in sorted(terms) if re.search(r'(?<![a-z0-9])' + re.escape(t) + r'(?![a-z0-9])', low)]
    hits += [t for t in sorted(surnames) if re.search(r'(?<![A-Za-z0-9])' + re.escape(t) + r'(?![A-Za-z0-9])', raw)]
    for mo, d, y in sorted(dates, key=str):
        pat = rf'\b0?{mo}/0?{d}/' + (rf'(?:{y}|{y[2:]})\b' if y else r'\d{2,4}\b')
        if re.search(pat, raw):
            hits.append(f'{mo}/{d}/{y or ""}')
    return hits


# ----------------------------------------------------------------------------------------------- data
@functools.lru_cache(maxsize=None)
def dataset(lang):
    rows = [json.loads(line) for line in (REPO / f'data/{lang}.json').read_text().splitlines() if line.strip()]
    out = []
    for r in rows:
        out.append({'id': f'C093:{lang}:{r["id"]:03d}', 'qid': r['id'], 'lang': lang, 'query': r['query'],
                    'answer': r['answer'], 'docs': docs_for(r)})
    return tuple(out)


@functools.lru_cache(maxsize=None)
def _leaky(qid, lang):
    x = next(r for r in dataset(lang) if r['qid'] == qid)
    return bool(leak_hits(x['answer'], x['docs']))


# ----------------------------------------------------------------------------------------------- program
class RAGRejection(Program):
    """condition 'negative': subject with 5 negative documents, then the Rej* judge;
    condition 'closed_book': the authors' passage_num=0 request, no judge."""
    name = 'rag_then_judge'

    def __init__(self, variant):
        self.system, self.instruction = variant.prompts['system'], variant.prompts['user']
        self.closed_system = variant.prompts.get('closed_book_system')
        self.judge = variant.prompts['judge']
        self.params = dict(variant.params)
        self.judge_model = variant.options['judge_model']
        self.judge_params = dict(variant.options['judge_params'])
        mincap = variant.options.get('min_cap_reasoning')
        self.case_extra = {'min_cap_reasoning': mincap} if mincap else {}

    @property
    def auxiliary_models(self):
        return (self.judge_model,)

    def subject_messages(self, item):
        if item['condition'] == 'closed_book':
            return [{'role': 'system', 'content': self.closed_system.render()},
                    {'role': 'user', 'content': self.instruction.render(QUERY=item['query'], DOCS='')}]
        return [{'role': 'system', 'content': self.system.render()},
                {'role': 'user', 'content': self.instruction.render(QUERY=item['query'], DOCS='\n'.join(item['docs']))}]

    def judge_messages(self, item, prediction):
        return [{'role': 'user', 'content': self.judge.render(question=item['query'], answer=prediction)}]

    def preview(self, item):
        return self.subject_messages(item)

    # plan support (shelflife.cli plan uses case()/params_for for the first request only; judge calls are extra)
    def params_for(self, item):
        return dict(self.params)

    def case(self, item):
        return {'id': item['id'], 'messages': self.subject_messages(item), 'params': self.params_for(item),
                **self.case_extra}

    def forward(self, item, lm):
        a = lm(self.subject_messages(item), self.params, step='answer')
        out = {'content': a.content, 'status': a.status, 'finish_reason': a.finish_reason, 'judge_content': None,
               'judge_status': None}
        if item['condition'] != 'negative':
            return out
        judge = lm.for_model(self.judge_model, case_extra=lm.case_extra)
        try:
            g = judge(self.judge_messages(item, a.content), self.judge_params, step='rej_judge')
        except CallFailed as e:   # keep the answer; Rej* stays unknown
            g = e.completion
        out.update(judge_content=g.content, judge_status=g.status)
        return out


# ----------------------------------------------------------------------------------------------- metrics
def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def summarize(recs):
    """recs: one per row: {'qid','condition','scored','rej','rej_star','closed_correct','gold_in_answer'}."""
    neg = [r for r in recs if r['condition'] == 'negative']
    cb = {r['qid']: r for r in recs if r['condition'] == 'closed_book'}
    subj = [r for r in neg if r['scored']]
    judged = [r for r in subj if r['rej_star'] is not None]
    rej = _mean(r['rej'] for r in subj)
    rej_star = _mean(r['rej_star'] for r in judged)
    nonrej = lambda rs: _mean(not r['rej_star'] for r in rs) if rs else float('nan')
    if judged:
        point, lo, hi = stats.cluster_bootstrap(judged, lambda r: r['qid'], nonrej, n_boot=2000, seed=0)
    else:
        point, lo, hi = None, None, None
    k_rej = sum(r['rej_star'] for r in judged)
    strat = {}
    for key, flag in (('closed_book_correct', True), ('closed_book_not_correct', False)):
        rs = [r for r in judged if r['qid'] in cb and cb[r['qid']]['scored'] and cb[r['qid']]['closed_correct'] is flag]
        strat[key] = {'n': len(rs), 'non_rejection': _mean(not r['rej_star'] for r in rs)}
    cb_scored = [r for r in cb.values() if r['scored']]
    clean = [r for r in judged if not r.get('leak')]
    leaky = [r for r in judged if r.get('leak')]
    if clean:
        lp, llo, lhi = stats.cluster_bootstrap(clean, lambda r: r['qid'], nonrej, n_boot=2000, seed=0)
    else:
        lp, llo, lhi = None, None, None
    leak = {'n_leak_questions': len({r['qid'] for r in neg if r.get('leak')}), 'n_excluding_leak': len(clean),
            'non_rejection_excluding_leak': lp, 'ci_excluding_leak': None if llo is None else [llo, lhi],
            'n_leak_judged': len(leaky), 'non_rejection_on_leak': _mean(not r['rej_star'] for r in leaky),
            'rule': 'documents contain a gold alias, gold surname or numeric M/D(/YYYY) date form (leak_hits)'}
    return {
        'n_negative': len(neg), 'subject_scored': len(subj), 'judged': len(judged),
        'unscored_subject': len(neg) - len(subj), 'unjudged': len(subj) - len(judged),
        'rej_exact': rej, 'rej_star': rej_star,
        'rej_star_bounds_all_requests': list(stats.bounds_with_missing(k_rej, len(judged) - k_rej, len(neg) - len(judged)))
        if neg else None,
        'rag_answer_contains_gold': _mean(r['gold_in_answer'] for r in subj),
        'n_closed_book': len(cb), 'closed_book_accuracy': _mean(r['closed_correct'] for r in cb_scored),
        'non_rejection_by_closed_book': strat,
        'blank_answer': sum(1 for r in neg if r.get('blank')),
        'leak_sensitivity': leak,
        'judge_unparsed_note': 'Rej* = "not addressed" substring; any other judge text counts as not rejected (paper rule)',
        'shelflife': {
            'metric': 'negative_rejection_failure_rate',
            'value': point, 'ci': None if lo is None else [lo, hi], 'n': len(judged), 'direction': +1,
            'null_value': 0.0, 'paper_value': PAPER_VALUE,
            'definition': 'Share of English RGB questions with 5 answerless retrieved documents where the judge does not '
                          'find a rejection (1 - Rej*, paper judge prompt; ChatGPT 0.55 in Table 3). Blank answers are '
                          'missing.',
            'leak_sensitivity_value': lp,
            'scoring_deviations': list(SCORING_DEVIATIONS),
        },
    }


# ---------------------------------------------------------------------------- earlier pipeline runs
# experiments/run_minimal_setups/runs/earlier/runs/<claim>-<variant>-L<N>[-<tag>]/<model>/: one shelflife.runner directory per model.
# Registered: every directory that sent at least one request (complete or partial); tests/test_prior_runs.py
# replays each one against this config. Listed explicitly (written 2026-10-05 from the run directories; nothing is globbed at import).
# Runs made before the 2026-10-05 scorer patch replay byte-identically (the patch changed scoring only); their
# saved scores are pre-patch and the collector rescores them offline.

_R2 = 'experiments/run_minimal_setups/runs/earlier/runs'


# ----------------------------------------------------------------------------------------------- experiment
@register
class C093RgbNegativeRejection(Experiment):
    claim_id = 'C093'
    claim_set_key = 'C093'
    maturity = 'ready'
    title = 'RGB retrieval-augmented generation: negative rejection'
    claim = ('In retrieval-augmented QA where none of the 5 retrieved documents contains the answer, models rarely give '
             "the instructed rejection ('I can not answer the question because of the insufficient information in "
             "documents'); best English rejection 45% (ChatGPT, LLM-judged Rej*), exact-match Rej 24.67% (Table 3).")
    paper = {'title': 'Benchmarking Large Language Models in Retrieval-Augmented Generation',
             'authors': ['Jiawei Chen', 'Hongyu Lin', 'Xianpei Han', 'Le Sun'], 'first_date': '2023-09-04',
             'venue': 'AAAI 2024 (arXiv 2309.01431)', 'url': 'https://arxiv.org/abs/2309.01431',
             'version': 'arXiv v2 LaTeX source (2023-12)'}
    source = Source(repo=str(REPO), path='evalue.py', commit=COMMIT, url=URL)
    default_variant = 'paper_en'

    def build_variants(self):
        dev = ('subject temperature 0.2 from the README ChatGPT command (paper states none); top_p 1 and no output cap '
               'are OpenAIAPIModel defaults; transport policies add a cap',
               'judge gpt-3.5-turbo pinned to the 0613 snapshot (the alias target in Sep 2023), served via OpenRouter')
        options = {'lang': 'en', 'subset': 'all', 'conditions': ('negative',), 'judge_model': JUDGE_MODEL,
                   'judge_params': dict(JUDGE_PARAMS)}
        en = Variant('paper_en', 'Table 3 English negative rejection: 300 questions, 5 negative documents, Rej and Rej*',
                     prompts={'system': SYSTEM_EN, 'user': INSTRUCTION_EN, 'judge': JUDGE}, params=dict(SUBJECT_PARAMS),
                     strategy='rag_then_judge', options=options, deviations=dev)
        minimal = Variant('minimal', 'claim-set minimal design: English negative rejection + closed-book probe per '
                          'question, Random(0) question order, interleaved; --limit 200 = 100 questions',
                          prompts={'system': SYSTEM_EN, 'user': INSTRUCTION_EN, 'judge': JUDGE,
                                   'closed_book_system': CLOSED_BOOK_SYSTEM},
                          params=dict(SUBJECT_PARAMS), strategy='rag_then_judge',
                          options={**options, 'subset': 'shuffled', 'seed': 0, 'conditions': ('negative', 'closed_book')},
                          parent='paper_en',
                          deviations=dev + ('closed-book probe added (the authors\' passage_num=0 path), not part of '
                                            'Table 3; subset of the 300 questions in seeded order',))
        # reproductions: the OpenRouter 'openai/gpt-3.5-turbo-0613' endpoint is not the 0613 model (o200k
        # tokenizer, knows mid-2023 events), so the ChatGPT judge here is the genuine OpenAI GPT-3.5-Turbo snapshot
        # still served, 0125 (direct). Paper prompts, data, Rej and Rej* rules unchanged; judge model substituted.
        en0125 = en.override('paper_en_judge0125', description='paper_en with the Rej* judge on openai-direct/'
                             'gpt-3.5-turbo-0125 (same-family substitute for the paper\'s ChatGPT judge)',
                             options={'judge_model': JUDGE_0125},
                             deviations=('Rej* judge gpt-3.5-turbo-0125 (OpenAI direct) instead of the Sep-2023 '
                                         'gpt-3.5-turbo; the OpenRouter 0613 endpoint failed an identity check',))
        # Robustness variants. robust_parse: the panel requests
        # (minimal), parse='robust', for offline rescoring. robust: for new panel runs.
        robust_parse = minimal.override(
            'robust_parse', description='minimal requests, parse=robust (paper Rej/Rej* rules; judge text without any '
            '"addressed" verdict is unparsed; tolerant reliability)', options={'parse': 'robust'},
            deviations=('scoring: Rej* is the paper rule ("not addressed" substring) whenever the judge gives a verdict; '
                        'a judge reply with no "addressed" verdict at all is unparsed (missing) instead of "not '
                        'rejected"; blank subject answers stay missing',
                        'reliability under parse=robust: <= 10% of negative items unparsed (blank answer, no judge '
                        'verdict) or missing (failed subject or judge call) (_robust.reliable)'))
        robust = robust_parse.override(
            'robust', description=f'robust_parse with min_cap_reasoning {ROBUST_MIN_CAP_REASONING}',
            options={'min_cap_reasoning': ROBUST_MIN_CAP_REASONING},
            deviations=(f'reasoning-model output cap 4096 -> {ROBUST_MIN_CAP_REASONING} (min_cap_reasoning, subject and '
                        'judge calls): M144 returned blank visible answers on 49/99 negative items (reasoning used the '
                        'whole cap); non-reasoning models keep the policy default (no truncation on the panel)',))
        return {'paper_en': en, 'minimal': minimal,
                'paper_en_judge0125': en0125, 'robust_parse': robust_parse, 'robust': robust}

    def items(self, variant, limit=None):
        o = variant.options
        qs = list(dataset(o['lang']))
        if o['subset'] == 'shuffled':
            random.Random(o['seed']).shuffle(qs)
        rows = [{**q, 'id': f'{q["id"]}:{c}', 'condition': c} for q in qs for c in o['conditions']]
        return rows[:limit] if limit is not None else rows

    def program(self, variant):
        jm = variant.options['judge_model']
        try:
            policy_for(jm)
        except KeyError as e:
            raise ValueError(f'C093 judge_model {jm!r} is accepted by no policy') from e
        return RAGRejection(variant)

    def score(self, item, output, variant):
        text = output.get('content') or ''
        usable = output.get('status') in {'ok', 'truncated', 'empty'}
        blank = item['condition'] == 'negative' and usable and not text.strip()
        labels = labels_for(text, item['answer']) if usable and not blank else None
        rec = {'blank': blank, 'labels': labels, 'rej': None if labels is None else labels[0] == -1,
               'gold_in_answer': None if labels is None else (0 not in labels and 1 in labels)}
        if item['condition'] == 'negative':
            j = output.get('judge_content')
            judged = usable and not blank and output.get('judge_status') in {'ok', 'truncated', 'empty'} and j is not None
            rec['rej_star'] = ('not addressed' in j) if judged else None
            if variant is not None and variant.options.get('parse', 'paper') == 'robust':
                verdict = judged and 'addressed' in _robust.normalize(j).lower()
                if judged and not verdict:          # judge gave no verdict: unparsed, not "not rejected"
                    rec['rej_star'] = None
                rec['parse_path'] = 'paper' if verdict else None
                rec['unparsed'] = usable and (blank or (judged and not verdict))
        else:
            rec['closed_correct'] = rec['gold_in_answer']
        return rec

    def aggregate(self, rows, variant):
        """Per-model metrics; ``leak`` flags come from the item's own documents (``leak_hits``)."""
        recs = []
        for r in rows:
            item, s = r['item'], r.get('score')
            scored = s is not None and s.get('labels') is not None
            recs.append({'qid': item['qid'], 'condition': item['condition'], 'scored': scored,
                         'rej': s['rej'] if scored else None, 'rej_star': (s or {}).get('rej_star') if scored else None,
                         'gold_in_answer': s['gold_in_answer'] if scored else None,
                         'closed_correct': (s or {}).get('closed_correct') if scored else None,
                         'blank': bool((s or {}).get('blank')),
                         'leak': item['condition'] == 'negative' and _leaky(item['qid'], item['lang'])})
        out = summarize(recs)
        out['truncated'] = sum(((r.get('output') or {}).get('finish_reason') == 'length') for r in rows)
        out['statuses'] = dict(Counter(r.get('status') for r in rows))
        if variant.options.get('parse', 'paper') == 'robust':
            neg = [r for r in rows if r['item']['condition'] == 'negative']
            sc = [r['score'] for r in neg if r.get('score') is not None]
            unparsed = sum(bool(s.get('unparsed')) for s in sc)
            parsed = sum(s.get('rej_star') is not None for s in sc)
            missing = len(neg) - parsed - unparsed      # failed subject or judge calls
            out['parse'] = {'parsed_paper': parsed, 'parsed_robust': 0, 'unparsed': unparsed, 'missing': missing}
            out.update({k: out['parse'][k] for k in ('parsed_paper', 'parsed_robust', 'unparsed')})
            out['validity_rate'] = parsed / len(neg) if neg else None
            out['headline_reliable'] = _robust.reliable(len(neg), unparsed, missing)
            out['shelflife'].update(headline_reliable=out['headline_reliable'], parse=out['parse'],
                                   validity_rate=out['validity_rate'])
        return out

    targets = (
        PaperTarget('english_questions', 300, 0, 'arXiv v2 Section "Data construction": 600 base questions, half '
                    'English', 'n/a', 'paper_en', 'released_data',
                    compute=lambda exp: len(exp.items(exp.variant('paper_en'))),
                    note='data/en.json rows (one negative-rejection item each)'),
        PaperTarget('chatgpt_en_rej_exact', 24.67, 0.005, 'arXiv v2 Table 3 (Rej, English)', 'gpt-3.5-turbo (Sep 2023 '
                    '= 0613; reachable as openai/gpt-3.5-turbo-0613)', 'paper_en', 'live',
                    note='no predictions released; temperature 0.2 sampling, so a rerun is a sampling check (SE ~2.5 pp)'),
        PaperTarget('chatgpt_en_rej_star', 45.00, 0.005, 'arXiv v2 Table 3 (Rej*, English)', 'gpt-3.5-turbo-0613',
                    'paper_en', 'live', note='judge gpt-3.5-turbo at default temperature: judge sampling adds noise'),
        PaperTarget('qwen7b_en_rej_exact', 31.00, 0.005, 'arXiv v2 Table 3', 'Qwen-7B-Chat (local HF)', 'paper_en',
                    'unverifiable', note='open models ran locally with their own chat wrappers (models/models.py)'),
    )


def judge_cost_bound(models=None, limit=MINIMAL_LIMIT, variant='minimal', reply_tokens=None):
    """Conservative bound on the Rej* judge calls for ``limit`` items (one per negative item, per subject model),
    priced with the judge's policy on the longest judge prompt the subject cap allows. shelflife.cli plan covers
    only the subject calls; add this. Returns (bound per subject model, n judge calls)."""
    from .. import registry
    exp = registry.get('C093')
    v = exp.variant(variant)
    prog = exp.program(v)
    jm = v.options['judge_model']
    pol = policy_for(jm)
    neg = [x for x in exp.items(v, limit) if x['condition'] == 'negative']
    answer = 'x' * (reply_tokens or 1024)
    total = 0.0
    for x in neg:
        payload, _ = pol.adapt(jm, {'id': x['id'], 'messages': prog.judge_messages(x, answer), 'params': dict(v.options['judge_params'])})
        total += pol.estimate(jm, payload)
    return total, len(neg)
