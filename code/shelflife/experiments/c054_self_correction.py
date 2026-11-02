"""C054 — Large Language Models Cannot Self-Correct Reasoning Yet (Huang et al., ICLR 2024)

Claim-set key: C054 (earlier pipeline). Claim (as adjudicated): asking a model to review and revise its own answer without
external feedback (intrinsic self-correction) does not raise reasoning accuracy and often lowers it, because correct
answers are changed to incorrect ones at least as often as the reverse.

Paper: arXiv 2310.01798v2 (ICLR 2024). Sec 3.1 (setup), Table 3 (GPT-3.5-Turbo gpt-3.5-turbo-0613 / GPT-4, intrinsic
self-correction: CommonSenseQA 75.8 -> 38.1 -> 41.8, GSM8K 75.9 -> 75.1 -> 74.7), Table 4 (GPT-4-Turbo / Llama-2,
"Assume ..." feedback prompt, temperature 0), Table 5 (GPT-4-Turbo with three feedback prompts), Figure 1 (answer
change breakdown), Appendix A (prompts).

Sources
  * No official code or outputs were released. Prompts are transcribed from the paper's LaTeX
    (data/sources/C054_huang2023_arxiv_v2/src/examples/*.tex, Appendix A: bold text = user turns; main.tex Sec 3.1 and
    Table 5 for the alternative feedback prompts). tests/claims/test_c054.py re-parses those files and checks the
    rendered turns byte for byte (the paper's example questions are rendered from our data and compared).
  * Data (data/sources/C054_datasets): CommonsenseQA dev (1,221; the paper's set) and GSM8K test (1,319).

Protocol (Program ``SelfCorrect``): a multi-turn chat. Step 'initial': user = task prompt; then per round r:
  'feedback{r}': user = feedback prompt (history includes every earlier turn); 'revise{r}': user = revise prompt. Two
  rounds = 5 calls (Table 3 '# calls' 1/3/5). Only visible content is carried in the history (provider reasoning traces
  are not re-sent).

Prompts (Appendix A, verbatim)
  CSQA initial  "Q: <stem>\\nAnswer Choices: (A) <a> (B) <b> (C) <c> (D) <d> (E) <e>.\\nExplain your reasoning. You must
                choose only one option from A to E. Your final answer should be a single letter from A to E, in the
                form (answer), at the end of your response."
  GSM8K initial "Can you solve the following math problem? <question> Explain your reasoning. Your final answer should
                be a single numerical number, in the form \\boxed{answer}, at the end of your response."
  feedback      'review' (GPT-3.5 / GPT-4, Table 3): "Review your previous answer and find problems with your answer."
  revise        CSQA "Based on the problems you found, improve your answer. You must choose only one option from A to
                E. Please reiterate your answer, with your final answer a single letter from A to E, in the form
                (answer)."; GSM8K "Based on the problems you found, improve your answer. Please reiterate your answer,
                with your final answer a single numerical number, in the form \\boxed{answer}."

Variants
  gpt35_review      default, faithful. Table 3 GPT-3.5 protocol: full CSQA dev + GSM8K test, 2 rounds, 'review'
                    feedback, temperature 1.
  minimal           RUN_VARIANT: gpt35_review restricted to CSQA and one round (minimal design: initial vs round-1
                    accuracy on the same CSQA questions); option change only, so it stays faithful.
  Not registered: the Table 4-5 GPT-4-Turbo / Llama-2 protocols ('assume', 'determine', 'verify' feedback),
  HotpotQA (Reflexion prompts, closed book), oracle-label self-correction (Table 2), multi-agent debate.

Items: CSQA dev and GSM8K test, each shuffled with random.Random(SAMPLE_SEED). Datasets are interleaved (csqa, gsm8k,
  csqa, ...) so a prefix is balanced. MINIMAL_LIMIT = 70 (selection n_units) CSQA questions for ``minimal``.

Scoring (reconstruction: the paper's evaluator is unreleased; its prompts ask for "(answer)" / "\\boxed{answer}")
  Post-audit headline parser (extract_csqa): anchored on the final-answer statement. Priority: the last 'final answer'
  statement followed within 40 characters by (X), **X**, \\boxed{X} or a bare X (any case inside markers, upper case
  when bare); else the last other answer statement ('answer is/remains/:', 'correct/best answer', 'my answer',
  'I choose/select/pick/stand by'); else the last \\boxed{X}; else a last line holding only a letter; else the last
  "(X)". Echoes of the option range ('from A to E', 'A-E') are blanked first. GSM8K: the last \\boxed{} number
  (nested braces, $ , % and \\text{} stripped, \\frac supported); without any \\boxed{}, the first number after the
  last answer statement. The pre-audit strict parsers (last "(X)"; last \\boxed{}) are kept as diagnostics
  (``strict_pred``; strict-parser drop in the shelflife ``secondary``).
  Unparsed counts as incorrect (paper accuracy over all questions); its count is reported. A chain with a failed call
  is missing for the stages it did not reach. Each answer turn carries ``truncated`` (finish_reason == 'length').
  Per answer turn: unparsed, truncated, strict-parser-unparsed and parser-disagreement rates. ``headline_reliable``
  is False when the unparsed or the truncated rate differs between the initial and the revise1 turn by more than
  5 points (RELIABILITY_GAP); the paired drop excluding chains with a truncated initial or revise1 turn is reported
  as a sensitivity.
Metrics per dataset: accuracy at each stage over chains that reached it; paired accuracy initial vs round 1 (and
round 2) over chains complete through that round, with an item bootstrap CI for the drop; Figure 1 answer-change
categories (no change / correct->incorrect / incorrect->correct / incorrect->incorrect) between initial and the final
round; unparsed counts. shelflife: CSQA accuracy drop after round 1 (initial minus revised, paired), direction +1, null
0, paper_value 0.377 (GPT-3.5: 75.8 - 38.1) for 'review' variants;
secondary anchor for 'review': GPT-4 under the same protocol dropped 0.025 (82.0 - 79.5, Table 3).

Deviations (all variants)
  * max_tokens not reported: 1024 (the providers' reasoning-model policies raise the cap for reasoning models).
  * The paper does not say whether the CSQA trailing period belongs to the template; it is placed after the last
    option, as printed. The 'A:' before the model's answer in the CSQA example (not bold) is model output.
  * Round 2 repeats the same feedback and revise prompts (not shown in the paper).
  * Scorer reconstructed (above).
  * GPT-4 (2023-08-29), gpt-4-1106-preview and Llama-2-70b-chat are not reachable; gpt-3.5-turbo-0613 is.

Robustness variants (non-faithful)
  * ``robust_parse``: ``minimal`` requests unchanged, options parse='robust'. The headline parser (extract_csqa /
    extract_gsm8k) runs first; only when it returns nothing: CSQA takes the letter after an echoed "(answer)"
    placeholder ("(answer) E", the commonest modern failure: M190 58% of initial turns, M171 14%, M017 20%), else the
    first A-E label right after the last explicit answer marker (_robust.answer_segment), else the last **bold** span
    equal to one option's text, else the only option text quoted (M190 often ends with an empty "(answer)" after
    naming the option in bold or quotes); GSM8K takes
    _robust.extract_number. A turn with no answer, or truncated (finish_reason length) without an answer, is
    ``unparsed``; a chain with an unparsed initial or revise1 turn is missing from the headline pair (not counted as a
    wrong answer). headline_reliable = _robust.reliable(chains, chains with an unparsed headline turn, chains that
    did not reach revise1); parsed_paper / parsed_robust / unparsed count answer turns.
    Where robust differs from paper on rows the paper parser DID score: none per turn (the fallback runs only where
    extract_csqa / extract_gsm8k returned nothing; a truncated turn the paper parser read keeps its answer). In the
    aggregate, a chain whose other headline turn is unparsed leaves the pair, so that turn's paper-scored answer is
    not used (paper: the unparsed turn counted wrong and the chain stayed in the pair).
  * ``robust``: robust_parse plus min_cap_reasoning 16384 (M143 / M144 revise1 turns hit the 4096 reasoning cap with
    empty content in 11% of chains); max_tokens 1024 already meets the _robust.cap_floor. Option truncated='score'
    (_robust.truncation_missing): at 1024 / 16384 tokens an answer turn cut off before any answer (finish_reason
    length, even with empty content) failed the task, so it is scored wrong as the paper rule scores it (pred None),
    NOT unparsed, and its chain stays in the headline pair; counted as ``truncated_scored``. A complete turn with no
    answer stays unparsed. Documented exception for robust: such cut-off turns are wrong (paper), not missing.
    robust_parse keeps truncated='missing'.
"""
import functools
import json
import random
import re

from ..paths import PROJECT
from ..program import CallFailed, Program
from ..prompts import PromptTemplate, Source
from ..registry import Experiment, PaperTarget, Variant, register
from .. import stats
from . import _robust

PAPER_SRC = PROJECT / 'code/shelflife/data/sources/C054_huang2023_arxiv_v2'
DATA = PROJECT / 'code/shelflife/data/sources/C054_datasets'
SRC = Source(repo=str(PAPER_SRC), path='src/examples', locator='Appendix A prompt listings (bold = user turns)',
             url='https://arxiv.org/abs/2310.01798v2', note='no official code released')
SAMPLE_SEED = 20231003
MINIMAL_LIMIT = 70                  # selection n_units: CSQA questions (one 3-call chain each)
RUN_VARIANT = 'minimal'
REPEATS = 1
REPRO_MODELS = ['openai-direct/gpt-3.5-turbo-0125']   # the OpenRouter 'gpt-3.5-turbo-0613' route is a GPT-4o-family proxy (reproduction pass)
REPRO_LIMIT = 1221                  # the full CSQA dev set, the paper's GPT-3.5 denominator
claim_set_key = 'C054'


def _src(path, locator):
    return Source(repo=str(PAPER_SRC), path=path, locator=locator, url='https://arxiv.org/abs/2310.01798v2')


CSQA_INIT = PromptTemplate(
    'csqa_initial', 'Q: {question}\nAnswer Choices: {choices}.\nExplain your reasoning. You must choose only one option '
    'from A to E. Your final answer should be a single letter from A to E, in the form (answer), at the end of your '
    'response.', style='replace', tokens={'{question}': 'question', '{choices}': 'choices'},
    source=_src('src/examples/csqa-example-failure.tex', 'Figure (fig:csqa-example-failure), first three bold lines'))
GSM_INIT = PromptTemplate(
    'gsm8k_initial', 'Can you solve the following math problem? {question} Explain your reasoning. Your final answer '
    'should be a single numerical number, in the form \\boxed{answer}, at the end of your response.',
    style='replace', tokens={'{question}': 'question'},
    source=_src('src/examples/gsm-example-failure.tex', 'Figure (fig:gsm-example-failure), first bold line'))
FEEDBACK = {
    'review': PromptTemplate('feedback_review', 'Review your previous answer and find problems with your answer.',
                             style='literal', source=_src('src/examples/csqa-example-failure.tex', 'second bold line '
                                                          '(identical in the GSM8K examples)')),
}
CSQA_REVISE = PromptTemplate(
    'csqa_revise', 'Based on the problems you found, improve your answer. You must choose only one option from A to E. '
    'Please reiterate your answer, with your final answer a single letter from A to E, in the form (answer).',
    style='literal', source=_src('src/examples/csqa-example-failure.tex', 'third bold line'))
GSM_REVISE = PromptTemplate(
    'gsm8k_revise', 'Based on the problems you found, improve your answer. Please reiterate your answer, with your final '
    'answer a single numerical number, in the form \\boxed{answer}.', style='literal',
    source=_src('src/examples/gsm-example-failure.tex', 'third bold line'))

PAPER_DROP_GPT4 = {'review': 0.820 - 0.795}   # Table 3 GPT-4 CSQA, same 'review' / T=1 protocol
RELIABILITY_GAP = 0.05
ROBUST_MIN_CAP_REASONING = 16384    # robust variant: reasoning models emptied the 4096 policy cap on revise turns
PAPER_DROP = {'review': 0.758 - 0.381}
DEVIATIONS = (
    'max_tokens not reported: 1024 (reasoning models get the policy cap; routes that reject temperature sample at the '
    'provider default)',
    'CSQA: the trailing period after option (E) is placed in the template, as printed in Appendix A',
    'CSQA: the "A:" before the model answer in the Appendix A figure is treated as model output, not as prompt text',
    'round 2 repeats the round-1 feedback and revise prompts (not shown in the paper)',
    'answer extraction not released: CSQA letter from the last final-answer statement ((X), **X**, \\boxed{X}, bare X; '
    'fallbacks \\boxed{X}, lone last-line letter, last (X)); GSM8K last \\boxed{} number, else the number after the '
    'last answer statement; unparsed counts as incorrect; the pre-audit strict parsers are kept as diagnostics',
    'reasoning traces of reasoning models are not re-sent in the conversation history (visible content only)',
)


# ---------------------------------------------------------------- data
@functools.lru_cache(maxsize=1)
def _csqa():
    out = []
    for line in (DATA / 'csqa_dev_rand_split.jsonl').read_text().splitlines():
        r = json.loads(line)
        q = r['question']
        out.append({'id': f'C054:csqa:{r["id"]}', 'dataset': 'csqa', 'question': q['stem'],
                    'choices': ' '.join(f'({c["label"]}) {c["text"]}' for c in q['choices']),
                    'labels': [c['label'] for c in q['choices']], 'answer': r['answerKey']})
    return tuple(out)


@functools.lru_cache(maxsize=1)
def _gsm8k():
    out = []
    for i, line in enumerate((DATA / 'gsm8k_test.jsonl').read_text().splitlines()):
        r = json.loads(line)
        out.append({'id': f'C054:gsm8k:{i:04d}', 'dataset': 'gsm8k', 'question': r['question'],
                    'answer': r['answer'].split('####')[-1].strip().replace(',', '')})
    return tuple(out)


def shuffled(dataset):
    rows = [dict(r) for r in (_csqa() if dataset == 'csqa' else _gsm8k())]
    random.Random(f'{SAMPLE_SEED}:{dataset}').shuffle(rows)
    return rows


# ---------------------------------------------------------------- scoring
CSQA_ANS = re.compile(r'\(\s*\**\s*([A-E])\s*\**\s*\)')
FINAL_ANCHOR = re.compile(r'final\s+answer', re.I)
ANSWER_ANCHOR = re.compile(r'(?:correct|best)\s+answer|answer\s+(?:is|remains|would\s+be|should\s+be|stays)|'
                           r'answer\s*:|my\s+answer|\bi\s+(?:choose|select|pick|stand\s+by)', re.I)
LETTER = re.compile(
    r'\\boxed\{\s*\(?\s*([A-Ea-e])\s*\)?\s*\}'                    # \boxed{E}, \boxed{(E)}
    r'|\(\s*\**\s*([A-Ea-e])\s*\**\s*\)'                            # (E), (**E**), (e)
    r'|\*\*\s*([A-Ea-e])\s*[.):]?\s*\*\*'                            # **E**, **E.**
    r'|(?<![A-Za-z\'])([A-E])(?![A-Za-z\'])(?=\s*(?:$|[.,;:!)\n*]|\s\(|\s[-\u2013\u2014]))'   # bare E (upper case)
    r'|(?<![A-Za-z\'])([a-e])(?![A-Za-z\'])(?=\s*(?:$|[.)\n]))')     # bare e at the end of a statement
LONE_LAST_LINE = re.compile(r'^\W*\(?\s*([A-Ea-e])\s*\)?\W*$')
ANCHOR_WINDOW = 40
RANGE_ECHO = re.compile(r'\b(?:from\s+)?\(?A\)?\s*(?:to|-|\u2013|through)\s*\(?E\)?(?![A-Za-z])')


def extract_csqa_strict(text):
    """Pre-audit parser (kept as a diagnostic): the last "(X)", X in A-E."""
    hits = CSQA_ANS.findall(text or '')
    return hits[-1] if hits else None


def _letter_after(text, end):
    m = LETTER.search(text[end:end + ANCHOR_WINDOW])
    return next(g for g in m.groups() if g).upper() if m else None


def _anchored(text, anchor):
    found = None
    for m in anchor.finditer(text):
        letter = _letter_after(text, m.end())
        if letter:
            found = letter
    return found


def extract_csqa(text):
    """Robust parser (revised headline), by priority:
    1. the last 'final answer' statement followed (within 40 characters) by (X), **X**, \\boxed{X} or a bare X;
    2. else the last other answer statement ('answer is/remains/:', 'correct answer', 'my answer', 'I choose ...');
    3. else the last \\boxed{X}; 4. else a last line that is only a letter; 5. else the last "(X)" (strict rule).
    Letters in (X), **X**, \\boxed{} may be lower case; returned upper case. Echoes of the instruction's option range
    ('from A to E', 'A-E') are blanked first so they are never read as an answer."""
    text = RANGE_ECHO.sub(' ', text or '')
    for anchor in (FINAL_ANCHOR, ANSWER_ANCHOR):
        letter = _anchored(text, anchor)
        if letter:
            return letter
    boxed = re.findall(r'\\boxed\{\s*\(?\s*([A-Ea-e])\s*\)?\s*\}', text)
    if boxed:
        return boxed[-1].upper()
    lines = [l for l in text.splitlines() if l.strip()]
    if lines:
        m = LONE_LAST_LINE.match(lines[-1].strip())
        if m:
            return m.group(1).upper()
    hits = re.findall(r'\(\s*\**\s*([A-Ea-e])\s*\**\s*\)', text)
    return hits[-1].upper() if hits else None


def _boxed_contents(text):
    out, i = [], 0
    while True:
        j = text.find('\\boxed{', i)
        if j < 0:
            return out
        k, depth = j + len('\\boxed{'), 1
        start = k
        while k < len(text) and depth:
            depth += {'{': 1, '}': -1}.get(text[k], 0)
            k += 1
        if depth:
            return out
        out.append(text[start:k - 1])
        i = k


def _number(s):
    s = re.sub(r'\\text\{[^}]*\}|\\[,! ]|\\?\$|\\?%|,', '', s).strip()
    m = re.fullmatch(r'[-+]?\d+(?:\.\d+)?', s)
    if not m:
        frac = re.fullmatch(r'\\[dt]?frac\{(-?\d+)\}\{(\d+)\}', s)
        return int(frac.group(1)) / int(frac.group(2)) if frac and int(frac.group(2)) else None
    return float(s)


def extract_gsm8k_strict(text):
    """Pre-audit parser (diagnostic): the last \\boxed{} must hold a number."""
    boxes = _boxed_contents(text or '')
    return _number(boxes[-1]) if boxes else None


GSM_NUMBER = re.compile(r'-?\$?\d[\d,]*(?:\.\d+)?')


def extract_gsm8k(text):
    """Headline: the last \\boxed{} number; without any \\boxed{}, the first number within 40 characters after the
    last 'final answer' (else 'answer is/:') statement."""
    text = text or ''
    if _boxed_contents(text):
        return extract_gsm8k_strict(text)
    for anchor in (FINAL_ANCHOR, ANSWER_ANCHOR):
        found = None
        for m in anchor.finditer(text):
            n = GSM_NUMBER.search(text[m.end():m.end() + ANCHOR_WINDOW])
            if n:
                found = _number(n.group())
        if found is not None:
            return found
    return None


ECHOED_PLACEHOLDER = re.compile(r'\(\s*answer\s*\)\s*[:=\-]?\s*\**\s*\(?\s*\**([A-E])\**\s*\)?(?![A-Za-z])', re.I)
ROBUST_LABEL = re.compile(r'(?<![A-Za-z0-9])\(?([A-E])\)?(?![A-Za-z0-9])')


def _option_texts(choices):
    """{normalized option text: label} from the rendered '(A) x (B) y ...' choices string."""
    parts = re.split(r'\(([A-E])\)\s*', choices or '')
    return {' '.join(parts[i + 1].lower().split()): parts[i] for i in range(1, len(parts) - 1, 2)}


def robust_csqa(text, choices=None):
    """Fallback (parse='robust', only when extract_csqa returns None): the letter after an echoed '(answer)'
    placeholder ('(answer) E'; last occurrence); else the first A-E label within 80 characters after the last explicit
    answer marker of the _robust-normalized text; else, when ``choices`` is given, the last **bold** span that equals
    one option's text exactly ('... most likely **cast shadow**' followed by an empty '(answer)'); else the option
    whose text is the only one quoted ('only “classroom” fits'). No unique-letter-mention guess (bare 'A' is
    also an article)."""
    text = text or ''
    hits = ECHOED_PLACEHOLDER.findall(text)
    if hits:
        return hits[-1].upper()
    t = _robust.normalize(RANGE_ECHO.sub(' ', text))
    seg = _robust.answer_segment(RANGE_ECHO.sub(' ', text))
    if seg != t:
        m = ROBUST_LABEL.search(seg[:80])
        if m:
            return m.group(1)
    if choices:
        opts = _option_texts(choices)
        for span in reversed(re.findall(r'\*\*([^*\n]{1,80})\*\*', text)):
            key = ' '.join(span.strip().strip('.').lower().split())
            if key in opts:
                return opts[key]
        quoted = {opts[k] for k in (' '.join(q.strip().strip('.').lower().split())
                                    for q in re.findall(r'[\u201c"]([^\u201d"\n]{1,80})[\u201d"]', text)) if k in opts}
        if len(quoted) == 1:
            return quoted.pop()
    return None


def judge(item, text, truncated=False, parse='paper', cut_missing=True):
    if item['dataset'] == 'csqa':
        pred, strict = extract_csqa(text), extract_csqa_strict(text)
        path = 'paper' if pred is not None else None
        if parse == 'robust' and pred is None:
            pred, path = _robust.fallback(None, robust_csqa, text, item.get('choices'))
        correct = pred == item['answer']
    else:
        pred, strict = extract_gsm8k(text), extract_gsm8k_strict(text)
        path = 'paper' if pred is not None else None
        if parse == 'robust' and pred is None:
            pred, path = _robust.fallback(None, _robust.extract_number, text)
        correct = pred is not None and abs(pred - float(item['answer'])) < 1e-6
    out = {'pred': pred, 'correct': correct, 'strict_pred': strict,
           'strict_correct': strict is not None and (strict == item['answer'] if item['dataset'] == 'csqa'
                                                     else abs(strict - float(item['answer'])) < 1e-6),
           'truncated': bool(truncated)}
    if parse == 'robust':
        if path is None and truncated and not cut_missing:
            # truncated='score' (robust): a turn cut off before any answer at a generous cap failed the task; the
            # paper rule scores it wrong (pred None), so it is scored, not unparsed
            path = 'paper'
            out['truncated_scored'] = True
        out.update(parse_path=path, unparsed=path is None)
    return out


# ---------------------------------------------------------------- program
class SelfCorrect(Program):
    name = 'intrinsic_self_correction'

    def __init__(self, variant):
        self.params = dict(variant.params)
        if variant.options.get('min_cap_reasoning'):
            self.case_extra = {'min_cap_reasoning': variant.options['min_cap_reasoning']}
        self.rounds = variant.options['rounds']
        self.feedback = variant.prompts['feedback']
        self.csqa_init, self.gsm_init = variant.prompts['csqa_initial'], variant.prompts['gsm8k_initial']
        self.csqa_revise, self.gsm_revise = variant.prompts['csqa_revise'], variant.prompts['gsm8k_revise']

    def initial_messages(self, item):
        if item['dataset'] == 'csqa':
            text = self.csqa_init.render(question=item['question'], choices=item['choices'])
        else:
            text = self.gsm_init.render(question=item['question'])
        return [{'role': 'user', 'content': text}]

    def preview(self, item):
        return self.initial_messages(item)

    def revise_text(self, item):
        return (self.csqa_revise if item['dataset'] == 'csqa' else self.gsm_revise).render()

    def forward(self, item, lm):
        out = {'complete': False, 'failed_step': None, 'failed_status': None, 'turns': {}, 'finish': {}}
        history = self.initial_messages(item)
        steps = [('initial', None)]
        for r in range(1, self.rounds + 1):
            steps += [(f'feedback{r}', self.feedback.render()), (f'revise{r}', self.revise_text(item))]
        try:
            for step, user in steps:
                if user is not None:
                    history.append({'role': 'user', 'content': user})
                c = lm(history, self.params, step=step)
                out['turns'][step] = c.content
                out['finish'][step] = c.finish_reason
                history.append({'role': 'assistant', 'content': c.content})
        except CallFailed as e:
            out.update(failed_step=e.completion.step, failed_status=e.completion.status)
            return out
        out['complete'] = True
        return out


def answer_turns(rounds):
    return ['initial'] + [f'revise{r}' for r in range(1, rounds + 1)]


# ---------------------------------------------------------------- aggregation
def _paired(chains, a, b, boot=True):
    both = [c for c in chains if a in c['stages'] and b in c['stages']]
    if not both:
        return None
    drop = lambda cs: (sum(c['stages'][a]['correct'] for c in cs) - sum(c['stages'][b]['correct'] for c in cs)) / len(cs)
    if boot:
        d, lo, hi = stats.cluster_bootstrap(both, lambda c: c['id'], drop, n_boot=2000, seed=0)
    else:
        d, lo, hi = drop(both), None, None
    return {'n': len(both), f'acc_{a}': sum(c['stages'][a]['correct'] for c in both) / len(both),
            f'acc_{b}': sum(c['stages'][b]['correct'] for c in both) / len(both),
            'drop': d, 'drop_ci95': [lo, hi]}


def _changes(chains, a, b):
    both = [c for c in chains if a in c['stages'] and b in c['stages']]
    out = {'n': len(both), 'no_change': 0, 'correct_to_incorrect': 0, 'incorrect_to_correct': 0,
           'incorrect_to_incorrect': 0}
    for c in both:
        x, y = c['stages'][a], c['stages'][b]
        if x['pred'] == y['pred'] and x['pred'] is not None:
            out['no_change'] += 1
        elif x['correct'] and not y['correct']:
            out['correct_to_incorrect'] += 1
        elif not x['correct'] and y['correct']:
            out['incorrect_to_correct'] += 1
        elif x['correct'] and y['correct']:
            out['no_change'] += 1          # both correct with differently formatted predictions (e.g. 18 vs 18.0)
        else:
            out['incorrect_to_incorrect'] += 1
    return out


# ---------------------------------------------------------------------------- earlier pipeline runs
# experiments/run_minimal_setups/runs/earlier/runs/<claim>-<variant>-L<N>[-<tag>]/<model>/: one shelflife.runner directory per model.
# Registered: every directory that sent at least one request (complete or partial); tests/test_prior_runs.py
# replays each one against this config. Excluded: aborted, smoke-test and superseded directories, and directories
# that sent nothing. Listed explicitly (written 2026-10-05 from the run directories; nothing is globbed at import).

_R2 = 'experiments/run_minimal_setups/runs/earlier/runs'
_NOTHING_SENT = ('no request reached a provider: every call is not_submitted (circuit breaker open, or stale ledger '
                 'rows after a package-hash change); the directory only records the attempts')


def _unsent(*runs):
    """excluded_runs entries for per-model directories that sent nothing: (run, space-separated models)."""
    return {f'{_R2}/{run}/{m}': _NOTHING_SENT for run, models in runs for m in models.split()}


# reproductions (2026-10-08): the OpenRouter route openai/gpt-3.5-turbo-0613 (Azure, fp_92f9364fa2) serves a
# GPT-4o-family model (o200k token counts, 2024 knowledge cutoff), so the round2 'repro' runs above are PROXY runs.
# The reproduction pass re-ran the minimal design on the genuine openai-direct/gpt-3.5-turbo-0125 (cl100k counts verified).
_SWEEP = 'experiments/initial-reproduction/runs/runs/C054'


@register
class C054SelfCorrection(Experiment):
    claim_id = 'C054'
    claim_set_key = 'C054'
    maturity = 'ready'
    title = 'Large Language Models Cannot Self-Correct Reasoning Yet'
    claim = ('Asking a model to review and revise its own answer without external feedback (intrinsic self-correction) '
             'does not raise reasoning accuracy and often lowers it, because correct answers are changed to incorrect '
             'ones at least as often as the reverse.')
    paper = {'title': 'Large Language Models Cannot Self-Correct Reasoning Yet',
             'authors': ['Jie Huang', 'Xinyun Chen', 'Swaroop Mishra', 'Huaixiu Steven Zheng', 'Adams Wei Yu',
                         'Xinying Song', 'Denny Zhou'],
             'first_date': '2023-10-03', 'venue': 'ICLR 2024', 'url': 'https://arxiv.org/abs/2310.01798',
             'version': 'v2 (targets and prompts)'}
    source = SRC
    default_variant = 'gpt35_review'

    def build_variants(self):
        def make(name, desc, feedback, temperature, subset, deviations=()):
            return Variant(name, desc, prompts={'csqa_initial': CSQA_INIT, 'gsm8k_initial': GSM_INIT,
                                                'feedback': FEEDBACK[feedback], 'csqa_revise': CSQA_REVISE,
                                                'gsm8k_revise': GSM_REVISE},
                           params={'temperature': temperature, 'max_tokens': 1024}, strategy='intrinsic_self_correction',
                           options={'datasets': ['csqa', 'gsm8k'], 'rounds': 2, 'subset': subset,
                                    'feedback': feedback, 'sample_seed': SAMPLE_SEED},
                           repeats=REPEATS, deviations=DEVIATIONS + tuple(deviations))
        v = {
            'gpt35_review': make('gpt35_review', 'Table 3 GPT-3.5-Turbo protocol: full CSQA dev + GSM8K test, 2 rounds, '
                                 '"review" feedback, temperature 1', 'review', 1.0, 'full'),
        }
        v['minimal'] = v['gpt35_review'].override(
            'minimal', description='Minimal design (RUN_VARIANT): gpt35_review on CommonSenseQA only, one round '
            '(initial -> review -> revise); Table 3 CSQA column, round 1', options={'datasets': ['csqa'], 'rounds': 1})
        # Robustness variants: robust_parse rescores saved `minimal` outputs (identical requests);
        # robust is the variant for new panel runs.
        v['robust_parse'] = v['minimal'].override(
            'robust_parse', description='minimal requests, parse=robust (headline parser first, "(answer) X" / '
            'answer-marker fallback; unparsed chains missing; tolerant reliability)', options={'parse': 'robust'},
            deviations=('scoring: headline parser first; when it returns nothing, the letter after an echoed '
                        '"(answer)" placeholder or after the last answer marker (GSM8K: _robust.extract_number); '
                        'chains with an unparsed or truncated-unanswered headline turn are dropped from the pair; '
                        'reliability = <=10% unparsed or missing chains',))
        v['robust'] = v['robust_parse'].override(
            'robust', description='robust_parse + min_cap_reasoning 16384 so reasoning models do not exhaust the cap',
            params=_robust.cap_floor(v['minimal'].params),
            options={'min_cap_reasoning': ROBUST_MIN_CAP_REASONING, 'truncated': 'score'},
            deviations=('reasoning-model output cap 4096 -> 16384 (min_cap_reasoning); max_tokens 1024 unchanged '
                        '(already at the _robust floor)',))
        return v

    def items(self, variant, limit=None):
        per = {d: shuffled(d) for d in variant.options['datasets']}
        rows = []
        for k in range(max(len(v) for v in per.values())):
            for d in variant.options['datasets']:
                if k < len(per[d]):
                    rows.append(per[d][k])
        return rows[:limit] if limit else rows

    def program(self, variant):
        return SelfCorrect(variant)

    def score(self, item, output, variant):
        stages = {}
        for step in answer_turns(variant.options['rounds']):
            if step in (output.get('turns') or {}):
                stages[step] = judge(item, output['turns'][step],
                                     (output.get('finish') or {}).get(step) == 'length',
                                     parse=variant.options.get('parse', 'paper'),
                                     cut_missing=_robust.truncation_missing(variant))
        return {'stages': stages, 'complete': bool(output.get('complete'))}

    def aggregate(self, rows, variant):
        rounds = variant.options['rounds']
        turns = answer_turns(rounds)
        out = {'n_rows': len(rows), 'rows_without_output': sum(r.get('output') is None for r in rows),
               'incomplete_chains': sum(1 for r in rows if r.get('score') is not None and not r['score']['complete']),
               'by_dataset': {}}
        caveats = []
        for d in variant.options['datasets']:
            chains = [{'id': r['item']['id'], 'stages': r['score']['stages']} for r in rows
                      if r['item']['dataset'] == d and r.get('score') is not None]
            n_rows = sum(r['item']['dataset'] == d for r in rows)
            res = {'n': n_rows, 'stage_accuracy': {}, 'unparsed': {}, 'per_turn': {}}
            for t in turns:
                reached = [c['stages'][t] for c in chains if t in c['stages']]
                n = len(reached)
                res['stage_accuracy'][t] = {'n': n, 'accuracy': (sum(s['correct'] for s in reached) / n) if n else None,
                                            'accuracy_strict_parser': (sum(s['strict_correct'] for s in reached) / n)
                                            if n else None}
                res['unparsed'][t] = sum(s['pred'] is None for s in reached)
                res['per_turn'][t] = {
                    'n': n, 'unparsed_rate': res['unparsed'][t] / n if n else None,
                    'truncated_rate': sum(s['truncated'] for s in reached) / n if n else None,
                    'strict_parser_unparsed_rate': sum(s['strict_pred'] is None for s in reached) / n if n else None,
                    'parser_disagreement_rate': sum(s['strict_pred'] != s['pred'] for s in reached) / n if n else None}
            for t in turns[1:]:
                res[f'paired_initial_vs_{t}'] = _paired(chains, 'initial', t)
                strict_chains = [{'id': c['id'], 'stages': {k: {**v, 'correct': v['strict_correct']}
                                                            for k, v in c['stages'].items()}} for c in chains]
                p = _paired(strict_chains, 'initial', t, boot=False)
                res[f'paired_initial_vs_{t}_strict_parser_drop'] = p['drop'] if p else None
                clean = [c for c in chains if t in c['stages']
                         and not (c['stages']['initial']['truncated'] or c['stages'][t]['truncated'])]
                p = _paired(clean, 'initial', t, boot=False)
                res[f'paired_initial_vs_{t}_excluding_truncated'] = p
            res['answer_changes_initial_to_final'] = _changes(chains, 'initial', turns[-1])
            a, b = res['per_turn']['initial'], res['per_turn'].get('revise1') or {}
            for key in ('unparsed_rate', 'truncated_rate'):
                if a.get(key) is not None and b.get(key) is not None and abs(b[key] - a[key]) > RELIABILITY_GAP:
                    caveats.append(f'{d}: {key} differs between initial ({a[key]:.1%}) and revise1 ({b[key]:.1%}) '
                                   f'by more than {RELIABILITY_GAP:.0%}')
            out['by_dataset'][d] = res
        robust = variant.options.get('parse', 'paper') == 'robust'
        if robust:
            # unparsed answer turns are missing, not wrong: the headline pair uses chains whose turns both parsed
            for d in variant.options['datasets']:
                res = out['by_dataset'][d]
                chains = [{'id': r['item']['id'], 'stages': r['score']['stages']} for r in rows
                          if r['item']['dataset'] == d and r.get('score') is not None]
                turns_all = [s for c in chains for s in c['stages'].values()]
                res['parse'] = {'parsed_paper': sum(s['parse_path'] == 'paper' for s in turns_all),
                                'parsed_robust': sum(s['parse_path'] == 'robust' for s in turns_all),
                                'unparsed': sum(s['parse_path'] is None for s in turns_all),
                                'truncated_scored': sum(bool(s.get('truncated_scored')) for s in turns_all)}
                head = [c for c in chains if 'initial' in c['stages'] and 'revise1' in c['stages']]
                parsed = [c for c in head if not (c['stages']['initial']['unparsed']
                                                  or c['stages']['revise1']['unparsed'])]
                res['paired_initial_vs_revise1_paper_denominator'] = res.get('paired_initial_vs_revise1')
                res['paired_initial_vs_revise1'] = _paired(parsed, 'initial', 'revise1')
                res['robust_reliability'] = {'n': res['n'], 'unparsed_chains': len(head) - len(parsed),
                                             'missing_chains': res['n'] - len(head)}
            caveats = []
            for d, res in out['by_dataset'].items():
                rr = res['robust_reliability']
                if not _robust.reliable(rr['n'], rr['unparsed_chains'], rr['missing_chains']):
                    caveats.append(f"{d}: {rr['unparsed_chains']} chains with an unparsed headline turn + "
                                   f"{rr['missing_chains']} missing of {rr['n']} > {_robust.ROBUST_TOLERANCE:.0%}")
            for k in ('parsed_paper', 'parsed_robust', 'unparsed', 'truncated_scored'):
                out[k] = sum(res['parse'][k] for res in out['by_dataset'].values())
            tot = out['parsed_paper'] + out['parsed_robust'] + out['unparsed']
            out['validity_rate'] = (out['parsed_paper'] + out['parsed_robust']) / tot if tot else None
        csqa_res = out['by_dataset'].get('csqa', {})
        csqa = csqa_res.get('paired_initial_vs_revise1')
        out['headline_reliable'] = (not caveats) if csqa else None
        out['headline_caveats'] = caveats
        fb = variant.options['feedback']
        out['shelflife'] = {
            'metric': 'csqa_accuracy_drop_round1', 'value': csqa['drop'] if csqa else None,
            'ci': csqa['drop_ci95'] if csqa else None, 'n': csqa['n'] if csqa else 0,
            'direction': +1, 'null_value': 0.0, 'paper_value': PAPER_DROP[fb],
            'definition': 'CommonSenseQA accuracy of the initial answer minus accuracy after one intrinsic '
                          'self-correction round (review then revise), paired over questions whose 3-call chain '
                          'completed; answers parsed from the final-answer statement; CI by question bootstrap.',
            'secondary': {
                'drop_strict_parser': csqa_res.get('paired_initial_vs_revise1_strict_parser_drop'),
                'drop_excluding_truncated': (csqa_res.get('paired_initial_vs_revise1_excluding_truncated') or {}).get('drop'),
                'paper_value_gpt4_same_protocol': PAPER_DROP_GPT4.get(fb),
            },
        }
        return out

    # ------------------------------------------------------------ targets
    _T3 = 'arXiv 2310.01798v2 Table 3 (table:pure-self-correction)'
    _LIVE = ('gpt-3.5-turbo-0613 is the paper\'s model and is reachable (openai/gpt-3.5-turbo-0613 via OpenRouter -> '
             'Azure); temperature 1 sampling: tolerance ±0.03 (~2 SE at n=1,221)')
    targets = (
        PaperTarget('csqa_dev_size', 1221, 0, 'arXiv v2 Sec 3.1 ("dev set ... 1,221 questions")', 'n/a',
                    'gpt35_review', 'released_data', compute=lambda exp: len(_csqa())),
        PaperTarget('gsm8k_test_size', 1319, 0, 'arXiv v2 Sec 3.1 ("test set of 1,319")', 'n/a', 'gpt35_review',
                    'released_data', compute=lambda exp: len(_gsm8k())),
        PaperTarget('calls_two_rounds', 5, 0, _T3 + ' "# calls" column', 'n/a', 'gpt35_review', 'released_data',
                    compute=lambda exp: _count_calls(exp, 'gpt35_review'),
                    note='our program issues 1 + 2 x rounds calls, as the table states'),
        PaperTarget('calls_one_round', 3, 0, _T3 + ' "# calls" column', 'n/a', 'minimal', 'released_data',
                    compute=lambda exp: _count_calls(exp, 'minimal')),
        PaperTarget('csqa_initial_gpt35', 0.758, 0.03, _T3 + ', GPT-3.5 CSQA Standard Prompting', 'gpt-3.5-turbo-0613',
                    'minimal', 'live', note=_LIVE),
        PaperTarget('csqa_round1_gpt35', 0.381, 0.03, _T3 + ', GPT-3.5 CSQA Self-Correct (round 1)',
                    'gpt-3.5-turbo-0613', 'minimal', 'live', note=_LIVE),
        PaperTarget('csqa_round2_gpt35', 0.418, 0.03, _T3 + ', GPT-3.5 CSQA Self-Correct (round 2)',
                    'gpt-3.5-turbo-0613', 'gpt35_review', 'live', note=_LIVE),
        PaperTarget('gsm8k_initial_gpt35', 0.759, 0.03, _T3 + ', GPT-3.5 GSM8K Standard Prompting',
                    'gpt-3.5-turbo-0613', 'gpt35_review', 'live', note=_LIVE),
        PaperTarget('gsm8k_round1_gpt35', 0.751, 0.03, _T3 + ', GPT-3.5 GSM8K round 1', 'gpt-3.5-turbo-0613',
                    'gpt35_review', 'live', note=_LIVE),
    )


def _count_calls(exp, variant_name):
    """Run the program against a stub LM and count calls (offline)."""
    from ..program import Completion
    v = exp.variant(variant_name)
    calls = []

    class Stub:
        def __call__(self, messages, params=None, step='main'):
            calls.append(step)
            return Completion(step=step, status='ok', content='The answer is (A). \\boxed{1}')
    exp.program(v).forward(exp.items(v, limit=1)[0], Stub())
    return len(calls)
