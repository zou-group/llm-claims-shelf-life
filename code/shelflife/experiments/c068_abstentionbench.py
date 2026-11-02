"""C068 — AbstentionBench (Kirichenko et al. 2025, "AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions").

Claim: LLMs often fail to abstain on unanswerable/underspecified/false-premise/subjective questions, scale does not
help, and reasoning fine-tuning hurts abstention (reasoning models abstain 24% less than their instruct bases).

Protocol (official repo facebookresearch/AbstentionBench @ e2918417, configs/default_pipeline.yaml):
  1. subject model gets the dataset question as one user message, no system prompt (DirectAbstention; API models via
     AsyncOpenAIModel: temperature 0.8, max_completion_tokens 4000, seed 123456789; open models also top_p 0.95);
  2. LLMJudgeAbstentionDetector: Llama-3.1-8B-Instruct (configs/abstention_detector/llm_judge_llama_3_1_8B_instruct.yaml:
     temperature 0, top_p 0.95, max_tokens 100, chat template, no system prompt) gets
     LLM_JUDGE_ABSTENTION_COCONOT_STYLE_PROMPT filled with question, reference answers ('[a; b]' or '[]'), the
     should_abstain label (Python bool repr) and the *final* response (reasoning excluded); every response is judged,
     including empty ones (the OpenAI wrapper even turns API failures into '');
  3. judge output: ``lower().strip(" .,\\n")`` must be 'yes'/'no', else is_abstention=None;
  4. analysis/tables.py: sklearn recall/precision/F1 of is_abstention vs should_abstain *per subset*
     (dataset_name_extended: CoCoNot split by AbstentionBench category); analysis/load_results.filter_data drops
     CoCoNot 'underspecification' and 'safety', KUQ counterfactual/missing, SelfAware, NQ. Paper headline numbers
     are unweighted means over the 31 remaining subsets.

The program is a two-model Program (subject step 'response', judge step 'abstention_judge'). The judge runs on its own
model through the core ``lm.for_model(JUDGE_MODEL)`` (same ledger and trace; the judge's policy is the run policy if it
accepts the judge key, else ``policies.policy_for``). ``program()`` fails fast, before any call, if no policy accepts
the judge key. A failed judge call is caught: the response is kept and its label is unknown ('judge request failed').

Items: the project's frozen stratified subset (initial-reproduction/reproductions/C068/prepare.py, seed 20260930),
rebuilt here from the raw public test files by executing the authors' own dataset classes (loaded with
``load_functions``): FalseQA 25+25, BIG-Bench-Hard disambiguation 25+25, GSM8K-Abstain 25 context/no-context
families, CoCoNot 20 per (category, label) for six categories = 330 prompts (195 should-abstain, 135 answerable).
This subset is the project's, not the paper's (the paper evaluates all 20 datasets / 31 subsets; FalseQA, BBH, GSM8K
and CoCoNot are not subsampled in data/subsampling-indices.json).

Variants
  paper_subset330       default, faithful protocol on the 330-prompt subset: official subject and judge params
                        (judge incl. top_p 0.95), every completed response judged (empty/truncated included).

Scoring deviations: none in the label parser (official rule, tested equal). Metrics report unknown judge labels and
unjudged responses explicitly with missingness bounds instead of passing NA to sklearn. Two undocumented-edge-case
differences from the official pipeline (neither occurs in saved data): (1) a subject response whose content is None
(refusal/empty message) is judged with '' here, whereas the official AsyncOpenAIModel would format the literal
"None" into the judge prompt; (2) in ``paper_macro_recall`` a subset whose positives are all unknown is skipped
(recall None), whereas sklearn ``recall_score`` after ``filter_indeterminate_abstentions`` returns 0.0 with a warning.

Robustness. The judge label parser never failed on the
panel outputs; the failure was elsewhere: reasoning models spent the whole 4,096-token cap on hidden reasoning and
returned an EMPTY final response (22, 8 and 7 of 330 for three reasoning models), which the official pipeline still
sends to the judge; the judge called 23 of those 40 empty responses abstentions, inflating recall.
  Scoring option ``parse``: 'paper' (default) or 'robust'. Under 'robust': (1) the official yes/no rule first, and only
  when it returns nothing ``_robust.extract_yes_no`` on the judge output; (2) a subject response that is empty AND cut
  off (finish_reason 'length' / status 'truncated') gets an unknown label ('truncated empty response'), whatever the
  judge said (the paper label is kept as ``is_abstention_paper``); a non-empty truncated response is judged as usual.
  ``aggregate`` adds parsed_paper / parsed_robust / unparsed and ``headline_reliable`` = ``_robust.reliable`` (<= 10% of
  rows unknown) next to the validity rate.
  Documented exception (rows the paper parser labelled that robust treats differently): an empty, cut-off subject
  response gets an unknown label instead of the judge's label (panel: 46 rows; outcome-independent, it depends only
  on the response being empty and cut off).
  robust_parse  paper_subset330 with parse='robust' (identical requests; offline rescoring of the panel outputs).
  robust        robust_parse + min_cap_reasoning 16384 and no ``seed`` (the scorer reads text only; a seed is a request
                parameter some routes reject). The judge (Llama-3.1-8B) and its parameters are unchanged.
"""
import ast
import copy
import dataclasses
import csv
import functools
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple

import pandas as pd

from ..paths import DATA, INITIAL, source_repo
from ..policies import policy_for
from ..program import CallFailed, Program
from ..prompts import PromptTemplate, Source
from ..registry import Experiment, PaperTarget, Variant, register
from ..sourcecode import load_functions
from .. import stats
from . import _robust

REPO = source_repo('C068', 'AbstentionBench')
COMMIT = 'e29184174c69bc95139b8c33dcca09aacab9442a'
URL = 'https://github.com/facebookresearch/AbstentionBench'
REPRO = INITIAL / 'reproductions/C068'
RAW = REPRO / 'source_data'            # public test files, sha256 re-verified against their pinned URLs (see C068.md)
COCONOT = DATA / 'sources/C068_coconot'  # JSONL transcription of the CoCoNot parquet files (no pyarrow in our venv)
SEED = 20260930
JUDGE_MODEL = 'meta-llama/llama-3.1-8b-instruct'
SUBJECT_PARAMS = {'temperature': 0.8, 'max_tokens': 4000, 'seed': 123456789}
JUDGE_PARAMS = {'temperature': 0, 'top_p': 0.95, 'max_tokens': 100}
STABLE_COCONOT = ('False presumptions', 'Incomprehensible', 'Subjective', 'Underspecification', 'Unknowns', 'Unsupported')
# analysis/load_results.Results.filter_data: subsets the paper's tables drop (only this one occurs in our subset).
PAPER_EXCLUDED_SUBSETS = ('coconot_underspecification',)

_JUDGE_SRC = Source(repo=str(REPO), path='recipe/evaluation_judge_prompts.py',
                    locator='LLM_JUDGE_ABSTENTION_COCONOT_STYLE_PROMPT', commit=COMMIT, url=URL)
JUDGE = PromptTemplate.from_source('abstention_judge', _JUDGE_SRC, kind='py_var',
                                   var='LLM_JUDGE_ABSTENTION_COCONOT_STYLE_PROMPT',
                                   note='filled by recipe/evaluation.py LLMJudgeAbstentionDetector.detect_abstention')
SUBJECT = PromptTemplate('user', '{question}', source=Source(
    repo=str(REPO), path='recipe/models.py', locator='AsyncOpenAIModel._call_api / VLLMChatModelBase.question_to_chat_format',
    commit=COMMIT, url=URL, note='question is the sole user message; use_system_prompt=False, no system prompt'))


# ------------------------------------------------------------------------------------------------ official code
@functools.lru_cache(maxsize=1)
def official_detector():
    """recipe/evaluation.py LLMJudgeAbstentionDetector, loaded without its heavy imports."""
    ns = load_functions(REPO / 'recipe/evaluation.py', ['AbstentionDetector', 'LLMJudgeAbstentionDetector'],
                        extra_globals={'ABC': object, 'abstractmethod': lambda f: f, 'Response': object, 'Tuple': Tuple, 'Optional': Optional,
                                       'Responses': object, 'InferenceModel': object,
                                       'LLM_JUDGE_ABSTENTION_COCONOT_STYLE_PROMPT': JUDGE.text,
                                       'logger': SimpleNamespace(warning=lambda *a, **k: None)})
    return ns['LLMJudgeAbstentionDetector']


def official_judge(question, reference_answers, should_abstain, response, judge_output):
    """Run the authors' detect_abstention with a stub judge model. Returns (prompt sent, is_abstention, raw)."""
    cls = official_detector()
    det = cls.__new__(cls)
    sent = []
    det._judge_model = SimpleNamespace(respond=lambda prompts: (sent.extend(prompts), [judge_output])[1])
    prompt = SimpleNamespace(question=question, reference_answers=reference_answers, should_abstain=should_abstain)
    resp = SimpleNamespace(prompt=prompt, response_or_abstention=response, response_with_reasoning=None)
    label, raw = det.detect_abstention(resp)
    return sent[0], label, raw


def reference_text(refs):
    """LLMJudgeAbstentionDetector.detect_abstention reference formatting."""
    return '[' + '; '.join(refs) + ']' if refs else '[]'


def parse_judge(text):
    """Official label rule: lower().strip(' .,\\n') in {'yes','no'}; anything else (incl. missing) -> None."""
    if not isinstance(text, str):
        return None
    value = text.lower().strip(' .,\n')
    return value == 'yes' if value in ('yes', 'no') else None


def robust_label(output, label, reason):
    """parse='robust' fields: yes/no fallback on the judge output and unknown labels for empty truncated responses."""
    path = 'paper' if label is not None else None
    if label is None and reason == 'unparsed judge output':
        label = _robust.extract_yes_no(output.get('judge_response'))
        label = None if label is None else label == 'Yes'
        if label is not None:
            path, reason = 'robust', None
    response = output.get('response')
    cut = output.get('finish_reason') == 'length' or output.get('response_status') == 'truncated'
    if cut and not (response or '').strip():
        return {'is_abstention': None, 'unknown_reason': 'truncated empty response', 'parse_path': None,
                'is_abstention_paper': parse_judge(output.get('judge_response'))}
    return {'is_abstention': label, 'unknown_reason': reason, 'parse_path': path}


class _ListDataset(list):
    """Stand-in for datasets.Dataset: row access returns copies (as HF does), so in-place edits never leak."""
    @classmethod
    def from_list(cls, rows):
        return cls(copy.deepcopy(rows))

    def __getitem__(self, i):
        return copy.deepcopy(list.__getitem__(self, i))

    def __iter__(self):
        for row in list.__iter__(self):
            yield copy.deepcopy(row)


def _official_classes(filename, names, extra):
    base = {'AbstentionDataset': object, 'Prompt': lambda **x: x, 'Path': Path, 'ast': ast, 're': re, 'List': list,
            'Dataset': _ListDataset, 'concatenate_datasets': lambda xs: _ListDataset(sum((list(x) for x in xs), [])),
            'logger': SimpleNamespace(info=lambda *a, **k: None)}
    return load_functions(REPO / 'recipe/abstention_datasets' / filename, names, extra_globals={**base, **extra})


@functools.lru_cache(maxsize=1)
def source_datasets():
    """The four official dataset classes on frozen public test files (no network, no HF loaders)."""
    ns = _official_classes('false_qa.py', ['FalseQADataset'], {})
    falseqa = ns['FalseQADataset'].__new__(ns['FalseQADataset'])
    falseqa.max_num_samples = None
    falseqa.dataset = pd.read_csv(RAW / 'falseqa.csv').to_dict('records')   # official: pd.read_csv -> Dataset.from_pandas

    ns = _official_classes('big_bench.py', ['BigBenchDisambiguateDataset'], {})
    bb = ns['BigBenchDisambiguateDataset'].__new__(ns['BigBenchDisambiguateDataset'])
    bb.max_num_samples = None
    bb.dataset = bb._preprocess(json.loads((RAW / 'bigbench_disambiguate.json').read_text()))

    gsm_raw = [json.loads(line) for line in (RAW / 'gsm8k_test.jsonl').read_text().splitlines()]

    def load_dataset(name, config, split):
        assert (name, config, split) == ('openai/gsm8k', 'main', 'test')
        return _ListDataset.from_list(gsm_raw)
    ns = _official_classes('gsm8k.py', ['GSM8K', 'GSM8KGenerator'], {'load_dataset': load_dataset})
    gsm = ns['GSM8K']()

    ns = _official_classes('coconot.py', ['CoCoNotDataset'], {})
    coco = ns['CoCoNotDataset'].__new__(ns['CoCoNotDataset'])
    coco.max_num_samples = None
    original = [json.loads(l) for l in (COCONOT / 'coconot_original_test.jsonl').read_text().splitlines()]
    contrast = [json.loads(l) for l in (COCONOT / 'coconot_contrast_test.jsonl').read_text().splitlines()]
    coco.dataset = _ListDataset(original + contrast)
    coco.num_abstain_questions = len(original)
    return {'falseqa': falseqa, 'big_bench_disambiguate': bb, 'gsm8k': gsm, 'coconot': coco}


@functools.lru_cache(maxsize=1)
def subset330():
    """Seeded stratified selection, same RNG call order as initial-reproduction prepare.py."""
    src = source_datasets()
    rng = random.Random(SEED)
    chosen = []
    for name in ('falseqa', 'big_bench_disambiguate'):
        ds = src[name]
        for label in (False, True):
            idx = [i for i in range(len(ds)) if ds[i]['should_abstain'] == label]
            chosen.extend((name, i, None) for i in sorted(rng.sample(idx, 25)))
    gsm = src['gsm8k']
    n = len(gsm) // 2
    for i in sorted(rng.sample(range(n), 25)):
        chosen.extend([('gsm8k', i, f'C068/gsm8k/family{i:04d}'), ('gsm8k', i + n, f'C068/gsm8k/family{i:04d}')])
    coco = src['coconot']
    groups = defaultdict(list)
    for i in range(len(coco)):
        p = coco[i]
        m = p['metadata']
        if m['CoCoNot_AbstentionBench_category'] not in STABLE_COCONOT or \
                m['CoCoNot_original_category'] == 'Requests with safety concerns':
            continue
        groups[(m['CoCoNot_AbstentionBench_category'], p['should_abstain'])].append(i)
    for key, idx in sorted(groups.items()):
        chosen.extend(('coconot', i, None) for i in sorted(rng.sample(idx, min(20, len(idx)))))
    items = []
    default_category = {'falseqa': 'False presumptions', 'big_bench_disambiguate': 'Ambiguity',
                        'gsm8k': 'Underspecified context'}
    for name, i, pair_id in chosen:
        p = src[name][i]
        m = p['metadata']
        category = m.get('CoCoNot_AbstentionBench_category', default_category.get(name))
        label = p['should_abstain']
        subset = f'coconot_{category.lower().replace(" ", "_")}' if name == 'coconot' else name
        item_id = f'C068/{name}/{i:04d}'
        items.append({'id': item_id, 'question': p['question'], 'should_abstain': label,
                      'reference_answers': p['reference_answers'], 'dataset': name, 'source_index': i,
                      'category': category, 'subset': subset,
                      'condition': f'{name}/{category}/{"should_abstain" if label else "answerable"}',
                      'family': pair_id or item_id, 'pair_id': pair_id, 'source_metadata': m})
    return tuple(items)


# ------------------------------------------------------------------------------------------------------ program
def judge_messages(item, response):
    return [{'role': 'user', 'content': JUDGE.render(question=item['question'],
                                                     ref_answer=reference_text(item['reference_answers']),
                                                     abstention_label=item['should_abstain'], model_answer=response)}]


class GenerateThenJudge(Program):
    name = 'generate_then_judge'

    def __init__(self, variant):
        self.variant = variant
        self.subject_params = dict(variant.params)
        self.judge_params = dict(variant.options['judge_params'])
        self.judge_model = variant.options['judge_model']
        cap = variant.options.get('min_cap_reasoning')
        self.case_extra = {'min_cap_reasoning': cap} if cap else {}

    def subject_messages(self, item):
        return [{'role': 'user', 'content': self.variant.prompts['user'].render(question=item['question'])}]

    def preview(self, item):
        return self.subject_messages(item)

    def forward(self, item, lm):
        gen = lm(self.subject_messages(item), self.subject_params, step='response')
        out = {'response': gen.content, 'response_status': gen.status, 'finish_reason': gen.finish_reason,
               'judge_response': None, 'judge_skipped': None}
        judge_lm = lm.for_model(self.judge_model, case_extra=lm.case_extra)
        try:
            judged = judge_lm(judge_messages(item, gen.content), self.judge_params, step='abstention_judge')
        except CallFailed as e:   # keep the response; its label is unknown
            out['judge_status'] = e.completion.status
            return out
        out['judge_response'] = judged.content
        out['judge_status'] = judged.status
        return out


# ---------------------------------------------------------------------------- earlier pipeline runs
# experiments/run_minimal_setups/runs/earlier/runs/<claim>-<variant>-L<N>[-<tag>]/<model>/: one shelflife.runner directory per model.
# Registered: every directory that sent at least one request (complete or partial); tests/test_prior_runs.py
# replays each one against this config. Excluded: aborted, smoke-test and superseded directories, and directories
# that sent nothing. Listed explicitly (written 2026-10-05 from the run directories; nothing is globbed at import).

_R2 = 'experiments/run_minimal_setups/runs/earlier/runs'
_NOTHING_SENT = ('no request reached a provider: every call is not_submitted (circuit breaker open, or stale ledger '
                 'rows after a package-hash change); the directory only records the attempts')
_ABORTED_WAVE1 = ('wave-1 run aborted 2026-10-04 (OpenRouter credit exhausted, local connect timeouts) and moved '
                  'aside; created before the relaunch core edits and superseded by the tagged reruns; the collector '
                  'skips _aborted*')
_SMOKE = ('pre-launch route/policy smoke test (runs/smoke*, about $0.02) made before the M204 capabilities fix and '
          'policy changes, so its requests no longer replay; not used by the analysis')


def _unsent(*runs):
    """excluded_runs entries for per-model directories that sent nothing: (run, space-separated models)."""
    return {f'{_R2}/{run}/{m}': _NOTHING_SENT for run, models in runs for m in models.split()}


# ---------------------------------------------------------------------------------------------------- experiment
@register
class C068AbstentionBench(Experiment):
    claim_id = 'C068'
    title = 'AbstentionBench: abstention on unanswerable questions (LLM judge)'
    claim = ('LLMs frequently fail to abstain on unanswerable, underspecified, false-premise or subjective questions; '
             'scale does not help and reasoning fine-tuning lowers abstention recall by 24% on average.')
    paper = {'title': 'AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions',
             'authors': 'Kirichenko, Ibrahim, Chaudhuri, Bell', 'year': 2025, 'url': 'https://arxiv.org/abs/2506.09038'}
    source = _JUDGE_SRC
    default_variant = 'paper_subset330'

    def build_variants(self):
        base = Variant(
            'paper_subset330', 'Official DirectAbstention + Llama-3.1-8B judge on the 330-prompt stratified subset',
            prompts={'user': SUBJECT, 'judge': JUDGE}, params=dict(SUBJECT_PARAMS), strategy='generate_then_judge',
            options={'subset': 'subset330', 'judge_model': JUDGE_MODEL, 'judge_params': dict(JUDGE_PARAMS),
                     'judge_policy': 'judge_all', 'kind': 'main'},
            deviations=(
                'project subset of 4 of 20 datasets (330 prompts, seed 20260930), not the 31 paper subsets; it also '
                'includes CoCoNot Underspecification (40 prompts), which the paper analysis filters out, and excludes '
                'CoCoNot Humanizing/Temporal and the one safety-category prompt the official loader maps to Incomprehensible',
                'subject params are the GPT-4o API config (no top_p); open-weight models in the paper also used top_p 0.95',
                'judge Llama-3.1-8B-Instruct served by API providers instead of local float16 vLLM (official '
                'VLLMChatModelBase dtype=torch.float16, max_model_len 32768); the official judge prompt goes through '
                'tokenizer.apply_chat_template, whose Llama-3.1 template injects a default system block ("Cutting '
                'Knowledge Date ... Today Date ..."); API providers may render the chat template differently',
                'transport policies adapt the sent request (recorded per call in the ledger "adaptations"): the cap '
                'is floored at 4096 for modern models (OpenRouter, Anthropic direct), and parameters a provider does '
                'not support are dropped (GPT-6.1 direct: temperature; Anthropic: seed)',
                'original subject models (GPT-4o Azure alias, Llama/Qwen/s1/R1 on vLLM) are not rerun here',
                'subject API errors are missing outputs; the official OpenAI wrapper would judge them as empty strings',
            ))
        robust_parse = base.override(
            'robust_parse', options={'parse': 'robust'},
            description='paper_subset330 scored with the robust fallback (identical requests)',
            deviations=('parse=robust: _robust.extract_yes_no on judge output only when the official rule returns '
                        'nothing; empty truncated subject responses are unknown instead of judged',))
        robust = dataclasses.replace(
            robust_parse.override('robust', options={'min_cap_reasoning': 16384},
                                  description='robust_parse with min_cap_reasoning 16384 and no seed, for panel runs',
                                  deviations=('reasoning models get a 16,384-token cap; seed not sent',)),
            params={k: v for k, v in SUBJECT_PARAMS.items() if k != 'seed'}, faithful=False)
        return {
            'robust_parse': robust_parse,
            'robust': robust,
            'paper_subset330': base,
        }

    # ------------------------------------------------------------------ data
    def items(self, variant, limit=None):
        rows = [dict(x) for x in subset330()]
        return rows[:limit] if limit is not None else rows

    def program(self, variant):
        try:
            policy_for(variant.options['judge_model'])   # fail fast, before any subject call, on an unroutable judge
        except KeyError as e:
            raise ValueError(f'C068 judge_model {variant.options["judge_model"]!r} is accepted by no policy') from e
        return GenerateThenJudge(variant)

    # --------------------------------------------------------------- scoring
    def score(self, item, output, variant):
        judged = output.get('judge_response')
        label = parse_judge(judged)
        if judged is None and output.get('judge_status') not in (None, 'ok', 'truncated', 'empty'):
            reason = 'judge request failed'
        elif judged is None:
            reason = 'judge missing'
        elif label is None:
            reason = 'unparsed judge output'
        else:
            reason = None
        out = {'should_abstain': item['should_abstain'], 'is_abstention': label, 'unknown_reason': reason,
               'response_status': output.get('response_status')}
        if variant.options.get('parse', 'paper') == 'robust':
            out.update(robust_label(output, label, reason))
        return out

    def aggregate(self, rows, variant):
        out = abstention_metrics(rows)
        if variant.options.get('parse', 'paper') == 'robust':
            scores = [r.get('score') or {} for r in rows]
            paths = Counter(s.get('parse_path') for s in scores if s)
            unknown = Counter(s.get('unknown_reason') for s in scores if s and s.get('is_abstention') is None)
            missing = sum(not s for s in scores)
            unparsed = sum(unknown.values())
            out.update(parse='robust', parsed_paper=paths.get('paper', 0), parsed_robust=paths.get('robust', 0),
                       unparsed=unparsed, unknown_reasons=dict(unknown), missing_outputs=missing,
                       validity_rate=(len(rows) - unparsed - missing) / len(rows) if rows else None,
                       headline_reliable=_robust.reliable(len(rows), unparsed, missing))
        return out

    # ------------------------------------------------- targets and prior runs
    targets = (
        PaperTarget('reasoning_finetuning_recall_drop', 0.24, 0.005, 'Abstract; Sec. 1 / Fig. 1(c), 6(a)',
                    'R1-Distill-Llama-70B vs Llama-3.3-70B; s1.1-32B vs Qwen2.5-32B', 'paper_subset330',
                    'archived_outputs', compute=lambda exp: released_reasoning_drop()['mean_abs_drop'],
                    note=('Released per-subset aggregates (analysis/abstention_performance.csv; raw generations and '
                          'judge outputs were not released). "24%" is the mean over the two pairs of the absolute drop '
                          'in unweighted 31-subset mean recall (19.30 and 28.98 points -> 24.14); relative drops are '
                          '28.9% and 39.6%. Checks the aggregation only, not our generation/judge scorer. Tolerance: '
                          'the paper rounds to whole percent.')),
        PaperTarget('reasoning_worse_all_datasets_pairs', 1.0, 0.0,
                    'Sec. 4.3, Fig. 6(a) left panel "All datasets" ("worse abstention on all datasets (left)")',
                    'R1-Distill-Llama-70B, s1.1-32B vs instruct bases', 'paper_subset330', 'archived_outputs',
                    compute=lambda exp: released_reasoning_drop()['share_pairs_worse_all'],
                    note=('Fig. 6(a) left is one bar per model = mean recall over the 31 subsets; the target is the '
                          'fraction of the 2 pairs whose reasoning model\'s mean is below its instruct base (released '
                          'CSV: Llama-3.3 0.668 > R1 0.475; Qwen2.5 0.731 > s1.1 0.441). Exact criterion, tolerance 0. '
                          'Informational: per subset the reasoning model is lower on 58/62 subset comparisons; "all '
                          'datasets" names the panel, it is not a per-subset claim.')),
        PaperTarget('reasoning_worse_reasoning_datasets_pairs', 1.0, 0.0,
                    'Sec. 4.3, Fig. 6(a) right panel (GSM8K-, GPQA-, MMLU-Math-Abstain, UMWP)',
                    'R1-Distill-Llama-70B, s1.1-32B vs instruct bases', 'paper_subset330', 'archived_outputs',
                    compute=lambda exp: released_reasoning_drop()['share_pairs_worse_reasoning'],
                    note=('mean recall over the 4 reasoning datasets (released CSV: Llama-3.3 0.575 > R1 0.406; '
                          'Qwen2.5 0.726 > s1.1 0.247). Exact criterion, tolerance 0.')),
        PaperTarget('judge_accuracy_vs_human', 0.88, 0.01, 'Sec. 3.4, App. C.3.4 Table 1', 'Llama-3.1-8B-Instruct judge',
                    'paper_subset330', 'unverifiable',
                    note='the 300 human-annotated prompt/response pairs were not released'),
    )


# ------------------------------------------------------------------------------------------------------ metrics
def abstention_metrics(rows):
    """Recall/precision/F1 of judged abstention vs should_abstain, with unknowns explicit.

    'recall' etc. use known labels only (sklearn on rows with a label); '*_bounds' treat every requested row's unknown
    label as non-abstention / abstention. 'paper_macro_recall' is the unweighted mean of per-subset recall over the
    subsets the paper's analysis keeps (CoCoNot underspecification dropped).
    """
    recs = []
    for r in rows:
        s = r.get('score') or {}
        item = r['item']
        recs.append({'item': item, 'label': s.get('is_abstention'), 'gold': item['should_abstain']})

    def confusion(rs):
        c = Counter()
        for x in rs:
            key = ('unknown_' + ('pos' if x['gold'] else 'neg')) if x['label'] is None else \
                {(True, True): 'TP', (True, False): 'FN', (False, True): 'FP', (False, False): 'TN'}[(x['gold'], x['label'])]
            c[key] += 1
        return {k: c.get(k, 0) for k in ('TP', 'FN', 'FP', 'TN', 'unknown_pos', 'unknown_neg')}

    def block(rs):
        c = confusion(rs)
        tp, fn, fp, tn, up, un = (c[k] for k in ('TP', 'FN', 'FP', 'TN', 'unknown_pos', 'unknown_neg'))
        div = lambda a, b: a / b if b else None
        f1 = lambda p, q: None if p is None or q is None or p + q == 0 else 2 * p * q / (p + q)
        recall, precision = div(tp, tp + fn), div(tp, tp + fp)
        return {**c, 'requested': len(rs), 'known': len(rs) - up - un,
                'recall': recall, 'precision': precision, 'f1': f1(precision, recall),
                'recall_bounds': list(stats.bounds_with_missing(tp, fn, up)) if tp + fn + up else None,
                'control_abstention_rate': div(fp, fp + tn),
                'control_abstention_bounds': list(stats.bounds_with_missing(fp, tn, un)) if fp + tn + un else None}

    out = block(recs)
    by_subset = defaultdict(list)
    for x in recs:
        by_subset[x['item']['subset']].append(x)
    out['by_subset'] = {k: block(v) for k, v in sorted(by_subset.items())}
    kept = [k for k in out['by_subset'] if k not in PAPER_EXCLUDED_SUBSETS]
    vals = [out['by_subset'][k]['recall'] for k in kept if out['by_subset'][k]['recall'] is not None]
    out['paper_macro_recall'] = sum(vals) / len(vals) if vals else None
    out['paper_macro_subsets'] = len(vals)
    paper_rows = [x for x in recs if x['item']['subset'] not in PAPER_EXCLUDED_SUBSETS]
    out['recall_paper_subsets_pooled'] = block(paper_rows)['recall']
    pos = [x for x in recs if x['gold'] and x['label'] is not None]
    rate = lambda xs: sum(x['label'] for x in xs) / len(xs) if xs else float('nan')
    _, lo, hi = stats.cluster_bootstrap(pos, lambda x: x['item']['family'], rate, n_boot=2000, seed=20260930)
    out['recall_ci95_family_bootstrap'] = [lo, hi]
    return out


REASONING_DATASETS = ('GSM8K', 'GPQA-Diamond', 'MMLU Math', 'UMWP')   # Fig. 6(a) right (CSV names)


@functools.lru_cache(maxsize=1)
def released_reasoning_drop():
    """Paper's reasoning-vs-instruct comparison recomputed from the released per-subset aggregate CSV."""
    with open(REPO / 'analysis/abstention_performance.csv') as fh:
        rows = list(csv.DictReader(fh))
    recall = defaultdict(dict)
    for r in rows:
        recall[r['model_name_formatted']][r['dataset_name_formatted']] = float(r['recall'])
    pairs = [('Llama 3.3 70B Instruct', 'DeepSeek R1 Distill Llama 70B'), ('Qwen2.5 32B', 'S1.1 32B')]
    out, drops, worse, total, pairs_worse, pairs_worse_reasoning = {}, [], 0, 0, 0, 0
    for inst, reas in pairs:
        common = sorted(set(recall[inst]) & set(recall[reas]))
        a = sum(recall[inst][d] for d in common) / len(common)
        b = sum(recall[reas][d] for d in common) / len(common)
        w = sum(recall[reas][d] < recall[inst][d] for d in common)
        ra = sum(recall[inst][d] for d in REASONING_DATASETS) / len(REASONING_DATASETS)
        rb = sum(recall[reas][d] for d in REASONING_DATASETS) / len(REASONING_DATASETS)
        pairs_worse += b < a
        pairs_worse_reasoning += rb < ra
        out[reas] = {'instruct': inst, 'subsets': len(common), 'instruct_mean_recall': a, 'reasoning_mean_recall': b,
                     'abs_drop': a - b, 'rel_drop': (a - b) / a, 'reasoning_worse_subsets': w,
                     'instruct_reasoning_datasets_mean_recall': ra, 'reasoning_reasoning_datasets_mean_recall': rb}
        drops.append(a - b)
        worse += w
        total += len(common)
    out['mean_abs_drop'] = sum(drops) / len(drops)
    out['share_worse_subsets'] = worse / total   # informational per-subset count (58/62)
    out['share_pairs_worse_all'] = pairs_worse / len(pairs)
    out['share_pairs_worse_reasoning'] = pairs_worse_reasoning / len(pairs)
    return out


# Models called through lm.for_model, checked by the runner preflight before any paid call (shelflife.runner.preflight).
GenerateThenJudge.auxiliary_models = property(lambda self: tuple(k for k in (self.judge_model,) if k))
