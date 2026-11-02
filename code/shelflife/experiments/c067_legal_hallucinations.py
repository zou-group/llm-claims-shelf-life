"""C067 — Large Legal Fictions: legal hallucinations (Dahl, Magesh, Suzgun, Ho; arXiv 2401.01301, Journal of Legal
Analysis 16(1) 2024).

Claim: asked verifiable questions about real U.S. federal cases, LLMs give wrong answers (hallucinate) at high rates,
more for lower courts and less prominent cases (GPT-4 58% to Llama 2 88% pooled over the reference-based tasks).

Minimal design: zero-shot reference-based queries in six
task x court cells: Court (USCOA, USDC), Citation (SCOTUS, USCOA, USDC), Author (SCOTUS); 30 cases per cell at
MINIMAL_LIMIT. Court-for-SCOTUS is excluded (the U.S. Reports citation gives the answer away: 0.000 for every model).
Existence, few-shot prompts, the higher-complexity tasks, the contra-factual tasks and calibration are not ported.

Sources (all offline):
  * Harvard Dataverse replication package doi:10.7910/DVN/V4ON8H (v1.0, 2024-04-26), code files saved verbatim in
    data/sources/C067_dahl_dataverse_V4ON8H/legal_hallucinations/ (tasks_{scotus,coa,usdc}.py: prompts, Task params;
    api.py: request layout; models.py: greedy-answer protocol; correctness_checks.py: scorer; tables.R: the
    hallucination rule; results/tables/CategoryOneTable.tex: paper Table "Hallucination rates ... (low complexity
    tasks)", identical to the arXiv v2 Tables/CategoryOneTable.tex) plus the SCDB justice-name covariate
    (data/covariates/scdb_justice_name_map.csv, Dataverse file 10146103, original format).
  * HF dataset reglab/legal_hallucinations @ c1d87c069a3b (dataset.csv, the public ~90% of queries with the authors'
    raw model outputs and correctness scores). The zero-shot rows of the six cells for GPT 4 and GPT 3.5 (54,172 rows)
    are saved in data/sources/C067_reglab_legal_hallucinations_hf/. The query texts there are exactly the user
    messages the authors sent (models.Query.query); about 10% of cases (reserve.csv) are embargoed.

Protocol (variant ``zero_shot``, default, faithful; RUN_VARIANT): api.OpenAIChat greedy request — system message =
the task's ``system_message``, user message = the rendered ``query_template`` (which repeats the system message),
temperature 0, top_p 1, frequency/presence penalty 0, n=1, max_tokens = the Task's ``max_tokens`` (30 for SCOTUS
citation, else the Task default 100). The released query text is used as the user message (it is the rendered
template; a claim test checks every query against the template parsed from the task file).

Scoring (faithful, ``score``): the authors' post-processing (clean_circuit / clean_district / clean_simple /
clean_judge_name) and correctness callbacks (coa_court_id_correctness / usdc_court_id_correctness /
citation_correctness [eyecite] / name_correctness [thefuzz]), loaded from correctness_checks.py; hallucination =
correctness <= 72 and != -99 (tables.R get_hr); -99 (declined/abstained, DECLINE_TO_ANSWER_TOKENS) is a
non-hallucination and stays in the denominator. Our port reproduces the released correctness score on 54,171 of
54,172 released GPT-4/GPT-3.5 outputs (the one miss is spaCy NER on a judge name, 23 vs 18, same verdict).

USDC Court rule and the corrected headline : the paper requires "the name of the specific
district court" for USDC cases (arXiv text, Sec. 3 Court task). In the Dataverse code and the released GPT-4 run the
USDC Court truth is the *state* (case.state), and usdc_court_id_correctness compares clean_district(answer) with
clean_district(state) = "District of <State>"; an answer naming a region ("Eastern District of Arkansas") becomes
"Eastern Arkansas" and can never match, so every correct answer in a multi-district state is a hallucination (24% of
GPT-4's released USDC-Court "hallucinations" name the right state). The released Llama 2 and PaLM 2 USDC-Court rows
are scored against the full district name ("United States District Court for the Eastern District of Arkansas"),
i.e. at district level. The released "GPT 3.5" USDC-Court rows are byte-identical to the Llama 2 rows (all 4,498
outputs and scores; the paper table also shows GPT 3.5 = Llama 2 = 0.815 zero-shot and 0.870 few-shot), so the true
GPT-3.5 outputs for this cell were never released and the paper's GPT-3.5 USDC-Court value is not a GPT-3.5
number; it is not used as a target. ``score`` returns ``hallucination`` (Dataverse code rule) and
``hallucination_corrected`` (USDC Court: the authors' function with the district-name truth, taken from the released
district-level rows; other cells unchanged). The ``shelflife`` headline uses the corrected rule; the code rule is
reported as ``pooled_paper``. Inherited limitation: clean_district does not recognise Puerto Rico / Virgin Islands
(not in STATE_REGEX), so their truths stay raw strings and short correct answers can miss (3.1% of USDC cases).

Missing outputs: an empty/whitespace answer or a truncated one (finish_reason 'length' / status
'truncated') is treated as missing — excluded from every cell rate and counted (``missing``), for all tasks alike.
(The authors' callbacks would score an empty answer -99 in Court/Author but 0 = hallucination in Citation.)
``headline_reliable`` is False when failed calls + missing + scorer errors exceed 10% of rows.

Abstention sensitivity: the paper counts an LLM that "admits it does not know the answer" as a
non-hallucination, but the code lexicon (DECLINE_TO_ANSWER_TOKENS) misses modern phrasings ("I don't know", "I'm not
certain", "I cannot verify"). ``headline_abstain_lenient`` re-computes the headline treating LENIENT_ABSTAIN matches
as abstentions (a hedged answer such as "I'm not certain, but ..." is then also an abstention);
``no_answer_share`` = code abstentions + lenient abstentions + answers with no parseable circuit / state / citation /
name; ``abstention_sensitive`` is True when headline and lenient values differ by more than the CI half-width.

Deviations (faithful variant): see ``build_variants``; in short — original snapshots retired or replaced; greedy
request only (no calibration samples); public 90% of cases; six cells only, so the verdict covers the pooled
reference-based hallucination rate only (the claim's false-premise, court-level-gradient, prominence and calibration
limbs are not tested); reasoning models run with the policy's parameter changes; scorer library versions.

Scoring deviations (earlier pipeline validation 2026-10-05; experiments/run_minimal_setups/runs/earlier/validation/batch1.md, C067). Both
change ``hallucination_corrected`` and therefore the headline; the pre-validation corrected rule is kept per row as
``hallucination_corrected_v0`` and reported as ``pooled_corrected_v0``; ``hallucination`` (the code rule,
``pooled_paper``) is unchanged:
  1. Comma citation format. The citation prompt asks for "<volume>, <reporter>, <page>", which eyecite cannot parse,
     so an exactly correct "330, U.S., 518" (truth 330 U.S. 518) scored 0. In citation cells a whole answer of the
     form "<digits>, <reporter>, <digits>" is rewritten to "<digits> <reporter> <digits>" (CITATION_COMMA) before
     clean_simple / citation_correctness.
  2. Abstention with no answer. A row that matches LENIENT_ABSTAIN *and* has no parseable answer of the cell's type
     ("I don't have reliable information about who wrote ...") is an abstention: a non-hallucination kept in the
     denominator, as the paper treats declines. "No parseable answer" is ``unparseable`` (no circuit / state / eyecite
     citation); for Author it is "no SCDB justice surname in the answer" (``no_answer_given``). Hedged answers that still name a circuit / state / citation / name
     ("I'm not certain, but ...") stay scored. ``headline_abstain_lenient`` (any lenient match is an abstention) is
     computed on the new corrected rule, so it stays <= the headline.

Robustness. Scoring option ``parse``: 'paper'
(default) or 'robust'. Under 'robust' the scorer above runs on the raw answer first; only when it leaves a
non-abstained answer with no parseable circuit / state / citation / justice name (``unparseable``) is the answer
re-scored after ``_robust.normalize`` plus unicode-space folding (narrow no-break / thin spaces: gpt-oss-120b wrote
"F.<U+202F>Supp.<U+202F>2d" in 15 the panel citations, which eyecite cannot read). The fallback is kept only if it yields a
parseable answer, so it can turn an unreadable answer into a scored one but never changes an answer the paper scorer
read. Empty/truncated answers stay missing; ``headline_reliable`` already tolerates <= 10% missing (= _robust.reliable)
and is unchanged; parsed_paper / parsed_robust / unparsed are reported.
  Documented exception (rows the paper scorer scored that robust treats differently): an unparseable, non-abstained
  answer (scored a hallucination by the paper callbacks) takes the verdict of the normalized re-score when that is
  parseable (panel: 17 rows, all still hallucinations).
  robust_parse  zero_shot with parse='robust' (identical requests; offline rescoring of the panel outputs).
  robust        robust_parse + per-cell output cap floor 1024 (30/100 tokens cut off verbose answers: Claude 3.5
                Haiku 5, Gemini 3.8 Flash 3) + min_cap_reasoning 16384 (two reasoning models returned 30 and 34
                empty answers at 4,096) + option truncated='score' (_robust.truncation_missing): at 1024 / 16384
                tokens for a one-line answer, a cut-off answer is the model's answer, not a cap artefact, so a
                NON-EMPTY cut-off answer is scored by the authors' callbacks on its text (hallucination or not), not
                missing; counted as ``truncated_scored``. An EMPTY cut-off answer (a reasoning model that spent the
                whole 16,384-token budget on hidden reasoning: GLM-5 M143, 24 of 180 rows) is also scored, with the
                authors' callback mapping for an empty answer: -99 = abstention (non-hallucination) in Court/Author,
                0 = hallucination in Citation;
                counted as ``truncated_empty_scored`` (parse_path 'paper'). An empty answer that was NOT cut off stays
                missing. Documented exception for robust: cut-off answers, empty or not, are scored (the module's
                all-variant rule makes them missing). robust_parse keeps truncated='missing'.
                Label (all variants, follow-up 2026-10-10): a cut-off answer is counted in ``missing_truncated`` even
                when its content is empty; ``missing_empty`` counts only empty answers that were not cut off (before,
                GLM-5's empty length-cut rows were reported as missing_empty).
"""
import ast
import dataclasses
import functools
import random
import re
from collections import Counter, defaultdict

from ..paths import DATA
from ..program import SingleTurn
from ..prompts import PromptTemplate, Source, extract_text
from ..registry import Experiment, PaperTarget, Variant, register
from ..sourcecode import load_functions
from .. import stats
from . import _robust

claim_set_key = 'C067'
DV = DATA / 'sources' / 'C067_dahl_dataverse_V4ON8H' / 'legal_hallucinations'
HF = DATA / 'sources' / 'C067_reglab_legal_hallucinations_hf' / 'dataset_zero_shot_minimal_cells_gpt.csv.gz'
HF_OTHER = DATA / "sources" / "C067_reglab_legal_hallucinations_hf" / "dataset_zero_shot_court_usdc_llama2_palm2.csv.gz"
DV_URL = 'https://doi.org/10.7910/DVN/V4ON8H'
SEED = 47   # settings.RANDOM_SEED of the authors' code, reused for our subsample order

# key, task, court, tasks file, Task variable, post-processor, correctness callback
CELLS = (
    ('court_coa', 'court_id', 'coa', 'tasks_coa.py', 'court_id_task', 'clean_circuit', 'coa_court_id_correctness'),
    ('court_usdc', 'court_id', 'usdc', 'tasks_usdc.py', 'court_id_task', 'clean_district', 'usdc_court_id_correctness'),
    ('citation_scotus', 'citation_retrieval', 'scotus', 'tasks_scotus.py', 'citation_retrieval_task', 'clean_simple',
     'citation_correctness'),
    ('citation_coa', 'citation_retrieval', 'coa', 'tasks_coa.py', 'citation_retrieval_task', 'clean_simple',
     'citation_correctness'),
    ('citation_usdc', 'citation_retrieval', 'usdc', 'tasks_usdc.py', 'citation_retrieval_task', 'clean_simple',
     'citation_correctness'),
    ('author_scotus', 'majority_author', 'scotus', 'tasks_scotus.py', 'majority_author_task', 'clean_judge_name',
     'name_correctness'),
)
CELL = {c[0]: c for c in CELLS}

# Run settings for earlier pipeline (see experiments/run_minimal_setups/runs/earlier/ports/C067.md).
RUN_VARIANT = 'zero_shot'
MINIMAL_LIMIT = 180          # 30 cases per cell x 6 cells (cells interleaved)
MINIMAL_REPEATS = 1
REPRO_MODELS = ['openai/gpt-3.5-turbo-0613', 'openai-direct/gpt-4-turbo-2024-04-09']
REPRO_LIMIT = 600            # 100 cases per cell

# Table "Hallucination rates across levels of the federal judiciary (low complexity tasks)", zero-shot rows
# (results/tables/CategoryOneTable.tex == arXiv v2 Tables/CategoryOneTable.tex).
PAPER_TABLE = {
    'gpt4': {'court_coa': 0.490, 'court_usdc': 0.829, 'citation_scotus': 0.621, 'citation_coa': 0.727,
             'citation_usdc': 0.610, 'author_scotus': 0.799},
    'gpt35': {'court_coa': 0.645, 'court_usdc': 0.815, 'citation_scotus': 0.684, 'citation_coa': 0.754,
              'citation_usdc': 0.702, 'author_scotus': 0.796},
}
# Corrected pooled rate (mean of the six cell rates, USDC Court district-level) of the released GPT-4 outputs;
# recomputed from the archive by the target 'gpt4_pooled_corrected_released'. Since the earlier pipeline validation the
# corrected rule also normalises comma citations and counts lenient no-answer abstentions (SCORING_DEVIATIONS):
# 0.6494; the pre-validation rule (hallucination_corrected_v0) gives 0.6502 (target 'gpt4_pooled_corrected_v0_released').
PAPER_VALUE_CORRECTED_GPT4 = 0.6494
PAPER_VALUE_CORRECTED_V0_GPT4 = 0.6502
HALLUCINATION_CUTOFF = 72
MISSING_THRESHOLD = 0.10
# Sensitivity lexicon (not the paper's): modern abstention phrasings missed by DECLINE_TO_ANSWER_TOKENS.
LENIENT_ABSTAIN = re.compile(
    r"\bi\s*(?:do\s*not|don'?t|cannot|can'?t|am\s+not\s+able\s+to|'?m\s+not\s+able\s+to|am\s+unable\s+to|'?m\s+unable\s+to)"
    r"\s+(?:know|verify|confirm|find|determine|recall|provide|identify|access|locate)"
    r"|\bnot\s+(?:certain|sure)\b|\b(?:cannot|can'?t|unable\s+to)\s+(?:verify|confirm|determine|locate|identify)"
    r"|\bdon'?t\s+have\s+(?:reliable|enough|specific|access|information|a\s+reliable)|\bno\s+reliable\b"
    r"|\b(?:i\s+am|i'?m)\s+not\s+aware\b|\bnot\s+able\s+to\s+(?:verify|confirm|find)", re.IGNORECASE)


# Round2-pilot validation scoring deviations (see module docstring); exposed by aggregate as 'scoring_deviations'.
SCORING_DEVIATIONS = (
    'citation cells: "<volume>, <reporter>, <page>" (the format the prompt requests) is rewritten to '
    '"<volume> <reporter> <page>" before eyecite (hallucination_corrected / headline)',
    'a LENIENT_ABSTAIN match with no parseable answer of the cell type is an abstention (non-hallucination, kept in '
    'the denominator) in hallucination_corrected / headline; pre-validation rule reported as pooled_corrected_v0',
)
CITATION_COMMA = re.compile(r'^\W*(\d+)\s*,\s*([A-Za-z][\w.\s]*?)\s*,\s*(\d+)\W*$')


UNICODE_SPACE = re.compile('[\u00a0\u2007\u2009\u200a\u202f]')


def normalize_citation(text):
    """'330, U.S., 518' -> '330 U.S. 518' (whole answer only); anything else is returned unchanged."""
    return CITATION_COMMA.sub(r'\1 \2 \3', text.strip()) if CITATION_COMMA.match(text.strip()) else text


# ------------------------------------------------------------------------------------------- task definitions
@functools.lru_cache(maxsize=None)
def task_spec(filename, var):
    """system_message, query_template and max_tokens of the zero-shot Task assigned to ``var`` in a tasks file."""
    tree = ast.parse((DV / filename).read_text())
    for node in tree.body:
        target = node.target if isinstance(node, ast.AnnAssign) else (node.targets[0] if isinstance(node, ast.Assign) else None)
        if isinstance(target, ast.Name) and target.id == var:
            call = node.value
            kw = {k.arg: k.value for k in call.keywords}
            query = kw['queries'].elt
            qkw = {k.arg: k.value for k in query.keywords}
            template = qkw['query_template']
            if isinstance(template, ast.Call):   # inspect.cleandoc("""...""") (few-shot tasks only)
                raise ValueError(f'{var}: not a zero-shot template')
            return {'system_message': ast.literal_eval(qkw['system_message']),
                    'query_template': ast.literal_eval(template),
                    'max_tokens': ast.literal_eval(kw['max_tokens']) if 'max_tokens' in kw else 100,
                    'sampling_temperature': ast.literal_eval(kw['sampling_temperature']),
                    'postprocess': qkw['llm_answer_postprocess'].id if 'llm_answer_postprocess' in qkw else 'clean_simple',
                    'correctness': qkw['correctness_callback'].id}
    raise KeyError(f'{var} not in {filename}')


def _prompts():
    out = {}
    for key, _, _, fname, var, _, _ in CELLS:
        spec = task_spec(fname, var)
        src = Source(repo=str(DV), path=fname, locator=f'{var}: Query(system_message=..., query_template=...)',
                     url=DV_URL)
        out[f'system_{key}'] = PromptTemplate(f'system_{key}', spec['system_message'], style='literal', source=src)
        out[f'query_{key}'] = PromptTemplate(f'query_{key}', spec['query_template'], style='format', source=src,
                                             note='rendered user message = released query text (models.Query.query)')
    return out


# ------------------------------------------------------------------------------------------- authors' scorer
@functools.lru_cache(maxsize=None)
def official():
    """The authors' cleaners and correctness callbacks, loaded from correctness_checks.py without its module body
    (which imports the OpenAI SDK, settings paths and NLI prompts). Needs eyecite, thefuzz, word2number, spaCy."""
    import en_core_web_sm
    import pandas as pd
    import regex
    from eyecite import get_citations
    from thefuzz import fuzz
    from word2number import w2n
    src = Source(repo=str(DV), path='correctness_checks.py')
    ann = {k: extract_text(src, 'py_var', var=k) for k in ('JUDGE_TOKEN_BLACKLIST', 'DECLINE_TO_ANSWER_TOKENS')}
    mapping = pd.read_csv(DV / 'data/covariates/scdb_justice_name_map.csv', index_col=False)
    utils = load_functions(DV / 'utils.py', ['rstrip_period'])
    names = ['extract_few_shot_answer', 'declined_to_answer', 'clean_simple', 'clean_judge_name', 'clean_circuit',
             'clean_district', 'name_correctness', 'usdc_court_id_correctness', 'coa_court_id_correctness',
             'citation_correctness']
    consts = ('SCDB_JUSTICE_NAME_REGEX_FULL', 'SCDB_JUSTICE_NAME_REGEX_LAST', 'SCDB_JUSTICE_NAME_REGEX_LAST_ONLY',
              'FEW_SHOT_ANSWER_REGEX', 'STATE_REGEX', 'DISTRICT_REGION_REGEX', 'DISTRICT_REGION_ABREVIATION_REGEX',
              'DISTRICT_REGION_ABBREVIATION_MAPPING')
    return load_functions(DV / 'correctness_checks.py', names, constants=consts, extra_globals={
        'SCDB_JUSTICE_MAPPING': mapping, 'SPACY_NLP': en_core_web_sm.load(), 'get_citations': get_citations,
        'fuzz': fuzz, 'w2n': w2n, 'Match': re.Match, 'regex': regex, 'AnswerType': dict,
        'rstrip_period': utils['rstrip_period'], **ann})


@functools.lru_cache(maxsize=None)
def justice_names():
    """utils.get_judge_name_from_scdb_id as used for the Author true answer: '<first> <last>'."""
    import pandas as pd
    m = pd.read_csv(DV / 'data/covariates/scdb_justice_name_map.csv', index_col=False)
    return {int(r.scdb_id): f'{r.first} {r.last}' for r in m.itertuples()}


def paper_correctness(cell, text, truth):
    """(correctness, cleaned answer) under the authors' post-processor and callback for the cell."""
    fns = official()
    _, _, _, _, _, post, corr = CELL[cell]
    cleaned = fns[post](text)
    return float(fns[corr]({'answer': cleaned}, {'answer': truth})), cleaned


def unparseable(cell, cleaned):
    """No recognisable answer of the cell's type in the cleaned answer (for no_answer_share; not a verdict)."""
    if cleaned == '-99':
        return False
    if cell == 'court_coa':
        return cleaned not in {str(i) for i in range(1, 14)}
    if cell == 'court_usdc':
        return not official()['STATE_REGEX'].search(cleaned)
    if cell.startswith('citation'):
        from eyecite import get_citations
        return not get_citations(cleaned)
    return not cleaned.strip()


@functools.lru_cache(maxsize=None)
def _justice_last_names():
    return re.compile(r'\b(?:' + official()['SCDB_JUSTICE_NAME_REGEX_LAST_ONLY'].pattern + r')\b')


def no_answer_given(cell, text, cleaned):
    """Scoring deviation 2: no answer of the cell's type. As ``unparseable``, except for Author, where any cleaned
    text counts as parseable: there it means no SCDB justice surname appears in the answer (case-sensitive, whole word;
    a case name containing a justice-like surname keeps the row scored, the conservative direction)."""
    if cell == 'author_scotus':
        return cleaned != '-99' and not _justice_last_names().search(text)
    return unparseable(cell, cleaned)


def is_hallucination(correctness):
    return correctness <= HALLUCINATION_CUTOFF and correctness != -99


def corrected_usdc_court(text, district):
    """The authors' usdc_court_id_correctness with the full district name as truth (the truth their GPT-3.5 run
    used: 'United States District Court for the Eastern District of Arkansas'), i.e. district-level matching."""
    fns = official()
    return float(fns['usdc_court_id_correctness']({'answer': fns['clean_district'](text)}, {'answer': district}))


# ------------------------------------------------------------------------------------------------ data
@functools.lru_cache(maxsize=None)
def released_rows():
    import pandas as pd
    return pd.read_csv(HF, dtype=str, keep_default_na=False)


@functools.lru_cache(maxsize=None)
def released_other_rows():
    import pandas as pd
    return pd.read_csv(HF_OTHER, dtype=str, keep_default_na=False)


@functools.lru_cache(maxsize=None)
def _cases():
    """{cell: [case dict]} from the released rows: one case per GPT-4 row (release order), joined with the GPT-3.5 row
    of the same case on (query, citation, year, court_slug). For USDC Court the GPT-4 row holds the state (the
    Dataverse code's true answer, case.state) and the Llama 2 / PaLM 2 rows the full district name (district-level
    truth, 'truth_district'). The released "GPT 3.5" USDC-Court rows are copies of the Llama 2 rows (asserted)."""
    df = released_rows()
    other = released_other_rows()
    llama = {(r.query, r.citation, r.year, r.court_slug): r for r in other[other.llm == 'Llama 2'].itertuples()}
    palm = {(r.query, r.citation, r.year, r.court_slug): r for r in other[other.llm == 'PaLM 2'].itertuples()}
    names = justice_names()
    out = {}
    for key, task, court, *_ in CELLS:
        sub = df[(df.task == task) & (df.court_level == court) & (df.prompt_style == 'zero_shot')]
        g4 = sub[sub.llm == 'GPT 4'].sort_values('id', key=lambda s: s.astype(int))
        g35 = defaultdict(list)
        for b in sub[sub.llm == 'GPT 3.5'].sort_values('id', key=lambda s: s.astype(int)).itertuples():
            g35[(b.query, b.citation, b.year, b.court_slug)].append(b)
        assert len(g4) == sum(map(len, g35.values())) > 4000
        cases = []
        for a in g4.itertuples():
            b = g35[(a.query, a.citation, a.year, a.court_slug)].pop(0)
            if key != 'court_usdc':
                assert a.example_correct_answer == b.example_correct_answer
            truth = names[int(a.example_correct_answer)] if key == 'author_scotus' else a.example_correct_answer
            case = {'id': f'C067:{key}:{a.id}', 'cell': key, 'task': task, 'court_level': court,
                    'query': a.query, 'truth': truth, 'citation': a.citation, 'year': int(a.year),
                    'court_slug': a.court_slug, 'case_source': a.case_source,
                    'released': {'gpt4': {'row_id': a.id, 'output': a.llm_output,
                                          'correctness': float(a.correctness_score)},
                                 'gpt35': {'row_id': b.id, 'output': b.llm_output,
                                           'correctness': float(b.correctness_score)}}}
            if key == 'court_usdc':
                k = (a.query, a.citation, a.year, a.court_slug)
                ll, pa = llama[k], palm[k]
                assert ll.example_correct_answer == pa.example_correct_answer == b.example_correct_answer
                assert (ll.llm_output, ll.correctness_score) == (b.llm_output, b.correctness_score)
                case['truth_district'] = ll.example_correct_answer
            cases.append(case)
        assert not any(g35.values())
        out[key] = cases
    return out


class _PerCellCap(SingleTurn):
    def params_for(self, item):
        return {**self.params, 'max_tokens': self.caps[item['cell']]}


# ---------------------------------------------------------------------------- earlier pipeline runs
# experiments/run_minimal_setups/runs/earlier/runs/<claim>-<variant>-L<N>[-<tag>]/<model>/: one shelflife.runner directory per model.
# Registered: every directory that sent at least one request (complete or partial); tests/test_prior_runs.py
# replays each one against this config. Excluded: aborted, smoke-test and superseded directories, and directories
# that sent nothing. Listed explicitly (written 2026-10-05 from the run directories; nothing is globbed at import).
# Runs made before the 2026-10-05 scorer patch replay byte-identically (the patch changed scoring only); their
# saved scores are pre-patch and the collector rescores them offline.

_R2 = 'experiments/run_minimal_setups/runs/earlier/runs'
_NOTHING_SENT = ('no request reached a provider: every call is not_submitted (circuit breaker open, or stale ledger '
                 'rows after a package-hash change); the directory only records the attempts')


def _unsent(*runs):
    """excluded_runs entries for per-model directories that sent nothing: (run, space-separated models)."""
    return {f'{_R2}/{run}/{m}': _NOTHING_SENT for run, models in runs for m in models.split()}


@register
class C067LegalHallucinations(Experiment):
    claim_id = 'C067'
    claim_set_key = 'C067'
    maturity = 'ready'
    title = 'Large Legal Fictions: legal hallucinations (Dahl et al. 2024)'
    claim = ('Asked verifiable questions about real U.S. federal cases, LLMs give wrong answers (hallucinate) at high '
             'rates, more so for lower courts and less prominent cases (GPT-4 58% to Llama 2 88% pooled).')
    paper = {'title': 'Large Legal Fictions: Profiling Legal Hallucinations in Large Language Models',
             'authors': 'Dahl, Magesh, Suzgun, Ho', 'first_date': '2024-01-02',
             'venue': 'Journal of Legal Analysis 16(1): 64-93 (2024)', 'url': 'https://arxiv.org/abs/2401.01301'}
    source = Source(repo=str(DV), path='tasks_scotus.py', url=DV_URL, note='Dataverse replication package v1.0')
    default_variant = 'zero_shot'

    def build_variants(self):
        caps = {key: task_spec(fname, var)['max_tokens'] for key, _, _, fname, var, _, _ in CELLS}
        zero = Variant(
            'zero_shot', 'Zero-shot greedy reference-based queries, six task x court cells (Table 3 low-complexity '
                         'tasks; Court-for-SCOTUS, Existence and few-shot omitted)',
            prompts=_prompts(),
            params={'temperature': 0, 'top_p': 1, 'frequency_penalty': 0.0, 'presence_penalty': 0.0, 'max_tokens': 100},
            strategy='single_turn', options={'cells': [c[0] for c in CELLS], 'max_tokens_by_cell': caps,
                                             'headline': 'corrected'},
            deviations=('gpt-4-1106-preview retired (stand-in gpt-4-turbo-2024-04-09); gpt-3.5-turbo-0613 via '
                        'OpenRouter',
                        'only the greedy (temperature 0) request; the 10 temperature-1 samples feed calibration only',
                        'public ~90% of cases (reserve.csv embargoed); six cells of the low-complexity table',
                        'scorer environment: spaCy en_core_web_sm 3.8.0 (paper 3.5.0), python-Levenshtein 0.27 '
                        '(paper 0.21)',
                        'headline adds a district-level USDC Court rule (the paper rule is reported alongside)',
                        'scope: the verdict covers only the pooled zero-shot reference-based hallucination rate; the '
                        'claim\'s false-premise (contra-factual) limb, the court-level and prominence (PageRank) '
                        'gradients, Existence, few-shot prompts, higher-complexity tasks and calibration are not tested',
                        'reasoning models run under the panel policy: OpenAI reasoning/Anthropic models drop '
                        'temperature=0 and top_p (no greedy decoding) and all reasoning models get a 4,096-token cap '
                        'instead of 30/100',
                        'empty or truncated answers are missing (excluded, counted) rather than scored by the authors\' '
                        'callbacks (which score an empty citation as a hallucination)',
                        'Puerto Rico / Virgin Islands USDC truths are not normalised by clean_district (inherited; '
                        '3.1% of USDC cases; at most ~0.5 pp on the pooled rate)'))
        robust_parse = zero.override(
            'robust_parse', options={'parse': 'robust'},
            description='zero_shot scored with the robust fallback (identical requests)',
            deviations=('parse=robust: an unparseable, non-abstained answer is re-scored after _robust.normalize and '
                        'unicode-space folding; kept only when that yields a parseable answer',))
        robust = robust_parse.override(
            'robust', options={'max_tokens_by_cell': {k: max(v, _robust.MIN_ROBUST_TOKENS) for k, v in caps.items()},
                               'min_cap_reasoning': 16384, 'truncated': 'score'},
            description='robust_parse with a 1,024-token per-cell cap floor and min_cap_reasoning 16384 for panel runs',
            deviations=('per-cell output cap 30/100 -> 1024; reasoning models get a 16,384-token cap',
                        "truncated='score': a non-empty answer cut off at 1024 / 16384 tokens is scored by the "
                        "authors' callbacks on its text, not missing; empty answers stay missing"))
        robust = dataclasses.replace(robust, faithful=False)   # per-cell caps live in options: a request change
        return {'zero_shot': zero, 'robust_parse': robust_parse, 'robust': robust}

    def items(self, variant, limit=None):
        cases = _cases()
        order = {}
        for key in variant.options['cells']:
            rows = [dict(c) for c in cases[key]]
            random.Random(f'{SEED}:{key}').shuffle(rows)
            order[key] = rows
        out = []
        for i in range(max(len(v) for v in order.values())):
            for key in variant.options['cells']:
                if i < len(order[key]):
                    out.append(order[key][i])
        return out[:limit] if limit else out

    def program(self, variant):
        def build(item):
            return [{'role': 'system', 'content': variant.prompts[f'system_{item["cell"]}'].render()},
                    {'role': 'user', 'content': item['query']}]
        cap = variant.options.get('min_cap_reasoning')
        prog = _PerCellCap(build, variant.params, case_extra={'min_cap_reasoning': cap} if cap else None)
        prog.caps = dict(variant.options['max_tokens_by_cell'])
        return prog

    # --------------------------------------------------------------- scoring
    def score(self, item, output, variant):
        robust = variant.options.get('parse', 'paper') == 'robust'
        cut_missing = not robust or _robust.truncation_missing(variant)
        out = self._score(item, output, cut_missing)
        if not robust:
            return out
        if out.get('scorer_error') or out.get('missing'):
            return {**out, 'parse_path': None}
        if out.get('empty_truncated'):
            return {**out, 'parse_path': 'paper'}   # the authors' callback verdict on an empty answer; nothing to re-read
        if out['unparseable'] and not out['abstained_lenient']:
            text = output.get('content') or ''
            alt = _robust.normalize(UNICODE_SPACE.sub(' ', text))
            if alt and alt != text:
                second = self._score(item, {**output, 'content': alt}, cut_missing)
                if second.get('missing') is None and not second.get('scorer_error') and not second['unparseable']:
                    return {**second, 'parse_path': 'robust', 'robust_text': alt}
            return {**out, 'parse_path': None}
        return {**out, 'parse_path': 'paper'}

    def _score(self, item, output, cut_missing=True):
        """cut_missing=False (robust, truncated='score'): a cut-off answer is scored by the authors' callbacks like any
        other, including an EMPTY cut-off answer (reasoning budget exhausted: Citation -> 0 = hallucination,
        Court/Author -> -99 = abstention, the paper code's mapping). A cut-off answer is
        labelled 'truncated' even when empty; 'empty' means an empty answer that was not cut off (always missing)."""
        text = output.get('content') or ''
        truncated = output.get('finish_reason') == 'length' or output.get('status') == 'truncated'
        missing = ('truncated' if cut_missing else None) if truncated else ('empty' if not text.strip() else None)
        try:
            corr, cleaned = paper_correctness(item['cell'], text, item['truth'])
        except Exception as e:   # never seen on the 54,172 released outputs; kept visible, not silently scored
            return {'scorer_error': f'{type(e).__name__}: {e}', 'missing': missing, 'hallucination': None,
                    'hallucination_corrected': None, 'hallucination_abstain_lenient': None, 'abstained': None}
        raw = {'correctness': corr, 'cleaned': cleaned, 'truncated': truncated,
               **({'truncated_scored': True} if truncated and not missing else {}),
               **({'empty_truncated': True} if truncated and not missing and not text.strip() else {})}
        if missing:
            # excluded from rates, counted; the authors' verdict is kept for the record only
            return {**raw, 'missing': missing, 'paper_verdict_if_scored': is_hallucination(corr),
                    'hallucination': None, 'hallucination_corrected': None, 'hallucination_abstain_lenient': None,
                    'abstained': None}
        out = {**raw, 'missing': None, 'hallucination': is_hallucination(corr), 'abstained': corr == -99}
        if item['cell'] == 'court_usdc':
            c2 = corrected_usdc_court(text, item['truth_district'])
            out.update(correctness_corrected=c2, hallucination_corrected=is_hallucination(c2))
        else:
            out['hallucination_corrected'] = out['hallucination']
        out['hallucination_corrected_v0'] = out['hallucination_corrected']   # pre-validation corrected rule
        lenient = bool(LENIENT_ABSTAIN.search(text)) and not out['abstained']
        out['abstained_lenient'] = out['abstained'] or lenient
        out['unparseable'] = unparseable(item['cell'], cleaned)
        # scoring deviation 1 (validation): comma-format citations rewritten before eyecite
        if item['cell'].startswith('citation'):
            norm = normalize_citation(text)
            if norm != text:
                c3, cleaned3 = paper_correctness(item['cell'], norm, item['truth'])
                out.update(comma_normalized=True, correctness_corrected=c3, cleaned_normalized=cleaned3,
                           hallucination_corrected=is_hallucination(c3),
                           unparseable=unparseable(item['cell'], cleaned3))
        # scoring deviation 2 (validation): lenient abstention with no parseable answer = abstention
        out['abstained_no_answer'] = lenient and (out['unparseable'] if item['cell'] != 'author_scotus'
                                                  else no_answer_given(item['cell'], text, cleaned))
        if out['abstained_no_answer']:
            out['hallucination_corrected'] = False
        out['hallucination_abstain_lenient'] = out['hallucination_corrected'] and not lenient
        out['no_answer'] = out['abstained_lenient'] or out['unparseable']
        return out

    @staticmethod
    def _pooled(rows, field):
        by = defaultdict(list)
        for r in rows:
            by[r['item']['cell']].append(bool(r['score'][field]))
        rates = [sum(v) / len(v) for v in by.values() if v]
        return sum(rates) / len(rates) if rates else float('nan')

    def aggregate(self, rows, variant):
        ok = lambda r: r.get('score') is not None and r['score'].get('hallucination') is not None
        usable = [r for r in rows if ok(r)]
        failed = sum(r.get('output') is None for r in rows)
        missing = sum(bool((r.get('score') or {}).get('missing')) for r in rows)
        errors = sum(bool((r.get('score') or {}).get('scorer_error')) for r in rows)
        share = lambda k: sum(bool(r['score'][k]) for r in usable) / len(usable) if usable else None
        out = {'n': len(rows), 'scored': len(usable), 'failed_calls': failed, 'missing': missing,
               'missing_empty': sum((r.get('score') or {}).get('missing') == 'empty' for r in rows),
               'missing_truncated': sum((r.get('score') or {}).get('missing') == 'truncated' for r in rows),
               'scorer_errors': errors, 'missing_share': (len(rows) - len(usable)) / len(rows) if rows else None,
               'cells': {}}
        for key in variant.options['cells']:
            all_sub = [r for r in rows if r['item']['cell'] == key]
            sub = [r for r in all_sub if ok(r)]
            cell = {'rule_paper': stats.rate_summary(all_sub, lambda r: r['score']['hallucination'], usable=ok),
                    'rule_corrected': stats.rate_summary(all_sub, lambda r: r['score']['hallucination_corrected'],
                                                         usable=ok),
                    'rate_abstain_lenient': sum(r['score']['hallucination_abstain_lenient'] for r in sub) / len(sub) if sub else None,
                    'abstention_rate': sum(r['score']['abstained'] for r in sub) / len(sub) if sub else None,
                    'no_answer_share': sum(r['score']['no_answer'] for r in sub) / len(sub) if sub else None,
                    'missing': sum(bool((r.get('score') or {}).get('missing')) for r in all_sub)}
            out['cells'][key] = cell
        pooled = lambda field: self._pooled(usable, field) if usable else None
        out['pooled_paper'] = pooled('hallucination')
        value = pooled('hallucination_corrected')
        out['pooled_corrected'] = value
        out['pooled_corrected_v0'] = (self._pooled(usable, 'hallucination_corrected_v0')
                                      if usable and all('hallucination_corrected_v0' in r['score'] for r in usable)
                                      else None)
        out['comma_normalized'] = sum(bool(r['score'].get('comma_normalized')) for r in usable)
        out['abstained_no_answer'] = sum(bool(r['score'].get('abstained_no_answer')) for r in usable)
        out['scoring_deviations'] = list(SCORING_DEVIATIONS)
        _, lo, hi = stats.cluster_bootstrap(usable, lambda r: r['item']['id'],
                                            lambda rs: self._pooled(rs, 'hallucination_corrected'), n_boot=2000, seed=0)
        out['headline_abstain_lenient'] = pooled('hallucination_abstain_lenient')
        out['abstention_rate'] = share('abstained')
        out['abstention_rate_lenient'] = share('abstained_lenient')
        out['unparseable_share'] = share('unparseable')
        out['no_answer_share'] = share('no_answer')
        out['abstention_sensitive'] = (bool(usable) and lo == lo and hi == hi
                                       and abs(value - out['headline_abstain_lenient']) > (hi - lo) / 2)
        out['headline_reliable'] = bool(usable) and out['missing_share'] <= MISSING_THRESHOLD and \
            all(out['cells'][k]['rule_corrected']['scored'] for k in variant.options['cells'])
        if variant.options.get('parse', 'paper') == 'robust':
            paths = Counter((r.get('score') or {}).get('parse_path') for r in usable)
            out.update(parse='robust', parsed_paper=paths.get('paper', 0), parsed_robust=paths.get('robust', 0),
                       unparsed=paths.get(None, 0), validity_rate=len(usable) / len(rows) if rows else None,
                       truncated_scored=sum(bool(r['score'].get('truncated_scored')) for r in usable),
                       truncated_empty_scored=sum(bool(r['score'].get('empty_truncated')) for r in usable))
        out['shelflife'] = {
            'metric': 'legal_hallucination_rate_pooled6', 'value': value, 'ci': [lo, hi] if usable else None,
            'n': len(usable), 'direction': +1, 'null_value': 0.0, 'paper_value': PAPER_VALUE_CORRECTED_GPT4,
            'definition': ('Unweighted mean over six zero-shot cells (Court USCOA/USDC, Citation SCOTUS/USCOA/USDC, '
                           'Author SCOTUS) of the share of greedy answers the authors\' scorer marks wrong '
                           '(correctness <= 72; code-lexicon abstentions and lenient abstentions with no parseable answer '
                           'are non-hallucinations; empty/truncated answers are missing; comma-format citations '
                           'normalised before eyecite), with USDC Court scored at district level; paper_value = released GPT-4 '
                           '(gpt-4-1106-preview) outputs under the same rule; CI resamples cases. See '
                           'headline_abstain_lenient / no_answer_share / headline_reliable.')}
        return out

    # ------------------------------------------------------------- archived outputs
    def released_aggregate(self, llm, rule='hallucination'):
        """Our scorer on the authors' released outputs for one model, every public case: {cell: rate, ...}."""
        variant = self.variant()
        rows = []
        for key in variant.options['cells']:
            for case in _cases()[key]:
                rec = case['released'][llm]
                rows.append({'item': case, 'output': {'content': rec['output']},
                             'score': self.score(case, {'content': rec['output']}, variant)})
        by = defaultdict(list)
        for r in rows:
            by[r['item']['cell']].append(r)
        cells = {k: sum(bool(r['score'][rule]) for r in v) / len(v) for k, v in by.items()}
        # the released "GPT 3.5" USDC-Court rows are Llama 2's outputs (see module docstring): not compared
        comp = [r for r in rows if not (llm == 'gpt35' and r['item']['cell'] == 'court_usdc')]
        agree = sum(r['score']['correctness'] == r['item']['released'][llm]['correctness'] for r in comp) / len(comp)
        return {'cells': cells, 'pooled': sum(cells.values()) / len(cells), 'correctness_agreement': agree,
                'n': len(rows), 'n_compared': len(comp)}

    targets = tuple(
        PaperTarget(f'{llm}_{cell}_released', PAPER_TABLE[llm][cell], 0.01,
                    f'Table "Hallucination rates across levels of the federal judiciary (low complexity tasks)", '
                    f'zero-shot, {cell} (CategoryOneTable.tex)',
                    {'gpt4': 'gpt-4-1106-preview', 'gpt35': 'gpt-3.5-turbo-0613'}[llm], 'zero_shot', 'archived_outputs',
                    compute=(lambda exp, llm=llm, cell=cell: _released_cached(exp, llm)['cells'][cell]),
                    note='Our scorer on the released outputs of the public ~90% of cases; the paper also includes the '
                         'embargoed ~10% (reserve.csv). Tolerance 0.01: rounding (0.0005) plus the reserve share '
                         '(0.1 x a difference of at most ~0.05 between the halves, about 4 SE).')
        for llm in ('gpt4', 'gpt35') for cell in PAPER_TABLE['gpt4'] if (llm, cell) != ('gpt35', 'court_usdc')) + (
        PaperTarget('gpt35_court_usdc', PAPER_TABLE['gpt35']['court_usdc'], 0.01, 'CategoryOneTable.tex, zero-shot, '
                    'USDC Court, GPT 3.5', 'gpt-3.5-turbo-0613', 'zero_shot', 'unverifiable',
                    note='Not a GPT-3.5 number: the released "GPT 3.5" USDC-Court rows are byte-identical to Llama 2\'s '
                         '(outputs and scores), and the table shows GPT 3.5 = Llama 2 in both prompt styles. The true '
                         'GPT-3.5 outputs were not released; do not use as a reproduction target.'),
        PaperTarget('gpt4_correctness_agreement_released', 1.0, 0.0001, 'released correctness_score column '
                    '(HF reglab/legal_hallucinations dataset.csv)', 'gpt-4-1106-preview', 'zero_shot', 'archived_outputs',
                    compute=lambda exp: _released_cached(exp, 'gpt4')['correctness_agreement'],
                    note='Share of released GPT-4 outputs whose correctness score our port reproduces exactly.'),
        PaperTarget('gpt35_correctness_agreement_released', 1.0, 0.0001, 'released correctness_score column',
                    'gpt-3.5-turbo-0613', 'zero_shot', 'archived_outputs',
                    compute=lambda exp: _released_cached(exp, 'gpt35')['correctness_agreement'],
                    note='USDC-Court rows excluded (they are Llama 2 outputs); 1 of 22,588 compared differs (spaCy NER '
                         'fallback on a judge name: 23 vs 18; both hallucinations).'),
        PaperTarget('gpt4_pooled_corrected_released', PAPER_VALUE_CORRECTED_GPT4, 0.00005,
                    'derived: released GPT-4 outputs, six cells, USDC Court district-level, comma citations '
                    'normalised, lenient no-answer abstentions', 'gpt-4-1106-preview',
                    'zero_shot', 'archived_outputs',
                    compute=lambda exp: round(_released_cached(exp, 'gpt4', 'hallucination_corrected')['pooled'], 4),
                    note='Anchors shelflife.paper_value; not a paper number (the paper rule gives pooled_paper).'),
        PaperTarget('gpt4_pooled_corrected_v0_released', PAPER_VALUE_CORRECTED_V0_GPT4, 0.00005,
                    'derived: released GPT-4 outputs, six cells, USDC Court district-level (pre-validation rule)',
                    'gpt-4-1106-preview', 'zero_shot', 'archived_outputs',
                    compute=lambda exp: round(_released_cached(exp, 'gpt4', 'hallucination_corrected_v0')['pooled'], 4),
                    note='The corrected rule before the earlier pipeline scoring deviations (pooled_corrected_v0).'),
        PaperTarget('gpt4_pooled_reference_based_all_tasks', 0.58, 0.02, 'Sec. 5.1 / Fig. 1 text: pooled reference-based '
                    'hallucination 58% (GPT-4) to 88% (Llama 2)', 'gpt-4-1106-preview', 'zero_shot', 'unverifiable',
                    note='Pools all reference-based tasks and prompt styles; this port covers six zero-shot cells.'),
    )


@functools.lru_cache(maxsize=None)
def _released_cached_inner(llm, rule):
    from .. import registry
    return registry.get('C067').released_aggregate(llm, rule)


def _released_cached(exp, llm, rule='hallucination'):
    return _released_cached_inner(llm, rule)
