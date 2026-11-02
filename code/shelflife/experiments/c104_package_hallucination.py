"""C104 (claim-set key HL-N01) — package hallucinations in code generation (Spracklen et al., "We Have a Package for
You! A Comprehensive Analysis of Package Hallucinations by Code Generating LLMs", arXiv 2406.10279, USENIX Sec 2025).

Claim: asked for code, or for the packages needed to solve a coding task, LLMs recommend package names that do not exist
on PyPI/npm: 19.7% of recommended packages across 16 models (GPT-4 Turbo 3.59%, GPT-3.5 Turbo 5.76%, open models 21.7%
on average); rates are higher for prompts about recent (last-year) topics than all-time topics.

Paper: arXiv 2406.10279 v3 (2025-03-02, the USENIX version; prompts, parameters and Table "Hallucination Percentages for
all models tested using Python code" are unchanged from v1 for Python). Sec. 3.3 "Detection Methodology and Heuristics",
Appendix "System messages", Table 9 (model parameters), Python results table, Fig. 4 (recent vs all-time).
Sources (all offline):
  * official repository https://github.com/Spracks/PackageHallucination @ bd3031cce6a0 (MIT), cloned to
    data/sources/C104_PackageHallucination/: prompt datasets Data/Python/LLM_{LY,AT}.json, the 10-Jan-2024 PyPI master
    list Data/Python/pypi_package_names.csv, the false-positive list, the prompt builder
    generate_package_names.py (mode 2) and the scorer package_detection.py / custom_parse_python.py; per-figure data in
    Plots/Data/. Per-prompt outputs were deliberately NOT released (Security & Ethics section of the README), so no
    archived-output target exists.
  * data/sources/C104_pypi_simple_index_2026-10-04/: every project name on PyPI on 2026-10-04 (PEP 691 simple index),
    for the "confirmed absent from the live index" rule of the minimal design, frozen so scoring is offline and
    deterministic.
  * data/sources/C104_spracklen_arxiv_v3/: the arXiv v3 LaTeX source (appendix prompt figures, parameter table, results
    tables), used by the claim tests.

Heuristic naming. The paper's three heuristics are (1) parse "pip install" in generated code, (2) ask the model which
packages the *generated code* needs, (3) ask the model which packages would help solve the *original prompt*. The
released code calls (2) "mode 1" and (3) "mode 2". The minimal design ("heuristic 2 only, direct package
recommendation") is the paper's heuristic 3 = code mode 2: one call per prompt, no code generation.

Variants
  h3_direct           default, faithful. Code mode 2 (generate_package_names.py): system message and user prefix
                      verbatim from the released code, LLM-generated Python prompts (LLM_LY recent / LLM_AT all-time),
                      temperature 0.01, top_p 0.9, max_tokens 64 (Table 9; code defaults). Items are a seeded
                      interleaving LY, AT, LY, AT, ... over the full 9,814 prompts, so ``--limit N`` gives N/2 of each.
  robust_parse        h3_direct requests, parse='robust': the paper rule first; only when
                      it yields no package from a non-blank, non-"None" answer, the name-shaped extraction judged by
                      the same Jan-2024 list; blank visible output unparsed; reliable iff <= 10% unparsed or missing.
                      metrics['paper'] is then this paper-first rate, metrics['paper_rule_only'] the unmodified rule.
  robust              robust_parse + max_tokens floor 1024 (cut names count as hallucinations under the paper rule)
                      and min_cap_reasoning 16384 via case_extra (M144 blank outputs), for new panel runs.
                      Robust != paper on rows the paper scored (as zero packages): blank or think-only output and
                      refusals / prose with no package-shaped entry become unparsed (missing) instead of contributing
                      nothing; newline/markdown lists the paper split into nothing are parsed. Rows where the paper
                      rule found >= 1 package are unchanged. the panel: 80 M144, 20 M183, <= 2 elsewhere unparsed;
                      1 robust-parsed row (M204).

Deviations (h3_direct): top_k=20, typical_p=1, epsilon/eta cutoff 0, repetition penalty 1 are HF-generate settings with
no OpenAI-compatible equivalent (the paper notes top-k is not modifiable through the OpenAI API, so its GPT runs had the
same limitation); the GPT snapshots used by the paper are not named (runs in early 2024); the paper's per-dataset
columns pool heuristics 2 and 3 (code modes 1 and 2; "pip install" is a separate column) and its Total adds Stack
Overflow prompts and pip install, whereas this port measures heuristic 3 on LLM-generated prompts only; the policy layer
raises the 64-token cap for reasoning models and drops temperature/top_p for OpenAI reasoning and Anthropic models.

Scoring (``score``) is the authors' package_search_python path for an API model (get_pre_post_info -> no custom pre/post
parsing): the response goes through the pandas CSV round trip of detect_packages (so a bare "None"/empty response
becomes NaN -> 'nan' and is skipped), is split on commas, each entry is normalize_python()-ed (their function, loaded
from package_detection.py), multi-word entries are dropped, duplicates and names of <= 2 characters are removed
(delete_dupes_and_empty), and check_packages() splits the rest into valid (in the normalised 10-Jan-2024 master list,
read with pandas exactly as they read it) and hallucinated (not in it and not in the false-positive list).
Metric (paper): package hallucination rate = hallucinated / (valid + hallucinated), pooled over prompts.

Headline (not in the paper), ``shelflife`` = package hallucination rate as of the model's release date:
  * name-shaped filter: <think>...</think> leakage removed; entries split on commas AND newlines; markdown, quotes,
    brackets, list markers, extras and version pins stripped; only valid PEP 508 names kept (others are counted as
    ``artifact_share``, not as hallucinations); truncated outputs lose their last entry. This removes model-specific
    formatting artefacts that the authors' false-positive list (built from the 16 2024 models' outputs) does not cover.
  * registry as of release: a name absent from the 10-Jan-2024 list (minus false positives) is a hallucination if it is
    also absent from the frozen 2026-10-04 index, or if its first PyPI upload is on/after the model's release date (or
    it has no files). First-upload dates are a POST-RUN step: ``candidate_names(run_dir)`` lists the names that need a
    date, ``freeze_first_uploads(names)`` fetches them from the PyPI JSON API into
    data/sources/C104_pypi_first_upload/first_upload.json (free metadata); rescoring then reads that frozen file.
    Until every candidate is frozen ``value`` is None and ``bounds`` = [2026-index rule, Jan-2024-list rule].
  * null_value 0.01 (design spec p_control); ``paper_value`` None because 0.0598 (``paper_value_reported``) is on a
    different scale (heuristics 2+3, paper rule): anchor on the REPRO_MODELS values under the headline metric.
  * ``no_package_share`` / ``reliable`` (False above 10%) are inside the shelflife dict.
The paper rule (``rate_paper``) and its 2026-snapshot variant (``rate_confirmed``) on raw responses stay as diagnostics.
"""
import ast
import functools
import gzip
import io
import json
import random
import re
from collections import Counter

from . import _robust
from ..paths import DATA
from ..program import Program, SingleTurn
from ..prompts import PromptTemplate, Source
from ..registry import Experiment, PaperTarget, Variant, register
from ..sourcecode import load_functions
from .. import stats

claim_set_key = 'HL-N01'
REPO = DATA / 'sources' / 'C104_PackageHallucination'
COMMIT = 'bd3031cce6a0007997a0604d3ad3365031ed79b2'
URL = 'https://github.com/Spracks/PackageHallucination'
PYDATA = REPO / 'Data' / 'Python'
LIVE = DATA / 'sources' / 'C104_pypi_simple_index_2026-10-04' / 'pypi_simple_names.txt.gz'
ARXIV = DATA / 'sources' / 'C104_spracklen_arxiv_v3'

SEED = 2406_10279
DATASETS = (('LLM_LY', 'LLM_LY.json'), ('LLM_AT', 'LLM_AT.json'))   # recent first, as in package_detection.datasets

# Run settings for earlier pipeline (see experiments/run_minimal_setups/runs/earlier/ports/C104.md).
RUN_VARIANT = 'h3_direct'
MINIMAL_LIMIT = 400            # 200 LLM_LY + 200 LLM_AT prompts -> ~900 packages (GPT-era models give ~2.1-2.4 per call)
REPRO_MODELS = ['openai-direct/gpt-3.5-turbo-0125', 'openai-direct/gpt-4-turbo-2024-04-09']
REPRO_LIMIT = 400              # 200 LY + 200 AT, the minimal design's full prompt count
NO_PACKAGE_THRESHOLD = 0.10
FIRST_UPLOAD = DATA / 'sources' / 'C104_pypi_first_upload' / 'first_upload.json'
ROBUST_MIN_CAP_REASONING = 16384   # robust: M144 returned blank visible output on 74/400 prompts at the 4096 cap
_EXPLICIT_NONE = re.compile(r'^(?:none|n/?a|nan|null|no (?:python )?packages?(?: (?:are )?(?:needed|required))?)\.?$', re.I)
PAPER_VALUE_REPORTED = 0.0598   # GPT-3.5 Turbo, LLM-generated prompts, heuristics 2+3 pooled, Jan-2024 list
# Release dates for models outside the release catalogue (shelflife.models.release_date returns None for them).
EXTRA_RELEASE_DATES = {'openai-direct/gpt-3.5-turbo-0125': '2024-01-25',
                       'openai-direct/gpt-4-turbo-2024-04-09': '2024-04-09',
                       'openai/gpt-3.5-turbo-0613': '2023-06-13'}
NAME_RX = re.compile(r'^[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?$')   # PEP 508 project name (after lowercasing)


def _fstring_in(path, function, target, mode):
    """Text of the f-string assigned to ``target`` inside ``if mode == <mode>`` of ``function`` (the code builds its
    prompts there), with {language} kept as a format field."""
    tree = ast.parse(path.read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function)
    for node in ast.walk(fn):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare) and \
                isinstance(node.test.comparators[0], ast.Constant) and node.test.comparators[0].value == mode:
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and stmt.targets[0].id == target:
                    parts = []
                    for v in stmt.value.values:
                        if isinstance(v, ast.Constant):
                            parts.append(v.value.replace('{', '{{').replace('}', '}}'))
                        else:
                            assert isinstance(v.value, ast.Name) and v.conversion == -1 and v.format_spec is None
                            parts.append('{' + v.value.id + '}')
                    return ''.join(parts)
    raise KeyError(f'{target} for mode {mode} not found in {path}')


_GEN = Source(repo=str(REPO), path='generate_package_names.py', commit=COMMIT, url=URL,
              locator='generate_packages, mode == 2')
SYSTEM = PromptTemplate('system', _fstring_in(REPO / 'generate_package_names.py', 'generate_packages', 'system_message', 2),
                        source=_GEN, note='f-string body; {language} = "Python"')
USER = PromptTemplate('user', _fstring_in(REPO / 'generate_package_names.py', 'generate_packages', 'prefix', 2)
                      + '{problem}', source=_GEN,
                      note='prefix f-string + sample.strip() (the LLM-generated prompt)')


def _fstring_assigned(path, function, target):
    """Text of the f-string assigned to ``target`` at the top level of ``function`` ({language} kept as a field)."""
    tree = ast.parse(path.read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function)
    for stmt in fn.body:
        if isinstance(stmt, ast.Assign) and getattr(stmt.targets[0], 'id', None) == target:
            parts = []
            for v in stmt.value.values:
                if isinstance(v, ast.Constant):
                    parts.append(v.value.replace('{', '{{').replace('}', '}}'))
                else:
                    assert isinstance(v.value, ast.Name) and v.conversion == -1 and v.format_spec is None
                    parts.append('{' + v.value.id + '}')
            return ''.join(parts)
    raise KeyError(f'{target} not found in {function} of {path}')


# Paper-table protocol:
# the "LLM Generated Prompts" column of the Python results table pools heuristic 2 (packages required to run the
# model's own generated code; released code mode 1) and heuristic 3 (mode 2, = h3_direct), scored with the paper rule
# against the 10-Jan-2024 list. Heuristic 2 needs the code-generation phase (generate_code.py) first.
_CODEGEN = Source(repo=str(REPO), path='generate_code.py', commit=COMMIT, url=URL,
                  locator='generate_code: messages = [{"role": "user", "content": system_message + prompt}]')
CODE_USER = PromptTemplate('code_user', _fstring_assigned(REPO / 'generate_code.py', 'generate_code', 'system_message')
                           + '{problem}', source=_CODEGEN,
                           note='released code sends system_message + prompt as ONE user message, no separator; '
                                'prompt = the LLM-generated prompt line as read (not stripped)')
_GEN1 = Source(repo=str(REPO), path='generate_package_names.py', commit=COMMIT, url=URL,
               locator='generate_packages, mode == 1')
H2_SYSTEM = PromptTemplate('h2_system', _fstring_in(REPO / 'generate_package_names.py', 'generate_packages',
                                                    'system_message', 1), source=_GEN1,
                           note='f-string body; {language} = "Python" (paper appendix adds a "respond None" clause)')
H2_USER = PromptTemplate('h2_user', _fstring_in(REPO / 'generate_package_names.py', 'generate_packages', 'prefix', 1)
                         + '{code}', source=_GEN1, note='prefix + generated code (row["Answers"]).strip()')
CODE_PARAMS = {'temperature': 0.7, 'top_p': 0.9, 'max_tokens': 2048}   # Table 9 code generation; generate_code defaults
PAPER_TABLE_TARGETS = {'openai-direct/gpt-3.5-turbo-0125': 0.0598, 'openai-direct/gpt-4-turbo-2024-04-09': 0.0329}
# GPT-3.5 Turbo LLM-prompt denominator: the cell is cut off ("41,7"); Total - Stack Overflow - pip install
# = 76,123 - 33,948 - 426 = 41,749 (2,495/41,749 = 5.976%); the earlier audit's 41,722 = 2,495/0.0598 rounded.
GPT35_LLM_DENOMINATOR = 76123 - 33948 - 426


# ----------------------------------------------------------------------------------------------- authors' scorer
@functools.lru_cache(maxsize=None)
def official():
    import pandas as pd
    fns = load_functions(REPO / 'package_detection.py', ['normalize_python', 'check_packages'],
                         extra_globals={'pd': pd})
    fns.update(load_functions(REPO / 'custom_parse_python.py', ['delete_dupes_and_empty'], extra_globals={'pd': pd}))
    return fns


@functools.lru_cache(maxsize=None)
def master_sets():
    """(valid names, false positives) exactly as package_search_python builds them (pandas read_csv, header=None)."""
    import pandas as pd
    normalize = official()['normalize_python']
    names = pd.read_csv(PYDATA / 'pypi_package_names.csv', header=None)
    names[0] = names[0].apply(normalize)
    fps = pd.read_csv(PYDATA / 'false_positive_packages.csv', header=None)
    return frozenset(names[0]), frozenset(fps[1])


@functools.lru_cache(maxsize=None)
def live_names():
    """Normalised (PEP 503) project names on PyPI on 2026-10-04."""
    with gzip.open(LIVE, 'rt') as f:
        return frozenset(re.sub(r'[-_.]+', '-', line.rstrip('\n')).lower() for line in f if line.strip())


def csv_roundtrip(text):
    """The response as detect_packages sees it: written with DataFrame.to_csv and read back with read_csv, then
    astype(str) (package_search_python). Empty strings and pandas NA tokens ('None', 'NULL', 'nan', ...) become 'nan'.

    Read with dtype=str: in the authors' whole-file read the column is mixed text, so pandas keeps numeric-looking cells
    as strings; a one-row read would otherwise turn '3' into 3.0. NaN -> 'nan' is pandas 2.2 (the authors' version)
    astype(str) behaviour, written out because pandas 3 keeps NaN under astype(str)."""
    import pandas as pd
    buf = io.StringIO()
    pd.DataFrame({'Test_2': [text]}).to_csv(buf, index=False)
    buf.seek(0)
    value = pd.read_csv(buf, dtype=str)['Test_2'].iloc[0]
    return 'nan' if pd.isna(value) else str(value)


def parse_packages(text):
    """package_search_python for one Test_2 cell (no custom pre/post parsing, as for the GPT models)."""
    fns = official()
    entries = [e for e in (fns['normalize_python'](x) for x in csv_roundtrip(text).split(',')) if len(e.split()) == 1]
    return fns['delete_dupes_and_empty'](entries)


def classify(text):
    valid_set, fps = master_sets()
    return official()['check_packages'](parse_packages(text), valid_set, fps)


def strip_think(text):
    """Drop leaked reasoning: <think>...</think> blocks, and everything after an unterminated <think>."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S | re.I)
    return re.split(r'<think>', text, flags=re.I)[0]


def name_shaped(entry):
    """Headline filter (not the paper's): strip markdown, quotes, brackets, list markers, extras and version pins
    from one recommended entry; return the PEP 503-normalised name if it is a valid PEP 508 project name, else None."""
    e = entry.strip()
    e = re.sub(r'^(?:[-*+\u2022]|\d+[.)])\s+', '', e)               # list markers
    e = e.replace('**', '').replace('__', '').strip(' `\'"*')
    e = re.split(r'\s+[-\u2013\u2014:(]|\s*\(|:\s', e)[0]                   # explanatory suffix: 'x (for y)', 'x - y', 'x: y'
    e = re.split(r'\s*(?:===|==|>=|<=|~=|!=|>|<|;|@)\s*', e)[0]     # version pins / markers / direct refs
    e = re.sub(r'(?<=[A-Za-z0-9])\[[^\]]*\]', '', e)                 # extras: package[extra]
    e = e.strip(' `\'"*[](){}<>.:;')
    e = e.lower()
    if not e or ' ' in e or not NAME_RX.match(e):
        return None
    return re.sub(r'[-_.]+', '-', e)


def headline_entries(text, truncated=False):
    """Recommended entries for the headline: <think> leakage removed, split on commas AND newlines (the paper splits
    on commas only, so newline lists yield nothing there), the last entry dropped when the output was truncated."""
    text = strip_think(text)
    parts = [p for p in re.split(r'[,\n]', text) if p.strip()]
    if truncated and parts:
        parts = parts[:-1]
    return parts


def classify_headline(text, truncated=False):
    """(valid, hallucinated vs the Jan-2024 list, artifacts) under the name-shaped rule; names deduplicated."""
    valid_set, fps = master_sets()
    valid, hallucinated, artifacts, seen = [], [], [], set()
    for raw in headline_entries(text, truncated):
        name = name_shaped(raw)
        if name is None:
            if raw.strip():
                artifacts.append(raw.strip())
            continue
        if name in seen or len(name) <= 2:      # delete_dupes_and_empty keeps names of 3+ characters
            continue
        seen.add(name)
        if name in valid_set:
            valid.append(name)
        elif name not in fps and raw.strip() not in fps:
            hallucinated.append(name)
    return valid, hallucinated, artifacts


def classify_robust(text, truncated=False):
    """parse='robust' (h3_direct): (valid, hallucinated, path). The paper rule (classify) first; only when it yields no
    package from a non-blank, non-'None' answer, the name-shaped extraction (classify_headline: <think> stripped,
    comma AND newline split, markdown/list markers/version pins removed, a truncated last entry dropped) judged by the
    same Jan-2024 list and false-positive rule. path: 'paper', 'robust', or None (unparsed: blank visible output, or
    no package-shaped entry at all)."""
    visible = _robust.normalize(text)
    if not visible:
        return [], [], None
    valid, hall = classify(text)
    if valid or hall:
        return valid, hall, 'paper'
    if _EXPLICIT_NONE.match(visible.strip().strip('"\'')):
        return [], [], 'paper'                 # the requested "None" answer: parsed, no packages
    h_valid, h_hall, _ = classify_headline(text, truncated)
    if h_valid or h_hall:
        return h_valid, h_hall, 'robust'
    return [], [], None


def paper_without_cut_entry(text):
    """Diagnostic for truncated outputs: the paper rule on the response without its last comma entry (the one the
    output cap may have cut mid-name, e.g. 'numpy, pandas, scikit-le')."""
    head = text.rsplit(',', 1)[0] if ',' in text else ''
    return classify(head)


def model_release_date(model):
    from ..models import release_date
    return release_date(model) or EXTRA_RELEASE_DATES.get(model)


@functools.lru_cache(maxsize=None)
def first_uploads():
    """Frozen {name: {'first_upload': ISO date or None, ...}} (empty until the post-run step has been run)."""
    if not FIRST_UPLOAD.exists():
        return {}
    return json.loads(FIRST_UPLOAD.read_text())['names']


def first_upload_from_json(meta):
    """Earliest upload time over all release files in a PyPI JSON API response (None if the project has no files)."""
    times = [f.get('upload_time_iso_8601') or f.get('upload_time')
             for files in (meta.get('releases') or {}).values() for f in files]
    times = [t for t in times if t]
    return min(times)[:10] if times else None


def candidate_names(run_dir, variant=None):
    """POST-RUN STEP 1. Names that need a first-upload date: hallucinated under the headline rule against the
    Jan-2024 list but present in the 2026-10-04 snapshot. Reads every response stream of a run directory."""
    from ..replay import run_records
    live = live_names()
    out = set()
    for rec in run_records(run_dir):
        if rec.get('status') not in {'ok', 'truncated', 'empty', 'complete'}:
            continue
        o = rec['output'] if 'output' in rec and 'trace' in rec else {'content': rec.get('content'),
                                                                     'finish_reason': rec.get('finish_reason')}
        text = (o or {}).get('content') or ''
        trunc = (o or {}).get('finish_reason') == 'length'
        out |= {n for n in classify_headline(text, trunc)[1] if n in live}
    return sorted(out)


def freeze_first_uploads(names, fetch=None, today=None):
    """POST-RUN STEP 2. Fetch first-upload dates for ``names`` not yet frozen from the PyPI JSON API
    (https://pypi.org/pypi/<name>/json; free public metadata, no API key) and merge them into FIRST_UPLOAD with
    provenance. ``fetch(name) -> (http_status, json or None)`` can be injected for tests. Returns the new entries."""
    import datetime
    import urllib.error
    import urllib.request
    if fetch is None:
        def fetch(name):
            try:
                with urllib.request.urlopen(f'https://pypi.org/pypi/{name}/json', timeout=30) as r:
                    return r.status, json.loads(r.read())
            except urllib.error.HTTPError as e:
                return e.code, None
    today = today or datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%MZ')
    data = json.loads(FIRST_UPLOAD.read_text()) if FIRST_UPLOAD.exists() else {
        'source': 'https://pypi.org/pypi/<name>/json: min upload_time_iso_8601 over all release files',
        'rule': 'name counts as real for a model iff first_upload < model release date; no files -> not real',
        'names': {}}
    new = {}
    for name in sorted(set(names) - set(data['names'])):
        status, meta = fetch(name)
        new[name] = {'first_upload': first_upload_from_json(meta) if meta else None, 'http_status': status,
                     'fetched': today}
    data['names'].update(new)
    FIRST_UPLOAD.parent.mkdir(parents=True, exist_ok=True)
    FIRST_UPLOAD.write_text(json.dumps(data, indent=1, sort_keys=True) + '\n')
    first_uploads.cache_clear()
    return new


class PaperTableProgram(Program):
    """Heuristics 1-3 of the paper for one LLM-generated prompt: code generation (step 'code'), then mode 1 on the
    generated code (step 'h2', paper heuristic 2) and mode 2 on the prompt (step 'h3', paper heuristic 3). The h3
    request is byte-identical to h3_direct's."""
    name = 'paper_table_h1_h2_h3'

    def __init__(self, variant):
        self.variant = variant
        self.language = variant.options['language']

    def code_messages(self, item):
        return [{'role': 'user', 'content': self.variant.prompts['code_user'].render(language=self.language,
                                                                                    problem=item['problem'])}]

    def h2_messages(self, code):
        return [{'role': 'system', 'content': self.variant.prompts['h2_system'].render(language=self.language)},
                {'role': 'user', 'content': self.variant.prompts['h2_user'].render(language=self.language,
                                                                                   code=code.strip())}]

    def h3_messages(self, item):
        return [{'role': 'system', 'content': self.variant.prompts['system'].render(language=self.language)},
                {'role': 'user', 'content': self.variant.prompts['user'].render(language=self.language,
                                                                                problem=item['problem'].strip())}]

    def preview(self, item):
        return self.code_messages(item)

    def forward(self, item, lm):
        code = lm(self.code_messages(item), self.variant.options['code_params'], step='code')
        h2 = lm(self.h2_messages(code.content), self.variant.params, step='h2')
        h3 = lm(self.h3_messages(item), self.variant.params, step='h3')
        return {'code': code.content, 'h2': h2.content, 'h3': h3.content, 'content': h3.content,
                'code_finish_reason': code.finish_reason, 'h2_finish_reason': h2.finish_reason,
                'h3_finish_reason': h3.finish_reason, 'finish_reason': h3.finish_reason,
                'completions': [code.summary(), h2.summary(), h3.summary()]}


@functools.lru_cache(maxsize=None)
def _pip_fns():
    import pandas as pd
    return load_functions(REPO / 'package_detection.py', ['parse_pip_install', 'normalize_pip', 'normalize_python',
                                                         'check_pips'], extra_globals={'pd': pd, 're': re})


def pip_packages(code_text):
    """Heuristic 1 (separate table column; diagnostic here): the authors' pip_numbers on the generated code.
    Their CSV round trip turns an empty/None answer into NaN, which parse_pip_install maps to []."""
    fns = _pip_fns()
    text = csv_roundtrip(code_text)
    if text == 'nan':
        return [], []
    pips = [fns['normalize_pip'](x) for x in fns['parse_pip_install'](text)]
    valid_set, _ = master_sets()
    valid, hall = fns['check_pips'](pips, valid_set)
    return valid, official()['delete_dupes_and_empty'](hall)


def _jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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


def _unsent(*runs):
    """excluded_runs entries for per-model directories that sent nothing: (run, space-separated models)."""
    return {f'{_R2}/{run}/{m}': _NOTHING_SENT for run, models in runs for m in models.split()}


_SWEEP = 'experiments/initial-reproduction/runs/runs/C104'


@register
class C104PackageHallucination(Experiment):
    claim_id = 'C104'
    claim_set_key = 'HL-N01'
    maturity = 'ready'
    title = 'Package hallucinations in code generation (Spracklen et al.)'
    claim = ('Asked which packages would help solve a coding task, LLMs recommend Python package names that do not '
             'exist on PyPI (GPT-3.5 Turbo 5.98% / GPT-4 Turbo 3.29% on LLM-generated prompts; 5.76% / 3.59% overall; '
             'open models 21.7%), more often for recent topics.')
    paper = {'title': 'We Have a Package for You! A Comprehensive Analysis of Package Hallucinations by Code '
                      'Generating LLMs', 'authors': 'Spracklen, Wijewickrama, Sakib, Maiti, Viswanath, Jadliwala',
             'first_date': '2024-06-12', 'venue': 'USENIX Security 2025', 'url': 'https://arxiv.org/abs/2406.10279'}
    source = _GEN
    default_variant = 'paper_table_llm'   # was h3_direct

    def build_variants(self):
        dev = ('top_k=20, typical_p=1, epsilon/eta cutoff 0 and repetition penalty 1 (Table 9, HF generate) have no '
               'OpenAI-compatible equivalent; the paper notes top-k is not modifiable through the OpenAI API',
               'the paper does not name the GPT snapshots (queried in early 2024)',
               'Python, LLM-generated prompts and heuristic 3 (code mode 2) only; the paper\'s LLM-prompt column '
               'pools heuristics 2 and 3 (code modes 1 and 2)',
               'reasoning models run under the panel policy (4,096-token cap instead of 64; OpenAI reasoning and '
               'Anthropic models drop temperature/top_p)',
               'headline (not faithful): name-shaped filter, comma+newline split, <think> stripped, truncation-safe, '
               'registry as of the model release date; the paper rule is reported as rate_paper',
               'released code, not the paper appendix, supplies the prompt wording (they differ in two words and a '
               'final period; see h3_direct_appendix)')
        base = Variant('h3_direct', 'Heuristic 3 / code mode 2: which Python packages would help solve the '
                                    'LLM-generated coding prompt (released-code wording)',
                       prompts={'system': SYSTEM, 'user': USER},
                       params={'temperature': 0.01, 'top_p': 0.9, 'max_tokens': 64},
                       strategy='single_turn', options={'language': 'Python', 'datasets': [d for d, _ in DATASETS]},
                       deviations=dev)
        table = Variant(
            'paper_table_llm', 'Paper Python results table, "LLM Generated Prompts" column: code generation, then '
                               'heuristic 2 (packages to run the generated code) + heuristic 3 (packages for the '
                               'prompt), pooled, paper rule vs the 10-Jan-2024 PyPI list',
            prompts={'code_user': CODE_USER, 'h2_system': H2_SYSTEM, 'h2_user': H2_USER, 'system': SYSTEM, 'user': USER},
            params={'temperature': 0.01, 'top_p': 0.9, 'max_tokens': 64}, strategy='code_then_h2_h3',
            options={'language': 'Python', 'datasets': [d for d, _ in DATASETS], 'code_params': dict(CODE_PARAMS),
                     'headline': 'paper_table'},
            deviations=(dev[0], dev[1],
                        'released code (not the paper appendix) supplies all wordings: code generation sends the '
                        'system text + prompt as one user message (generate_code.py); mode-1 system text lacks the '
                        'appendix "respond None" clause; the authors\' GPT-API driver was not released',
                        'subset of the 9,814 LLM-generated prompts (seeded LY/AT interleave), not all of them; Stack '
                        'Overflow prompts and the pip-install column (heuristic 1, reported as a diagnostic) are not '
                        'in this cell',
                        'reasoning models run under the panel policy (raised caps; dropped sampling params)',
                        'headline = paper rule (faithful); release-date rule and name-shaped filter are secondary'))
        # Robustness variants. robust_parse: h3_direct requests (the
        # panel variant), parse='robust', for offline rescoring. robust: for new panel runs.
        robust_parse = base.override(
            'robust_parse', description='h3_direct requests, parse=robust (paper rule first; name-shaped fallback when '
            'it finds no package; blank output unparsed)', options={'parse': 'robust'},
            deviations=('scoring: the paper rule (comma split of the CSV round trip) first; only when it yields no '
                        'package from a non-blank, non-"None" answer, the name-shaped extraction (comma and newline '
                        'split, markdown/list markers/pins removed, <think> stripped, a truncated last entry dropped) '
                        'judged by the same Jan-2024 list and false-positive rule; blank visible output is unparsed '
                        '(missing). Under parse=robust metrics["paper"] is this paper-first rate and '
                        'metrics["paper_rule_only"] the unmodified paper rule',
                        'reliability under parse=robust: <= 10% of prompts unparsed or missing (_robust.reliable)'))
        robust = robust_parse.override(
            'robust', description=f'robust_parse + max_tokens floor 1024 and min_cap_reasoning {ROBUST_MIN_CAP_REASONING}',
            params=_robust.cap_floor(base.params), options={'min_cap_reasoning': ROBUST_MIN_CAP_REASONING},
            deviations=('max_tokens 64 -> 1024 (_robust.cap_floor): at 64 tokens 15.5% of M049 lists were cut, and a '
                        'name cut mid-word ("scikit-le") counts as a hallucination under the paper rule; uncapped lists '
                        'can also be longer than the paper\'s',
                        f'reasoning-model cap 4096 -> {ROBUST_MIN_CAP_REASONING} (min_cap_reasoning): M144 returned blank '
                        'visible output on 18.5% of prompts'))
        return {'h3_direct': base, 'paper_table_llm': table,
                'robust_parse': robust_parse, 'robust': robust}

    # ------------------------------------------------------------------ data
    @functools.lru_cache(maxsize=None)
    def _all_items(self):
        per = {}
        for name, fname in DATASETS:
            rows = [{'id': f'C104:{name}:{i:04d}', 'dataset': name, 'recency': 'recent' if name == 'LLM_LY' else 'all_time',
                     'line': i, 'problem': text} for i, text in enumerate(_jsonl(PYDATA / fname))]
            random.Random(f'{SEED}:{name}').shuffle(rows)
            per[name] = rows
        out = []
        for a, b in zip(per['LLM_LY'], per['LLM_AT']):
            out += [a, b]
        n = min(len(v) for v in per.values())
        for name in ('LLM_LY', 'LLM_AT'):
            out += per[name][n:]
        return tuple(out)

    def items(self, variant, limit=None):
        rows = [dict(r) for r in self._all_items()]
        return rows[:limit] if limit else rows

    def program(self, variant):
        if variant.strategy == 'code_then_h2_h3':
            return PaperTableProgram(variant)
        language = variant.options['language']

        def build(item):
            return [{'role': 'system', 'content': variant.prompts['system'].render(language=language)},
                    {'role': 'user', 'content': variant.prompts['user'].render(language=language,
                                                                               problem=item['problem'].strip())}]
        mincap = variant.options.get('min_cap_reasoning')
        return SingleTurn(build, variant.params, case_extra={'min_cap_reasoning': mincap} if mincap else None)

    # --------------------------------------------------------------- scoring
    def score(self, item, output, variant):
        if variant.strategy == 'code_then_h2_h3':
            return self.score_table(item, output, variant)
        text = output.get('content') or ''
        valid, hallucinated = classify(text)            # paper rule (faithful), raw response
        live = live_names()
        truncated = output.get('finish_reason') == 'length' or output.get('status') == 'truncated'
        h_valid, h_hall, artifacts = classify_headline(text, truncated)
        return {'valid': sorted(valid), 'hallucinated': sorted(hallucinated),
                'confirmed': sorted(p for p in hallucinated if p not in live), 'n_packages': len(valid) + len(hallucinated),
                'truncated': truncated,
                # headline rule: name-shaped, <think> stripped, comma+newline split, truncation-safe
                'h_valid': sorted(h_valid), 'h_hallucinated': sorted(h_hall),
                'h_post2024': sorted(p for p in h_hall if p in live),        # need a first-upload date
                'h_confirmed': sorted(p for p in h_hall if p not in live),   # absent from PyPI even in 2026
                'h_n_packages': len(h_valid) + len(h_hall), 'artifacts': len(artifacts),
                **(self._robust_fields(text, truncated) if variant.options.get('parse', 'paper') == 'robust' else {})}

    @staticmethod
    def _robust_fields(text, truncated):
        r_valid, r_hall, path = classify_robust(text, truncated)
        out = {'r_valid': sorted(r_valid), 'r_hallucinated': sorted(r_hall), 'r_n_packages': len(r_valid) + len(r_hall),
               'parse_path': path, 'unparsed': path is None}
        if truncated:
            v, h = paper_without_cut_entry(text)
            out.update(cut_valid=sorted(v), cut_hallucinated=sorted(h))
        return out

    @staticmethod
    def _ratio(rows, num, den='n_packages'):
        n = sum(r['score'][den] if isinstance(den, str) else den(r['score']) for r in rows)
        k = sum(len(r['score'][num]) if isinstance(r['score'][num], list) else r['score'][num] for r in rows)
        return k / n if n else float('nan')

    def _rate(self, rows, num, den='n_packages', seed=0):
        point, lo, hi = stats.cluster_bootstrap(rows, lambda r: r['item']['id'],
                                                lambda rs: self._ratio(rs, num, den), n_boot=2000, seed=seed)
        n = sum(r['score'][den] if isinstance(den, str) else den(r['score']) for r in rows)
        k = sum(len(r['score'][num]) if isinstance(r['score'][num], list) else r['score'][num] for r in rows)
        return {'k': k, 'packages': n, 'rate': k / n if n else None, 'ci95': [lo, hi] if n else None}

    @staticmethod
    def _asof_counts(score, release, uploads):
        """Headline hallucinations as of ``release``: confirmed (never on PyPI) + post-2024 names first uploaded on or
        after the release date (or with no files). None if a post-2024 name has no frozen date yet."""
        k = len(score['h_confirmed'])
        for name in score['h_post2024']:
            if name not in uploads:
                return None
            first = uploads[name]['first_upload']
            k += first is None or first >= release
        return k

    def score_table(self, item, output, variant):
        h2 = self.score(item, {'content': output.get('h2') or '', 'finish_reason': output.get('h2_finish_reason')},
                        self.variant('h3_direct'))
        h3 = self.score(item, {'content': output.get('h3') or '', 'finish_reason': output.get('h3_finish_reason')},
                        self.variant('h3_direct'))
        pip_valid, pip_hall = pip_packages(output.get('code') or '')
        return {'h2': h2, 'h3': h3, 'pip_valid': pip_valid, 'pip_hallucinated': pip_hall,
                'code_truncated': output.get('code_finish_reason') == 'length',
                # pooled paper rule (the table cell): hallucinated_1 + hallucinated_2 over all four columns
                'hallucinated': h2['hallucinated'] + h3['hallucinated'],
                'n_packages': h2['n_packages'] + h3['n_packages']}

    def aggregate_table(self, rows, variant):
        scored = [r for r in rows if r.get('score') is not None]
        out = {'n_prompts': len(rows), 'scored': len(scored), 'failed_prompts': len(rows) - len(scored)}
        pooled = self._rate(scored, 'hallucinated')
        out['paper_table'] = pooled
        for h in ('h2', 'h3'):
            sub = [{**r, 'score': r['score'][h]} for r in scored]
            out[h] = {'paper': self._rate(sub, 'hallucinated'),
                      'release_date_rule': self.aggregate(sub, self.variant('h3_direct'))['headline_asof'],
                      'responses_without_packages': sum(r['score']['n_packages'] == 0 for r in sub),
                      'truncated': sum(r['score']['truncated'] for r in sub)}
        pip_k = sum(len(r['score']['pip_hallucinated']) for r in scored)
        pip_n = pip_k + sum(len(r['score']['pip_valid']) for r in scored)
        out['pip_install_h1'] = {'k': pip_k, 'packages': pip_n, 'rate': pip_k / pip_n if pip_n else None}
        out['code_truncated'] = sum(r['score']['code_truncated'] for r in scored)
        by = {}
        for name in ('recent', 'all_time'):
            sub = [r for r in scored if r['item']['recency'] == name]
            by[name] = {'prompts': len(sub), 'paper_table': self._rate(sub, 'hallucinated')['rate']}
        out['by_recency'] = by
        # secondary: release-date rule pooled over h2 + h3 (None until first-upload dates are frozen)
        asof = {h: out[h]['release_date_rule'] for h in ('h2', 'h3')}
        out['release_date_rule_pooled'] = (
            None if any(a is None for a in asof.values()) else
            {'k': sum(a['k'] for a in asof.values()), 'packages': sum(a['packages'] for a in asof.values()),
             'rate': (sum(a['k'] for a in asof.values()) / max(1, sum(a['packages'] for a in asof.values())))})
        models = {r.get('model') for r in rows} - {None}
        model = next(iter(models)) if len(models) == 1 else None
        no_pkg = sum(r['score']['n_packages'] == 0 for r in scored)
        out['headline_reliable'] = bool(scored) and no_pkg / len(scored) <= NO_PACKAGE_THRESHOLD
        out['shelflife'] = {
            'metric': 'package_hallucination_rate_paper_table_llm_prompts',
            'value': pooled['rate'], 'ci': pooled['ci95'], 'n': pooled['packages'], 'direction': +1,
            'null_value': 0.01, 'paper_value': PAPER_TABLE_TARGETS.get(model, 0.0598),
            'paper_value_note': ('Python results table, LLM Generated Prompts: GPT-3.5 Turbo 5.98% (2,495/41,749), '
                                 'GPT-4 Turbo 3.29% (1,518/46,204); paper_value is the GPT-4 Turbo cell for '
                                 'gpt-4-turbo-2024-04-09 and the GPT-3.5 Turbo cell otherwise'),
            'secondary_release_date_rule': out['release_date_rule_pooled'],
            'reliable': out['headline_reliable'],
            'definition': ('Paper rule: hallucinated / (valid + hallucinated) distinct names per response, pooled over '
                           'heuristic 2 (packages to run the model\'s generated code) and heuristic 3 (packages for '
                           'the prompt) on LLM-generated prompts; a name is hallucinated if absent from the '
                           '10-Jan-2024 PyPI list and not on the authors\' false-positive list. CI resamples prompts.')}
        return out

    def aggregate(self, rows, variant):
        if variant.strategy == 'code_then_h2_h3':
            return self.aggregate_table(rows, variant)
        scored = [r for r in rows if r.get('score') is not None]
        out = {'n_prompts': len(rows), 'scored': len(scored), 'failed_calls': len(rows) - len(scored)}
        # paper rule (faithful) and its 2026-snapshot variant, raw responses
        out['paper'] = self._rate(scored, 'hallucinated')
        out['confirmed'] = self._rate(scored, 'confirmed')
        out['rate_paper'] = out['paper']['rate']
        out['rate_confirmed'] = out['confirmed']['rate']
        # headline rule bounds: upper = Jan-2024 list only, lower = also absent in 2026
        out['headline_upper'] = self._rate(scored, 'h_hallucinated', 'h_n_packages')
        out['headline_lower'] = self._rate(scored, 'h_confirmed', 'h_n_packages')
        models = {r.get('model') for r in rows} - {None}
        model = next(iter(models)) if len(models) == 1 else None
        release = model_release_date(model) if model else None
        uploads = first_uploads()
        post = sorted({p for r in scored for p in r['score']['h_post2024']})
        unresolved = [p for p in post if p not in uploads]
        out['release_date'] = release
        out['n_post2024_names'] = len(post)
        out['post2024_unresolved'] = len(unresolved)
        asof = None
        if scored and release and not unresolved:
            local = [{**r, 'score': {**r['score'], 'h_asof': self._asof_counts(r['score'], release, uploads)}}
                     for r in scored]
            asof = self._rate(local, 'h_asof', 'h_n_packages')
        out['headline_asof'] = asof
        out['artifact_share'] = (sum(r['score']['artifacts'] for r in scored)
                                 / max(1, sum(r['score']['artifacts'] + r['score']['h_n_packages'] for r in scored))
                                 if scored else None)
        by = {}
        for name in ('recent', 'all_time'):
            sub = [r for r in scored if r['item']['recency'] == name]
            by[name] = {'prompts': len(sub), 'paper': self._rate(sub, 'hallucinated')['rate'],
                        'headline_upper': self._rate(sub, 'h_hallucinated', 'h_n_packages')['rate'],
                        'headline_lower': self._rate(sub, 'h_confirmed', 'h_n_packages')['rate']}
        out['by_recency'] = by
        no_pkg = sum(r['score']['h_n_packages'] == 0 for r in scored)
        out['responses_without_packages'] = no_pkg
        out['responses_without_packages_paper_rule'] = sum(r['score']['n_packages'] == 0 for r in scored)
        out['truncated_responses'] = sum(r['score']['truncated'] for r in scored)
        share = no_pkg / len(scored) if scored else None
        reliable = bool(scored) and share <= NO_PACKAGE_THRESHOLD
        out['headline_reliable'] = reliable
        counts = Counter(p for r in scored for p in r['score']['h_hallucinated'])
        out['top_headline_hallucinations'] = counts.most_common(10)
        up, lo = out['headline_upper'], out['headline_lower']
        if variant.options.get('parse', 'paper') == 'robust':
            self._robust_aggregate(out, rows, scored)
        out['shelflife'] = {
            'metric': 'package_hallucination_rate_asof_release',
            'value': asof['rate'] if asof else None, 'ci': asof['ci95'] if asof else None,
            'n': up['packages'], 'direction': +1, 'null_value': 0.01,
            'paper_value': None, 'paper_value_reported': PAPER_VALUE_REPORTED,
            'paper_value_note': ('0.0598 is GPT-3.5 Turbo on LLM-generated prompts, heuristics 2+3 pooled, paper rule; '
                                 'not on this metric\'s scale (heuristic 3 only, name-shaped filter, as-of-release '
                                 'registry). Anchor the verdict on the REPRO_MODELS values under this metric.'),
            'bounds': [lo['rate'], up['rate']],
            'no_package_share': share, 'reliable': reliable,
            'definition': ('Share of distinct name-shaped Python package names recommended per response (heuristic 3, '
                           'LLM-generated prompts; <think> stripped; comma/newline split; truncated last entry dropped) '
                           'that did not exist on PyPI at the model\'s release date: absent from the 10-Jan-2024 list '
                           '(minus the authors\' false positives) and either absent from the 2026-10-04 index or first '
                           'uploaded on/after the release date (frozen PyPI JSON dates). None until those dates are '
                           'frozen; bounds = [2026-index rule, Jan-2024-list rule]. CI resamples prompts.')}
        return out

    def _robust_aggregate(self, out, rows, scored):
        """parse='robust' (h3_direct): metrics['paper'] becomes the paper-first rate with the name-shaped fallback
        (unparsed prompts excluded); the unmodified paper rule moves to metrics['paper_rule_only']."""
        parsed = [r for r in scored if not r['score']['unparsed']]
        parse = {'parsed_paper': sum(r['score']['parse_path'] == 'paper' for r in scored),
                 'parsed_robust': sum(r['score']['parse_path'] == 'robust' for r in scored),
                 'unparsed': sum(bool(r['score']['unparsed']) for r in scored), 'missing': len(rows) - len(scored)}
        out['paper_rule_only'] = out['paper']
        out['paper'] = self._rate(parsed, 'r_hallucinated', 'r_n_packages')
        out['rate_paper'] = out['paper']['rate']
        out['parse'] = parse
        out.update({k: parse[k] for k in ('parsed_paper', 'parsed_robust', 'unparsed')})
        out['validity_rate'] = (parse['parsed_paper'] + parse['parsed_robust']) / len(rows) if rows else None
        out['headline_reliable'] = _robust.reliable(len(rows), parse['unparsed'], parse['missing'])
        # truncation diagnostic: the paper rule with each truncated response's last (possibly cut) entry removed
        cut = [{**r, 'score': {**r['score'], 'r_hallucinated': r['score']['cut_hallucinated'],
                               'r_n_packages': len(r['score']['cut_valid']) + len(r['score']['cut_hallucinated'])}}
               if r['score'].get('truncated') and r['score']['parse_path'] == 'paper' and 'cut_valid' in r['score']
               else r for r in parsed]
        out['paper_truncated_last_entry_dropped'] = self._ratio(cut, 'r_hallucinated', 'r_n_packages') if cut else None

    # ------------------------------------------------- targets and prior runs
    targets = (
        PaperTarget('llm_generated_python_prompt_count', 9814, 0, 'v3 Appendix (LLM_Prompts_List.tex): last item '
                    'numbered 9814)', 'n/a', 'h3_direct', 'released_data',
                    compute=lambda exp: len(exp.items(exp.variant())),
                    note='LLM_LY (4,892) + LLM_AT (4,922) lines in the released Data/Python files'),
        PaperTarget('gpt35_turbo_llm_prompts_rate', 0.0598, 0.0005, 'v3 Python results table, GPT-3.5 Turbo LLM Generated '
                    'Prompts 5.98% (2,495/41,7[49])', 'gpt-3.5-turbo (snapshot unnamed, early 2024)', 'paper_table_llm',
                    'live',
                    note='Paper-table protocol (heuristics 2+3 pooled, paper rule, Jan-2024 list). No per-prompt outputs were released; the live anchor is the reproduction pass '
                         'run on gpt-3.5-turbo-0125 (same-family substitute; the paper does not name the snapshot).'),
        PaperTarget('gpt4_turbo_llm_prompts_rate', 0.0329, 0.0005, 'v3 Python results table, GPT-4 Turbo LLM Generated '
                    'Prompts 3.29% (1,518/46,204)', 'gpt-4-turbo (snapshot unnamed)', 'paper_table_llm', 'live',
                    note='As above; stand-in snapshot gpt-4-turbo-2024-04-09.'),
        PaperTarget('gpt35_llm_prompts_denominator', 0.0598, 0.00005, 'v3 Python results table, GPT-3.5 Turbo: Total '
                    '76,123 - Stack Overflow 33,948 - pip 426 = 41,749 LLM-prompt packages; 2,495/41,749', 'n/a',
                    'paper_table_llm', 'released_data',
                    compute=lambda exp: round((4387 - 1868 - 24) / GPT35_LLM_DENOMINATOR, 4),
                    note='Paper arithmetic only: recovers the cut-off denominator ("41,7") from the row totals.'),
        PaperTarget('gpt4_turbo_recent_minus_all_time', 0.0389 - 0.0326, 0.02, 'v3 Fig. 4 data (Plots/Data/figure_4.csv): '
                    'GPT 4 Turbo last year 3.89% vs all-time 3.26%', 'gpt-4-turbo', 'h3_direct', 'live',
                    note='Pools heuristics and both prompt sources; underpowered at the minimal design.'),
    )
