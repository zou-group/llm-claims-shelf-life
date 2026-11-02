"""`shelflife new CNNN`: scaffold a claim module, test module and findings file, pre-filled from the claim records.

Fills paper metadata and the claim statement from an earlier reproduction record or a claim record,
locates (or clones) the official repository and its commit, and pre-lists saved runs in an explicit JSON manifest.
The module starts as maturity='draft' so the shared test suite stays green while you work;
`shelflife check CNNN` shows what is left.
"""
import datetime
import json
import re
import subprocess
from pathlib import Path

from .paths import CODE, DATA, EXPERIMENTS, INITIAL, PROJECT

CLAIMS_100 = EXPERIMENTS / 'claim-records/claims.json'
EARLIER_RECORDS = EXPERIMENTS / 'earlier/selection/claim_records'
FINDINGS = CODE / 'findings'


def claim_record(claim_id, claim_json=None):
    """Explicit record, earlier reproduction record, earlier pipeline record, or the claim record set."""
    if claim_json:
        return json.loads(Path(claim_json).read_text())
    hits = sorted((INITIAL / 'claims').glob(f'{claim_id}_*/claim.json'))
    if hits:
        return json.loads(hits[0].read_text())
    if (EARLIER_RECORDS / f'{claim_id}.json').exists():
        return json.loads((EARLIER_RECORDS / f'{claim_id}.json').read_text())
    if CLAIMS_100.exists():
        for rec in json.loads(CLAIMS_100.read_text()):
            if rec.get('final_id') == claim_id:
                return rec
    return {}


def find_repo(claim_id):
    hits = sorted(p for p in (INITIAL / 'claims').glob(f'{claim_id}_*/code/*') if p.is_dir())
    hits += sorted(p for p in DATA.glob(f'sources/{claim_id}_*') if (p / '.git').exists())
    return hits[0] if hits else None


def git_head(repo):
    out = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    return out.stdout.strip() or None


def clone(claim_id, url, commit=None):
    name = url.rstrip('/').split('/')[-1].removesuffix('.git')
    dest = DATA / 'sources' / f'{claim_id}_{name}'
    if dest.exists():
        raise FileExistsError(f'{dest} exists')
    subprocess.run(['git', 'clone', '--quiet', url, str(dest)], check=True)
    if commit:
        subprocess.run(['git', '-C', str(dest), 'checkout', '--quiet', commit], check=True)
    head = git_head(dest)
    (dest / 'PROVENANCE.md').write_text(
        f'Cloned from {url} at commit {head} on {datetime.date.today().isoformat()} by `shelflife new {claim_id}`.\n'
        'Official repository for the claim; read-only source for prompts, data and scorers.\n')
    return dest


def top_level_runs(claim_id):
    from .run_manifest import claim_run_dirs
    found = claim_run_dirs(claim_id)
    return [f for f in found if not any(f != g and f.startswith(g + '/') for g in found)]


def _slugify(text, max_words=4):
    words = [w for w in re.split(r'[^a-z0-9]+', text.lower()) if w]
    return '_'.join(words[:max_words]) or 'claim'


def _q(text):
    return repr(text or '')


def _text(value, default=''):
    """Extract prose without accidentally turning an entire structured record into a title."""
    if isinstance(value, str):
        return value or default
    if isinstance(value, dict):
        for key in ('statement', 'text', 'claim', 'title', 'label'):
            result = _text(value.get(key))
            if result:
                return result
    return default


DOCSTRING = '''{cid} — {title}

Claim: {claim}

Paper: {paper_title} ({paper_url}); location: {location}.
Reported result: {main_result}
Sources: {repo_desc}

TODO (remove each line when done; `python -m shelflife.cli check {cid}` lists what is left):
  - variants: the paper's protocol(s) with verbatim prompts (PromptTemplate.from_source) and exact params
  - items(): load the official release with stable ids (reuse earlier case ids)
  - score()/aggregate(): the original scorer and the paper's metric/denominator
  - denominator: state how refusal, parse failure, truncation and failed/missing calls count
  - paired unit: identify the base item/cluster and keep treatment/control pairs together
  - targets: paper numbers, preferably checked on released outputs
  - prior_runs: assign each listed run its variant (or move it to excluded_runs with a reason)
See code/docs/INTEGRATING.md for the five-method contract and runnable examples.
'''

MODULE = '''{docstring}
from pathlib import Path

from ..paths import PROJECT
from ..program import Program, SingleTurn
from ..prompts import PromptTemplate, Source
from ..registry import Experiment, PaperTarget, Variant, register
from ..results import Headline
from ..sourcecode import load_functions
from .. import stats

REPO = {repo_expr}
COMMIT = {commit!r}
SRC = Source(repo=None if REPO is None else str(REPO), path='TODO/file.py', locator='TODO', commit=COMMIT,
             url={repo_url!r})

# TODO: verbatim prompt, e.g. PromptTemplate.from_source('user', SRC, kind='py_var', var='PROMPT')
USER = PromptTemplate('user', 'TODO {{question}}', source=SRC)


@register
class {cls}(Experiment):
    claim_id = {cid!r}
    maturity = 'draft'          # set to 'ready' when `shelflife check {cid}` passes and the audit is resolved
    title = {title_q}
    claim = {claim_q}
    paper = {paper!r}
    source = SRC
    default_variant = 'paper'
    run_manifest = {run_manifest!r}  # review paths, variants and exclusions in this JSON file

    def build_variants(self):
        paper = Variant('paper', 'TODO: headline protocol (section/table)', prompts={{'user': USER}},
                        params={{'temperature': 0, 'max_tokens': 256}}, repeats=1,
                        deviations=('TODO: list every departure from the paper, or remove this line',))
        return {{'paper': paper}}

    def items(self, variant, limit=None):
        rows = []   # TODO: [{{'id': ..., 'question': ..., 'answer': ...}}, ...] from the official release
        return rows[:limit] if limit is not None else rows

    def program(self, variant):
        return SingleTurn(lambda item: [{{'role': 'user', 'content': variant.prompts['user'].render(question=item['question'])}}],
                          variant.params)

    def score(self, item, output, variant):
        text = (output.get('content') or '').strip()
        return {{'correct': text == item['answer']}}   # TODO: the original scorer

    def aggregate(self, rows, variant):
        # TODO: this example uses a scored-row denominator; choose the paper's denominator explicitly.
        # Rows with score=None must stay visible in n/unscored/bounds. Pair/cluster by base item if required.
        result = stats.rate_summary(rows, lambda r: r['score']['correct'])
        result['shelflife'] = Headline(
            metric='accuracy', value=result['rate'], n=result['scored'], direction=1, unit='fraction',
            denominator='TODO: scored item/repeat rows; replace with paper inclusion rule',
            ci=result['ci95'] if result['rate'] is not None else None,
            missing=result['unscored'], missingness_note='Unscored rows retained in bounds_with_missing',
        ).to_dict()  # TODO: metric, direction, anchors and reliability must match the claim
        return result

    targets = (
        # PaperTarget('name', value, tolerance, 'paper vN Table X', 'model', 'paper', 'archived_outputs',
        #             compute=lambda exp: ..., note='why this tolerance'),
    )
'''

TEST = '''"""{cid}: prompts and scorer against the authors' code; earlier runs reproduce. See the integrate-experiment skill."""
import os

import pytest

from shelflife import registry

exp = registry.get({cid!r})
# While the claim is a draft these tests run only under `python -m shelflife.cli check {cid}`, so the shared suite
# stays green for everyone else. Delete nothing here when promoting: just set maturity = 'ready' in the module.
pytestmark = pytest.mark.skipif(exp.maturity == 'draft' and os.environ.get('SHELFLIFE_INCLUDE_DRAFTS') != '1',
                                reason='draft claim: run `python -m shelflife.cli check {cid}`')


def test_items_load():
    items = exp.items(exp.variant())
    assert items, 'TODO: items() returns nothing yet'


def test_prompts_byte_equal_to_original_code():
    pytest.fail('TODO: compare exp.program(variant).preview(item) with the authors\\' prompt builder for every item')


def test_scorer_matches_original():
    pytest.fail('TODO: compare score() with the original scorer on saved outputs and edge cases '
                '(empty, markdown, refusal, verbose reasoning output, out-of-range)')


@pytest.mark.parametrize('text', [
    '', '**B**', 'After checking each option, my final answer is B.',
    'I cannot answer this question.', 'Z',
])
def test_modern_output_fixtures(text):
    # Replace each case with (item, output, expected score) justified by the paper/released scorer.
    # Keep paper scoring and any more permissive diagnostic separately labelled.
    pytest.fail(f'TODO: provide an independent expected score and parse/refusal status for {{text!r}}')


def test_failed_calls_and_paired_denominator():
    # Use literal rows: one correct, one wrong, one failed call (output=None, score=None),
    # plus an incomplete treatment/control pair if paired. Assert literal counts and the paper metric.
    # Never filter failed rows before aggregate() or compute the expected answer with aggregate() itself.
    pytest.fail('TODO: assert paper denominator, missing count/bounds, and base-item pairing on explicit fixtures')


def test_prior_runs_rescore_to_reported_numbers():
    pytest.fail('TODO: exp.rescore(run_dir, variant) reproduces the numbers in each earlier run\\'s results/report '
                '(cite file and value in the docstring), or remove if there are no prior runs')
'''

FINDINGS_TMPL = '''# {cid} — {title}

## Sources
TODO: repo + commit, paper version, prompt sources (code vs PDF), released outputs found or not.

## Variants registered
TODO: name → paper section/table → prior runs using it.

## Metric contract
TODO: numerator, denominator, paired unit / bootstrap cluster, missing-call rule, refusal/parse-failure/truncation
rules, units and direction of the headline; identify diagnostic metrics separately.

## Paper reproduction
| Target | Ours | Paper | Check | Verdict |
|---|---|---|---|---|

## Prior implementation audit
TODO: per prior run: requests match? scorer matches original? rescoring reproduces reported numbers? Bugs vs deliberate
deviations, with quantified impact.

## Open issues
'''


def scaffold(claim_id, slug=None, repo=None, clone_url=None, commit=None, claim_json=None, force=False):
    if not re.fullmatch(r'C\d{3}', claim_id):
        raise ValueError('claim id must look like C123')
    rec = claim_record(claim_id, claim_json)
    if not isinstance(rec, dict):
        raise ValueError('claim record must be a JSON object')
    selected = rec.get('claim_set_record') or {}
    selected = selected if isinstance(selected, dict) else {}
    paper = rec.get('paper') or {}
    paper = dict(paper) if isinstance(paper, dict) else {'title': _text(paper)}
    paper.setdefault('title', _text(selected.get('paper'), 'TODO title'))
    paper.setdefault('url', _text(selected.get('url'), 'TODO'))
    exp_info = rec.get('experiment') or {}
    exp_info = exp_info if isinstance(exp_info, dict) else {}
    repro = rec.get('reproducibility', {}) if isinstance(rec.get('reproducibility'), dict) else {}
    title = _text(paper.get('title'), _text(selected.get('label'), 'TODO title'))
    paper['title'] = title
    claim = _text(rec.get('claim'), _text(selected.get('claim'), 'TODO claim statement'))
    bench = re.split(r'[,;(:]', _text(exp_info.get('task_or_benchmark')))[0].strip()
    slug = slug or _slugify(bench or _text(rec.get('topic')) or title)
    if not re.fullmatch(r'[a-z0-9][a-z0-9_]*', slug):
        raise ValueError('slug must contain only lowercase letters, digits and underscores')
    if clone_url == 'auto':
        clone_url = repro.get('repo_url')
        if not clone_url:
            raise ValueError('no repo_url in the claim record; pass --clone URL')
    repo_path = Path(repo).resolve() if repo else (clone(claim_id, clone_url, commit) if clone_url else find_repo(claim_id))
    head = git_head(repo_path) if repo_path else None
    if repo_path is None:
        repo_expr = 'None  # TODO: official repo path'
    elif repo_path.is_relative_to(PROJECT):
        repo_expr = f"PROJECT / {str(repo_path.relative_to(PROJECT))!r}"
    else:
        repo_expr = f"Path({str(repo_path)!r})"

    runs = top_level_runs(claim_id)
    manifest_path = CODE / 'run-manifests' / f'{claim_id}.json'
    run_manifest = {
        'version': 1, 'claim_id': claim_id,
        'prior_runs': [{'path': r, 'variant': 'paper', 'description': 'TODO: describe; set the matching variant'}
                       for r in runs],
        'excluded_runs': {},
    }
    fields = dict(
        cid=claim_id, title=title, title_q=_q(title), claim=claim,
        claim_q=_q(claim),
        paper_title=title, paper_url=paper.get('url', 'TODO'), location=exp_info.get('location_in_paper', 'TODO'),
        main_result=exp_info.get('main_result', 'TODO'),
        repo_desc=(f'{repo_path.relative_to(PROJECT) if repo_path.is_relative_to(PROJECT) else repo_path} @ {head}'
                   if repo_path else 'TODO: official repository'),
        paper={k: paper[k] for k in ('title', 'authors', 'first_date', 'venue', 'url') if k in paper} or {'url': 'TODO'},
        repo_expr=repo_expr, commit=head, repo_url=repro.get('repo_url') or clone_url,
        cls=claim_id + ''.join(w.capitalize() for w in slug.split('_')),
        run_manifest=str(manifest_path.relative_to(PROJECT)))
    prose = DOCSTRING.format(**fields).replace('\\', '\\\\').replace('"""', '\\"""')
    fields['docstring'] = '"""' + prose + '"""'
    targets = {
        CODE / 'shelflife/experiments' / f'{claim_id.lower()}_{slug}.py': MODULE.format(**fields),
        CODE / 'tests/claims' / f'test_{claim_id.lower()}.py': TEST.format(**fields),
        FINDINGS / f'{claim_id}.md': FINDINGS_TMPL.format(**fields),
        manifest_path: json.dumps(run_manifest, indent=2) + '\n',
    }
    existing = [p for p in targets if p.exists()] + sorted((CODE / 'shelflife/experiments').glob(f'{claim_id.lower()}_*.py'))
    if existing and not force:
        raise FileExistsError(f'already exists: {[str(p) for p in existing]}')
    for path, text in targets.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return {'files': [str(p.relative_to(PROJECT)) for p in targets], 'repo': str(repo_path) if repo_path else None,
            'commit': head, 'prior_runs': runs, 'claim_record_found': bool(rec)}
