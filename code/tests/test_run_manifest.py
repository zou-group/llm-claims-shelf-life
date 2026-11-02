"""Run metadata is explicit, validated and separate from experiment source."""
import json

import pytest

from shelflife.registry import Experiment, PriorRun, Variant
from shelflife.run_manifest import build_manifest, load_manifest, merged_runs, manifest_saved_run_dirs


class Claim(Experiment):
    claim_id = 'C999'
    default_variant = 'paper'

    def build_variants(self):
        return {'paper': Variant('paper', 'paper protocol'), 'pilot': Variant('pilot', 'pilot protocol')}


def save_manifest(root, data=None):
    path = root / 'runs.json'
    path.write_text(json.dumps(data if data is not None else document()))
    return path


def document(**updates):
    return {'version': 1, 'claim_id': 'C999', 'prior_runs': [
        {'path': 'experiments/sample/runs/C999/M001', 'variant': 'paper',
         'known_deviations': ['documented prompt difference']}],
        'excluded_runs': {'experiments/fixtures/C999': 'Offline fixture, not a research run'}, **updates}


def configured(root, relative, *, claim='C999', variant='paper', models=None, sent=True):
    run = root / relative
    run.mkdir(parents=True)
    (run / 'configuration.json').write_text(json.dumps({
        'claim_id': claim, 'variant': variant, 'models': ['M001'] if models is None else models}))
    if sent:
        (run / 'raw').mkdir()
        # Deliberately not JSON: the indexer must not read request/response bodies.
        (run / 'raw' / 'call.json').write_text('private response data')
    return run


def test_load_and_merge_with_legacy(tmp_path):
    exp = Claim()
    exp.run_manifest = 'runs.json'
    exp.prior_runs = (PriorRun('experiments/old/C999', 'paper'),)
    exp.excluded_runs = {'experiments/old-fixture/C999': 'Legacy synthetic fixture'}
    save_manifest(tmp_path)
    runs, excluded = merged_runs(exp, project=tmp_path)
    assert len(runs) == 2 and all(isinstance(r, PriorRun) for r in runs)
    assert runs[-1].known_deviations == ('documented prompt difference',)
    assert len(excluded) == 2
    assert len(exp.prior_runs) == 1 and len(exp.excluded_runs) == 1


def test_accessors_no_implicit_discovery(tmp_path, monkeypatch):
    exp = Claim()
    exp.prior_runs = (PriorRun('experiments/old/C999/', 'paper'),)
    # Existing files have no effect unless explicitly declared.
    save_manifest(tmp_path)
    assert exp.all_prior_runs() == exp.prior_runs
    assert exp.all_excluded_runs() == {}
    from shelflife import run_manifest
    original = run_manifest.merged_runs
    monkeypatch.setattr(run_manifest, 'merged_runs', lambda obj: original(obj, project=tmp_path))
    exp.run_manifest = 'runs.json'
    assert len(exp.all_prior_runs()) == 2
    assert len(exp.all_excluded_runs()) == 1
    before = exp.manifest()
    save_manifest(tmp_path, document(prior_runs=[]))
    assert len(exp.all_prior_runs()) == 1  # no stale cache
    assert exp.manifest() == before  # execution source snapshot does not expand


def test_missing_declared_manifest_fails(tmp_path):
    exp = Claim()
    exp.run_manifest = 'missing.json'
    with pytest.raises(FileNotFoundError):
        merged_runs(exp, project=tmp_path)


@pytest.mark.parametrize('update,match', [
    ({'version': 2}, 'version'), ({'version': True}, 'version'),
    ({'claim_id': 'C001'}, 'claim_id'), ({'typo': []}, 'unknown'),
    ({'prior_runs': {}}, 'list'), ({'excluded_runs': []}, 'object'),
    ({'excluded_runs': {'experiments/a': ''}}, 'reason'),
])
def test_rejects_bad_document(tmp_path, update, match):
    save_manifest(tmp_path, document(**update))
    with pytest.raises(ValueError, match=match):
        load_manifest(Claim(), 'runs.json', project=tmp_path)


@pytest.mark.parametrize('entry,match', [
    ({'variant': 'absent'}, 'unknown variant'),
    ({'path': '/tmp/run'}, 'project-relative'),
    ({'path': '../run'}, 'project-relative'),
    ({'path': 'a/../run'}, 'project-relative'),
    ({'path': 'a//run'}, 'project-relative'),
    ({'path': 'a/run/'}, 'project-relative'),
    ({'path': '.'}, 'escapes'),
    ({'path': 'a\\run'}, 'project-relative'),
    ({'cases': '/tmp/cases.jsonl'}, 'project-relative'),
    ({'no_requests': 'false'}, 'boolean'),
    ({'known_deviations': 'unspecified'}, 'list'),
    ({'known_deviations': [1]}, 'list'),
    ({'description': []}, 'string'),
    ({'unknown': True}, 'unknown fields'),
])
def test_rejects_bad_entry(tmp_path, entry, match):
    save_manifest(tmp_path, document(prior_runs=[{'path': 'experiments/run', 'variant': 'paper', **entry}]))
    with pytest.raises(ValueError, match=match):
        load_manifest(Claim(), 'runs.json', project=tmp_path)


def test_rejects_symlink_escape(tmp_path):
    (tmp_path / 'outside').symlink_to(tmp_path.parent, target_is_directory=True)
    save_manifest(tmp_path, document(prior_runs=[{'path': 'outside/run', 'variant': 'paper'}]))
    with pytest.raises(ValueError, match='escapes'):
        load_manifest(Claim(), 'runs.json', project=tmp_path)


def test_duplicate_and_conflicting_entries(tmp_path):
    exp = Claim()
    exp.run_manifest = 'runs.json'
    run = {'path': 'experiments/run', 'variant': 'paper'}
    save_manifest(tmp_path, document(prior_runs=[run, run]))
    with pytest.raises(ValueError, match='duplicate'):
        merged_runs(exp, project=tmp_path)
    save_manifest(tmp_path, document(prior_runs=[run]))
    exp.prior_runs = (PriorRun(**run),)
    assert len(merged_runs(exp, project=tmp_path)[0]) == 1
    exp.prior_runs = (PriorRun('experiments/run', 'pilot'),)
    with pytest.raises(ValueError, match='conflicting registration'):
        merged_runs(exp, project=tmp_path)


@pytest.mark.parametrize('excluded', ['experiments/run', 'experiments', 'experiments/run/child'])
def test_rejects_registered_excluded_overlap(tmp_path, excluded):
    exp = Claim()
    exp.run_manifest = 'runs.json'
    exp.excluded_runs = {excluded: 'Deliberate offline fixture'}
    save_manifest(tmp_path, document(prior_runs=[{'path': 'experiments/run', 'variant': 'paper'}]))
    with pytest.raises(ValueError, match='registered and excluded'):
        merged_runs(exp, project=tmp_path)


def test_rejects_conflicting_exclusion(tmp_path):
    exp = Claim()
    exp.run_manifest = 'runs.json'
    exp.excluded_runs = {'experiments/fixtures/C999': 'Conflicting legacy explanation'}
    save_manifest(tmp_path)
    with pytest.raises(ValueError, match='conflicting exclusion'):
        merged_runs(exp, project=tmp_path)


def test_rejects_parent_child_variant_conflict(tmp_path):
    save_manifest(tmp_path, document(prior_runs=[
        {'path': 'experiments/run', 'variant': 'paper'},
        {'path': 'experiments/run/M001', 'variant': 'pilot'}]))
    with pytest.raises(ValueError, match='different variants'):
        load_manifest(Claim(), 'runs.json', project=tmp_path)


def test_rejects_conflicting_legacy_entries_when_opted_in(tmp_path):
    exp = Claim()
    exp.run_manifest = 'runs.json'
    exp.prior_runs = (PriorRun('experiments/run', 'paper'), PriorRun('experiments/run', 'pilot'))
    save_manifest(tmp_path, document(prior_runs=[]))
    with pytest.raises(ValueError, match='conflicting legacy'):
        merged_runs(exp, project=tmp_path)


def test_builder_per_model_dirs_and_unsent_runs(tmp_path):
    configured(tmp_path, 'experiments/sample/C999/M001')
    run = configured(tmp_path, 'experiments/sample/C999/M002', models=['M002'], sent=False)
    # Runner can write failure rows even if the budget stops every request.
    (run / 'responses.jsonl').write_text('{"status":"not_submitted"}\n')
    configured(tmp_path, 'experiments/sample/C001/M001', claim='C001')
    configured(tmp_path, 'experiments/elsewhere/C999/M003')
    configured(tmp_path, 'experiments/sample/code_snapshot/C999/M004')
    data = build_manifest(Claim(), ['experiments/sample'], project=tmp_path)
    assert [p['path'] for p in data['prior_runs']] == [
        'experiments/sample/C999/M001', 'experiments/sample/C999/M002']
    assert [p['no_requests'] for p in data['prior_runs']] == [False, False]
    assert 'no raw files found' in data['prior_runs'][1]['description']
    assert 'M002' in data['prior_runs'][1]['description']
    save_manifest(tmp_path, data)
    assert len(load_manifest(Claim(), 'runs.json', project=tmp_path)[0]) == 2
    assert data == build_manifest(Claim(), ['experiments/sample', 'experiments/sample/C999'], project=tmp_path)


def test_builder_respects_legacy_and_exclusions(tmp_path):
    exp = Claim()
    exp.prior_runs = (PriorRun('experiments/sample/C999/M001', 'paper'),)
    exp.excluded_runs = {'experiments/sample/fixture': 'Synthetic fixture for testing'}
    configured(tmp_path, 'experiments/sample/C999/M001')
    configured(tmp_path, 'experiments/sample/fixture/C999/M001')
    assert build_manifest(exp, ['experiments/sample'], project=tmp_path)['prior_runs'] == []
    exp.prior_runs = (PriorRun('experiments/sample/C999/M001', 'pilot'),)
    with pytest.raises(ValueError, match='conflicts with registered'):
        build_manifest(exp, ['experiments/sample'], project=tmp_path)


def test_builder_fails_bad_same_claim_config(tmp_path):
    configured(tmp_path, 'experiments/sample/C999/M001', variant='unknown')
    with pytest.raises(ValueError, match='unknown variant'):
        build_manifest(Claim(), ['experiments/sample'], project=tmp_path)


def test_explicit_coverage_needs_no_claim_id_in_path(tmp_path):
    exp = Claim()
    exp.run_manifest = 'runs.json'
    configured(tmp_path, 'experiments/sample/run/M001')
    configured(tmp_path, 'experiments/sample/unregistered/M002')
    configured(tmp_path, 'experiments/sample/run/code_snapshot/copied')
    save_manifest(tmp_path, document(prior_runs=[{'path': 'experiments/sample/run', 'variant': 'paper'}]))
    assert manifest_saved_run_dirs(exp, project=tmp_path) == ['experiments/sample/run/M001']
    exp.run_manifest = None
    assert manifest_saved_run_dirs(exp, project=tmp_path) == []


def test_builder_rejects_config_symlink_outside_supplied_root(tmp_path):
    outside = configured(tmp_path, 'experiments/outside/run')
    root = tmp_path / 'experiments/sample'
    root.mkdir()
    (root / 'configuration.json').symlink_to(outside / 'configuration.json')
    with pytest.raises(ValueError, match='supplied run root'):
        build_manifest(Claim(), ['experiments/sample'], project=tmp_path)


def test_builder_preserves_existing_annotations_and_exclusions(tmp_path):
    exp = Claim()
    exp.run_manifest = 'runs.json'
    configured(tmp_path, 'experiments/sample/old')
    configured(tmp_path, 'experiments/sample/new')
    configured(tmp_path, 'experiments/sample/excluded')
    original = document(prior_runs=[{
        'path': 'experiments/sample/old', 'variant': 'paper', 'description': 'Independently checked',
        'cases': 'experiments/old/cases.jsonl', 'known_deviations': ['Documented prompt difference'],
        'no_requests': True,
    }], excluded_runs={'experiments/sample/excluded': 'Offline replay fixture, not paid output'})
    path = save_manifest(tmp_path, original)
    before = path.read_text()
    data = build_manifest(exp, ['experiments/sample'], project=tmp_path)
    assert len(data['prior_runs']) == 2
    assert data['prior_runs'][0] == original['prior_runs'][0]
    assert data['prior_runs'][1]['path'] == 'experiments/sample/new'
    assert data['excluded_runs'] == original['excluded_runs']
    assert path.read_text() == before


def test_builder_missing_declared_manifest_fails(tmp_path):
    exp = Claim()
    exp.run_manifest = 'missing.json'
    configured(tmp_path, 'experiments/sample/run')
    with pytest.raises(FileNotFoundError):
        build_manifest(exp, ['experiments/sample'], project=tmp_path)


@pytest.mark.parametrize('roots,error', [([], ValueError), (['.'], ValueError),
                                        (['../elsewhere'], ValueError), (['missing'], FileNotFoundError)])
def test_builder_requires_explicit_contained_existing_roots(tmp_path, roots, error):
    with pytest.raises(error):
        build_manifest(Claim(), roots, project=tmp_path)
