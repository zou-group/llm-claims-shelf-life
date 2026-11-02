"""Structural checks applied to every registered experiment."""
import json

import pytest

from shelflife import registry
from shelflife.prompts import git_commit
from shelflife.replay import resolve

CLAIMS = registry.list_experiments()


def test_all_modules_import():
    assert not registry.IMPORT_ERRORS, registry.IMPORT_ERRORS


@pytest.fixture(params=CLAIMS)
def exp(request):
    e = registry.get(request.param)
    registry.skip_if_draft(e)
    return e


# Configs whose paper protocol cannot be reproduced faithfully, so every variant is a labelled reconstruction
# (reasons are in each config's deviations).
NO_FAITHFUL_VARIANT = {'C083': 'original PCT text and scoring matrix need archived validation'}

# Paper-original variants that raise on purpose because the original data was never released.
BLOCKED_BY_DESIGN = {('C013', 'paper_original_blocked'), ('C051', 'paper_original')}


def test_metadata(exp):
    assert exp.title and exp.claim and exp.paper.get('url') and exp.source is not None
    assert exp.default_variant in exp.variants
    if exp.claim_id not in NO_FAITHFUL_VARIANT:
        assert any(v.faithful for v in exp.variants.values()), 'needs at least one paper-faithful variant'


def test_variants_are_serializable_and_distinct(exp):
    fps = {v.fingerprint for v in exp.variants.values()}
    assert len(fps) == len(exp.variants)
    json.dumps(exp.manifest(), default=str)


def test_prompt_templates_match_sources(exp):
    for v in exp.variants.values():
        for t in v.prompts.values():
            assert t.verify_source() in (True, None), f'{v.name}/{t.name} drifted from {t.source.describe()}'
            if v.faithful:
                assert t.source is not None, f'faithful variant {v.name} has a prompt without provenance'


def test_source_commit(exp):
    if exp.source.commit and exp.source.repo:
        if not (resolve(exp.source.repo) / '.git').exists():
            pytest.skip('official repository is not checked out as a git clone here')
        assert git_commit(resolve(exp.source.repo)) == exp.source.commit


def test_items_and_previews_deterministic(exp):
    for v in exp.variants.values():
        if (exp.claim_id, v.name) in BLOCKED_BY_DESIGN:
            continue
        items = exp.items(v, limit=3)
        assert items and len({x['id'] for x in items}) == len(items)
        again = exp.items(v, limit=3)
        assert [x['id'] for x in items] == [x['id'] for x in again]
        prog = exp.program(v)
        for x, y in zip(items, again):
            assert prog.preview(x) == prog.preview(y)


def test_targets_well_formed(exp):
    for t in exp.targets:
        assert t.where and t.variant in exp.variants
