"""Paper numbers recomputed offline from released outputs or saved original-model runs."""
import pytest

from shelflife import registry

TARGETS = [(cid, t) for cid in registry.list_experiments() for t in registry.get(cid).targets]


@pytest.mark.parametrize('cid,target', TARGETS, ids=[f'{c}:{t.name}' for c, t in TARGETS])
def test_paper_target(cid, target):
    registry.skip_if_draft(registry.get(cid))
    if target.check in {'live', 'unverifiable'}:
        pytest.skip(f'{target.check}: {target.note or ""}')
    if target.check == 'prior_run':
        pytest.skip('prior_run: recomputed from our saved model outputs, which are not included')
    value = target.compute(registry.get(cid))
    within = abs(value - target.value) <= target.tolerance
    if target.known_discrepancy:
        assert not within, f'documented discrepancy no longer holds (got {value}); update the config: {target.known_discrepancy}'
        pytest.xfail(f'reproduced {value:.4g} vs paper {target.value:.4g}: {target.known_discrepancy}')
    assert within, f'{target.name}: reproduced {value:.4g}, paper {target.value:.4g} ± {target.tolerance} ({target.where})'
