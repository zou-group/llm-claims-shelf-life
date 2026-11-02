"""The model table (MODELS), the 20-model panel (PANEL20) and the earlier model table (PANEL)."""
import json

import pytest

from shelflife import models as M
from shelflife.paths import EXPERIMENTS
from shelflife.policies import get_policy, policy_for

INPUTS = EXPERIMENTS / 'run_minimal_setups' / 'analysis' / 'inputs'
CATALOG = INPUTS / 'model-catalog-with-size-and-pricing.json'


def _frozen(file):
    path = INPUTS / file
    if not path.exists():
        pytest.skip(f'{path} not present')
    return json.loads(path.read_text())


def test_panel_sets_match_frozen_panel_files():
    assert set(M.MODELS) == set(_frozen('models.json')['record_ids'])
    assert set(M.PANEL20) == set(_frozen('panel20.json')['record_ids'])
    assert len(M.MODELS) == 50 and len(M.PANEL20) == 20 and set(M.PANEL20) <= set(M.MODELS)


def test_shared_models_keep_legacy_entries():
    for rid in set(M.PANEL) & set(M.MODELS):
        assert M.MODELS[rid] == M.PANEL[rid], rid


def test_panel_models_expansion():
    assert M.panel_models(['panel20']) == list(M.PANEL20)
    assert M.panel_models(['M206', 'panel20', 'M001'])[:2] == ['M206', 'M015']
    assert M.panel_models(['M999']) == ['M999']


def test_policy_resolution():
    assert policy_for('M206').name == 'panel_v1'          # legacy IDs keep the legacy policy
    assert policy_for('M204').name == 'panel_v2'        # new IDs use the model-table policy
    p = get_policy('panel_v2')
    assert all(p.accepts(m) for m in M.MODELS) and not p.accepts('M061')


ITEM = {'id': 'x', 'messages': [{'role': 'system', 'content': 'Be brief.'}, {'role': 'user', 'content': 'Hi'}],
        'params': {'temperature': 0.7, 'top_p': 0.9, 'max_tokens': 256}}


def test_shared_models_same_payload_under_both_policies():
    old, new = get_policy('panel_v1'), get_policy('panel_v2')
    for rid in set(M.PANEL) & set(M.MODELS):
        assert old.adapt(rid, ITEM) == new.adapt(rid, ITEM), rid
        assert old.endpoint(rid)[0] == new.endpoint(rid)[0]


@pytest.mark.parametrize('rid', sorted(set(M.MODELS) - set(M.PANEL)))
def test_new_models_adapt_and_route(rid):
    p = get_policy('panel_v2')
    route = M.MODELS[rid][0]
    payload, ad = p.adapt(rid, ITEM)
    url, _ = p.endpoint(rid)
    assert ad['route'] == route
    assert {'anthropic': 'api.anthropic.com', 'openai': 'api.openai.com', 'openrouter': 'openrouter.ai'}[route] in url
    if route == 'anthropic':
        assert payload['system'] == 'Be brief.' and payload['model'] == M.MODELS[rid][1]
    rejected = set(M.model_capabilities().get(rid, {}).get('rejected_params', []))
    assert not (rejected & set(payload)), rid
    assert p.estimate(rid, payload) > 0
