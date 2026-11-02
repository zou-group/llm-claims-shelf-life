"""Model identities: the model tables, OpenRouter route entries, and release dates.

Tables (record IDs from the model catalogue):
  PANEL     earlier model table, kept verbatim so that earlier runs reproduce exactly.
  MODELS    the registered model table (50 models; routes, API IDs, prices, reasoning flag).
  PANEL20   the post's 20-model panel, a subset of MODELS.
New work should use MODELS / PANEL20.

Model keys are the strings the policies accept:
  panel record IDs ('M206'), OpenRouter IDs ('anthropic/claude-sonnet-5.5'), and direct labels
  ('openai-direct/gpt-4-0613', 'anthropic-direct/claude-sonnet-5-5', 'openai-direct/gpt-6.1-sol').
"""
import functools
import json

from .paths import DATA

# record_id -> (route, api model id, input $/M, output $/M, reasoning-capable)
# Copied verbatim from the first-pass panel client (prices frozen 2026-10-01).
PANEL = {
    'M001': ('openai', 'gpt-4o-2024-05-13', 5.0, 15.0, False),
    'M015': ('openai', 'gpt-4o-mini-2024-07-18', 0.15, 0.6, False),
    'M017': ('openrouter', 'meta-llama/llama-3.1-70b-instruct', 0.4, 0.4, False),
    'M025': ('openrouter', 'qwen/qwen-2.5-72b-instruct', 0.36, 0.4, False),
    'M041': ('openrouter', 'meta-llama/llama-3.3-70b-instruct', 1.04, 1.04, False),
    'M045': ('openrouter', 'deepseek/deepseek-chat', 0.2574, 1.0287, False),
    'M046': ('openrouter', 'deepseek/deepseek-r1', 0.7, 2.5, True),
    'M061': ('openrouter', 'google/gemma-3-27b-it', 0.08, 0.45, False),
    'M068': ('openai', 'gpt-4.1-2025-04-14', 2.0, 8.0, False),
    'M073': ('openai', 'o3-2025-04-16', 2.0, 8.0, True),
    'M076': ('openrouter', 'qwen/qwen3-235b-a22b', 0.455, 1.82, True),
    'M090': ('openrouter', 'google/gemini-2.5-flash', 0.3, 2.5, True),
    'M092': ('openrouter', 'google/gemini-2.5-pro', 1.25, 10.0, True),
    'M093': ('openrouter', 'mistralai/mistral-small-3.2-24b-instruct', 0.09375, 0.25, False),
    'M097': ('openrouter', 'z-ai/glm-4.5', 0.6, 2.2, True),
    'M102': ('openrouter', 'openai/gpt-oss-120b', 0.15, 0.6, True),
    'M104': ('openai', 'gpt-5-2025-08-07', 1.25, 10.0, True),
    'M116': ('anthropic', 'claude-sonnet-4-5-20250929', 3.0, 15.0, False),
    'M119': ('anthropic', 'claude-haiku-4-5-20251001', 1.0, 5.0, False),
    'M136': ('openrouter', 'google/gemini-3-flash-preview', 0.5, 3.0, True),
    'M140': ('anthropic', 'claude-opus-4-6', 5.0, 25.0, False),
    'M143': ('openrouter', 'z-ai/glm-5', 0.6, 1.92, True),
    'M144': ('openrouter', 'qwen/qwen3.5-397b-a17b', 0.55, 3.5, True),
    'M145': ('anthropic', 'claude-sonnet-4-6', 3.0, 15.0, False),
    'M180': ('openai', 'gpt-5.6-luna', 0.2, 1.2, True),
    'M196': ('openrouter', 'z-ai/glm-5.3-flash', 0.15, 0.5, True),
    'M199': ('openrouter', 'google/gemini-3.8-flash', 0.75, 3.75, True),
    'M201': ('openrouter', 'deepseek/deepseek-v4.1-flash', 0.015543, 0.396, True),
    'M206': ('anthropic', 'claude-sonnet-5-5', 2.0, 10.0, True),
    'M207': ('openai', 'gpt-6.1-sol', 2.0, 10.0, True),
}

# The registered model table (record IDs listed in experiments/run_minimal_setups/analysis/inputs/models.json).
# Same tuple layout as PANEL.
# Route rule, as for PANEL: Anthropic direct > OpenAI direct > OpenRouter (no Together). Models shared with PANEL
# reuse its entries unchanged. For the others: direct IDs and official prices from the catalogue's direct records
# (2026-10-01); a retired direct API falls back to OpenRouter (Claude Sonnet 4). OpenRouter prices are the frozen
# 2026-10-01 listing; reasoning-capable = OpenRouter lists the 'reasoning' parameter (the rule that reproduces every
# PANEL OpenRouter flag), OpenAI o-/GPT-5+/GPT-6 models, and Claude 5-series. Claude 4.x stays non-reasoning on any
# route, as in PANEL. Note: GPT-5.6 Sol's direct price ($4/$20) is double its OpenRouter listing ($2/$10).
MODELS = {
    'M001': PANEL['M001'],  # GPT-4o (2024-05-13); direct API shutdown announced for 2026-10-23
    'M015': PANEL['M015'],  # GPT-4o mini
    'M017': PANEL['M017'],  # Llama 3.1 70B Instruct
    'M019': ('openrouter', 'mistralai/mistral-large-2407', 2.0, 6.0, False),  # Mistral Large 2 (2407)
    'M020': ('openai', 'gpt-4o-2024-08-06', 2.5, 10.0, False),  # GPT-4o (2024-08-06)
    'M025': PANEL['M025'],  # Qwen2.5-72B-Instruct
    'M039': ('openai', 'o1', 15.0, 60.0, True),  # o1; direct API shutdown announced for 2026-10-23
    'M041': PANEL['M041'],  # Llama 3.3 70B Instruct
    'M045': PANEL['M045'],  # DeepSeek V3
    'M046': PANEL['M046'],  # DeepSeek R1
    'M049': ('openrouter', 'mistralai/mistral-small-24b-instruct-2501', 0.05, 0.08, False),  # Mistral Small 3 (2501)
    'M066': ('openrouter', 'meta-llama/llama-4-maverick', 0.1875, 0.6525, False),  # Llama 4 Maverick
    'M068': PANEL['M068'],  # GPT-4.1
    'M073': PANEL['M073'],  # o3
    'M076': PANEL['M076'],  # Qwen3-235B-A22B
    'M084': ('openrouter', 'anthropic/claude-sonnet-4', 3.0, 15.0, False),  # Claude Sonnet 4; direct API retired
    'M090': PANEL['M090'],  # Gemini 2.5 Flash
    'M092': PANEL['M092'],  # Gemini 2.5 Pro
    'M097': PANEL['M097'],  # GLM-4.5
    'M102': PANEL['M102'],  # gpt-oss-120b
    'M104': PANEL['M104'],  # GPT-5
    'M109': ('openrouter', 'deepseek/deepseek-chat-v3.1', 0.25, 0.95, True),  # DeepSeek V3.1
    'M116': PANEL['M116'],  # Claude Sonnet 4.5
    'M119': PANEL['M119'],  # Claude Haiku 4.5
    'M132': ('openrouter', 'mistralai/mistral-large-2512', 0.5, 1.5, False),  # Mistral Large 3
    'M133': ('openai', 'gpt-5.2', 1.75, 14.0, True),  # GPT-5.2
    'M136': PANEL['M136'],  # Gemini 3 Flash Preview
    'M140': PANEL['M140'],  # Claude Opus 4.6
    'M143': PANEL['M143'],  # GLM-5
    'M144': PANEL['M144'],  # Qwen3.5-397B-A17B
    'M145': PANEL['M145'],  # Claude Sonnet 4.6
    'M146': ('openrouter', 'google/gemini-3.1-pro-preview', 2.0, 12.0, True),  # Gemini 3.1 Pro Preview
    'M152': ('openai', 'gpt-5.4', 2.5, 15.0, True),  # GPT-5.4
    'M155': ('openai', 'gpt-5.4-mini', 0.75, 4.5, True),  # GPT-5.4 mini
    'M171': ('openrouter', 'qwen/qwen3.7-max', 1.475, 4.425, True),  # Qwen3.7-Max
    'M172': ('anthropic', 'claude-opus-4-8', 5.0, 25.0, False),  # Claude Opus 4.8
    'M178': ('anthropic', 'claude-sonnet-5', 2.0, 10.0, True),  # Claude Sonnet 5
    'M179': ('openrouter', 'meta/muse-spark-1.1', 1.25, 4.25, True),  # Muse Spark 1.1
    'M180': PANEL['M180'],  # GPT-5.6 Luna
    'M181': ('openai', 'gpt-5.6-sol', 4.0, 20.0, True),  # GPT-5.6 Sol
    'M183': ('openrouter', 'qwen/qwen3.7-flash', 0.03, 0.13, True),  # Qwen3.7-Flash
    'M190': ('openrouter', 'deepseek/deepseek-v4-pro-0813', 1.32, 3.96, True),  # DeepSeek V4 Pro 0813
    'M193': ('openrouter', 'z-ai/glm-5.3', 0.2219, 6.34, True),  # GLM-5.3
    'M196': PANEL['M196'],  # GLM-5.3-Flash
    'M198': ('openrouter', 'qwen/qwen3.8-max-0902', 2.0, 6.0, True),  # Qwen3.8-Max-0902
    'M199': PANEL['M199'],  # Gemini 3.8 Flash
    'M203': ('anthropic', 'claude-opus-5-5', 4.0, 20.0, True),  # Claude Opus 5.5
    'M204': ('openai', 'gpt-6-luna', 0.1, 0.5, True),  # GPT-6 Luna
    'M206': PANEL['M206'],  # Claude Sonnet 5.5
    'M207': PANEL['M207'],  # GPT-6.1 Sol
}

# The post's 20-model panel (experiments/run_minimal_setups/analysis/inputs/panel20.json).
PANEL20 = ('M015', 'M017', 'M025', 'M045', 'M049', 'M066', 'M090', 'M102', 'M119', 'M133',
           'M140', 'M143', 'M171', 'M183', 'M190', 'M196', 'M199', 'M204', 'M206', 'M207')

# Named model sets accepted by `--models` (expanded by panel_models).
PANEL_SETS = {'panel20': PANEL20, 'all': tuple(MODELS)}


def panel_entry(model):
    """(route, api model id, $in/M, $out/M, reasoning) for a panel record ID: PANEL first (so legacy behaviour is
    unchanged), then MODELS; None for other keys."""
    return PANEL.get(model) or MODELS.get(model)


def panel_models(names):
    """Expand panel-set names ('panel20', 'all') in a model list into record IDs, keeping order and
    dropping duplicates; other model keys pass through unchanged."""
    out = []
    for name in names:
        for m in PANEL_SETS.get(name, (name,)):
            if m not in out:
                out.append(m)
    return out


# Keys used by initial-reproduction for the same underlying models (for joining old and new outputs).
ALIASES = {
    'anthropic/claude-sonnet-5.5': 'M206', 'anthropic-direct/claude-sonnet-5-5': 'M206',
    'openai/gpt-6.1-sol': 'M207', 'openai-direct/gpt-6.1-sol': 'M207',
    'openai-direct/gpt-4o-2024-05-13': 'M001', 'gpt-4o-2024-05-13': 'M001',
}


OPENROUTER_PARAMETERS = ('temperature', 'top_p', 'top_k', 'max_tokens', 'stop', 'seed', 'logprobs', 'top_logprobs',
                         'frequency_penalty', 'presence_penalty', 'repetition_penalty', 'response_format',
                         'include_reasoning', 'reasoning', 'tools', 'tool_choice')


OPENROUTER_DEFAULT_PRICING = {'prompt': '0.000005', 'completion': '0.000015'}   # assumed for the helper routes below

# OpenRouter IDs that configs use for judges, coders or paper-model reproductions and that are not in the tables.
# They are routed with every parameter accepted and the default pricing above (a cost bound, not a quote).
OPENROUTER_HELPERS = ('anthropic/claude-sonnet-5.5', 'google/gemma-7b-it',
                      'meta-llama/Llama-2-13b-chat-hf', 'meta-llama/llama-3.1-8b-instruct', 'mistralai/Mistral-7B-Instruct-v0.1',
                      'openai/gpt-3.5-turbo-0613', 'openai/gpt-4-0613', 'openai/gpt-4.1')


@functools.lru_cache(maxsize=None)
def openrouter_catalog():
    """OpenRouter model entries derived from the model tables: id -> pricing (USD per token) and the parameters the
    route is assumed to accept."""
    out = {}
    for table in (PANEL, MODELS):
        for route, mid, pin, pout, _ in table.values():
            if route == 'openrouter':
                out[mid] = {'id': mid, 'pricing': {'prompt': str(pin / 1e6), 'completion': str(pout / 1e6)},
                            'supported_parameters': list(OPENROUTER_PARAMETERS)}
    for mid in OPENROUTER_HELPERS:
        out.setdefault(mid, {'id': mid, 'pricing': dict(OPENROUTER_DEFAULT_PRICING),
                             'supported_parameters': list(OPENROUTER_PARAMETERS), 'assumed': True})
    return out


@functools.lru_cache(maxsize=None)
def panel_capabilities():
    """Params each direct panel model rejects (the first-pass capabilities log)."""
    return json.loads((DATA / 'panel_capabilities.json').read_text())


@functools.lru_cache(maxsize=None)
def model_capabilities():
    """Params each direct MODELS model rejects: the verified panel_capabilities entries, plus family-based
    assumptions for direct models new in MODELS (data/model_capabilities.json, each entry says which)."""
    caps = dict(panel_capabilities())
    caps.update({k: v for k, v in json.loads((DATA / 'model_capabilities.json').read_text()).items() if not k.startswith('_')})
    return caps


# Release dates of the registered models (the date used for release-ordered analyses).
RELEASE_DATES = {
    'M001': '2024-05-13',
    'M015': '2024-07-18',
    'M017': '2024-07-23',
    'M019': '2024-07-24',
    'M020': '2024-08-06',
    'M025': '2024-09-19',
    'M039': '2024-12-05',
    'M041': '2024-12-06',
    'M045': '2024-12-26',
    'M046': '2025-01-20',
    'M049': '2025-01-30',
    'M061': '2025-03-12',
    'M066': '2025-04-05',
    'M068': '2025-04-14',
    'M073': '2025-04-16',
    'M076': '2025-04-29',
    'M084': '2025-05-22',
    'M090': '2025-05-20',
    'M092': '2025-06-05',
    'M093': '2025-06-20',
    'M097': '2025-07-28',
    'M102': '2025-08-05',
    'M104': '2025-08-07',
    'M109': '2025-08-21',
    'M116': '2025-09-29',
    'M119': '2025-10-15',
    'M132': '2025-12-02',
    'M133': '2025-12-11',
    'M136': '2025-12-17',
    'M140': '2026-02-05',
    'M143': '2026-02-12',
    'M144': '2026-02-16',
    'M145': '2026-02-17',
    'M146': '2026-02-19',
    'M152': '2026-03-05',
    'M155': '2026-03-17',
    'M171': '2026-05-20',
    'M172': '2026-05-28',
    'M178': '2026-06-30',
    'M179': '2026-07-09',
    'M180': '2026-07-09',
    'M181': '2026-07-09',
    'M183': '2026-07-21',
    'M190': '2026-08-13',
    'M193': '2026-08-14',
    'M196': '2026-08-26',
    'M198': '2026-09-02',
    'M199': '2026-09-02',
    'M201': '2026-09-10',
    'M203': '2026-09-22',
    'M204': '2026-09-22',
    'M206': '2026-09-28',
    'M207': '2026-09-29',
}


def record_id(model):
    """Panel record ID for a model key, or None when the model is not in the release catalogue."""
    if model in PANEL or model in MODELS:
        return model
    if model in ALIASES:
        return ALIASES[model]
    for table in (PANEL, MODELS):
        for rid, (route, mid, *_rest) in table.items():
            if model in {mid, f'{route}-direct/{mid}', f'openai-direct/{mid}'}:
                return rid
    return None


def release_date(model):
    """Release date of a panel model from the model table, or None."""
    rid = record_id(model)
    return RELEASE_DATES.get(rid) if rid else None