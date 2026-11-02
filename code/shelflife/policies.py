"""Request-adaptation policies, ported exactly from the experiment harnesses.

Each policy turns a case ``{"id", "messages", "params"}`` into a provider payload and records every adaptation
(omitted parameters, changed output caps), so a run under a given policy is directly comparable with the earlier data.

Policies
  initial_openrouter_v1   experiments/initial-reproduction/harness/client.py (OpenRouter)
  initial_anthropic_v1    experiments/initial-reproduction/harness/direct_anthropic.py
  initial_openai_v1       experiments/initial-reproduction/harness/direct_openai.py (historical snapshots)
  initial_gpt6_v1         experiments/initial-reproduction/harness/direct_gpt6.py
  panel_v1                experiments/the first-pass panel client (earlier model table)
  panel_v2                panel_v1's adaptation rules over the MODELS table
  originals_v1            historical OpenAI chat snapshots for original-model reproductions (earlier pipeline)
"""
import json
import os

from . import models as M

OR_FEE = 0.055  # OpenRouter credit-purchase fee (panel accounting adds it on top of usage.cost)


class Policy:
    name = None
    retry_http = frozenset()  # explicit, unbilled rejections that may be resubmitted with the identical payload
    pacing = {}               # model key -> requests per minute

    def accepts(self, model):
        raise NotImplementedError

    def adapt(self, model, item):
        raise NotImplementedError

    def estimate(self, model, payload):
        raise NotImplementedError

    def endpoint(self, model):
        """(url, headers). Headers read keys from the environment and are never persisted."""
        raise NotImplementedError

    def summarize(self, model, response, reserve):
        """Return content, reasoning, finish_reason, returned_model, provider, charged_usd, cost_kind."""
        raise NotImplementedError

    def http_error_cost(self, model, status, reserve):
        return reserve, 'conservative_bound_failed_request'


def _openai_style(response):
    choice = response['choices'][0]
    message = choice.get('message', {})
    reasoning = message.get('reasoning') or message.get('reasoning_content') or ''
    return message.get('content') or '', reasoning, choice.get('finish_reason')


def _openrouter_headers(title):
    return {'Authorization': 'Bearer ' + os.environ.get('OPENROUTER_API_KEY', ''), 'Content-Type': 'application/json',
            'X-OpenRouter-Title': title}


class InitialOpenRouter(Policy):
    """initial-reproduction/harness/client.py: Client.adapt/estimate/response_cost."""
    name = 'initial_openrouter_v1'
    MODERN = {'anthropic/claude-sonnet-5.5', 'openai/gpt-6.1-sol', 'z-ai/glm-5.3'}
    pacing = {'anthropic/claude-sonnet-5.5': 180, 'openai/gpt-6.1-sol': 120}

    def __init__(self, catalog=None):
        self.catalog = catalog or M.openrouter_catalog()

    def accepts(self, model):
        return model in self.catalog

    def _entry(self, model):
        return self.catalog[model]

    def adapt(self, model, item):
        info = self._entry(model)
        supported = set(info.get('supported_parameters', []))
        requested = dict(item.get('params', {}))
        if int(requested.get('n', 1)) != 1:
            raise ValueError('n>1 requires explicit within-replication sampling; generic client cannot silently replace it')
        requested.pop('n', None)
        # Both keys are popped: the inner pop is evaluated before the outer one (legacy behaviour, kept).
        original_max = requested.pop('max_tokens', requested.pop('max_completion_tokens', 1024))
        modern = model in self.MODERN
        cap = max(int(original_max), 4096) if modern else max(int(original_max), 2)
        effective, omitted = {}, {}
        for k, v in requested.items():
            if k in supported:
                effective[k] = v
            else:
                omitted[k] = v
        cap_param = 'max_tokens' if 'max_tokens' in supported else 'max_completion_tokens'
        effective[cap_param] = cap
        if modern and 'include_reasoning' in supported:
            effective['include_reasoning'] = True
        payload = {'model': model, 'messages': item['messages'], **effective,
                   'provider': {'require_parameters': True}}
        adaptations = {'requested_params': item.get('params', {}), 'omitted_unsupported': omitted,
                       'output_cap_changed': cap != original_max, 'original_cap': original_max,
                       'max_tokens_unspecified_in_source': not any(k in item.get('params', {}) for k in ('max_tokens', 'max_completion_tokens')),
                       'effective_cap': cap, 'reasoning': 'provider default; reasoning saved where exposed'}
        return payload, adaptations

    def estimate(self, model, payload):
        pricing = self._entry(model)['pricing']
        prompt_bound = len(json.dumps(payload['messages'], ensure_ascii=False).encode()) + 128 * len(payload['messages'])
        if 'system' in payload:
            prompt_bound += len(json.dumps(payload['system'], ensure_ascii=False).encode()) + 128
        cap = payload.get('max_tokens', payload.get('max_completion_tokens'))
        return 1.25 * (prompt_bound * float(pricing['prompt']) + cap * float(pricing['completion']))

    def endpoint(self, model):
        return 'https://openrouter.ai/api/v1/chat/completions', _openrouter_headers('LLM claim initial reproduction')

    def summarize(self, model, response, reserve):
        content, reasoning, finish = _openai_style(response)
        cost = response.get('usage', {}).get('cost')
        charged, kind = (float(cost), 'provider_reported') if cost is not None else (reserve, 'conservative_bound_missing_usage_cost')
        return dict(content=content, reasoning=reasoning, finish_reason=finish, returned_model=response.get('model'),
                    provider=response.get('provider'), charged_usd=charged, cost_kind=kind)


class InitialAnthropicDirect(Policy):
    """initial-reproduction/harness/direct_anthropic.py (Sonnet 5.5, adaptive thinking, provider-default sampling)."""
    name = 'initial_anthropic_v1'
    LABEL = 'anthropic-direct/claude-sonnet-5-5'
    pacing = {LABEL: 120}

    def accepts(self, model):
        return model == self.LABEL

    def adapt(self, model, item):
        if model != self.LABEL:
            raise ValueError('Only verified Sonnet5.5 supported')
        params = item.get('params', {})
        if params.get('n', 1) != 1:
            raise ValueError('Only n=1 supported')
        system, ordinary = [], []
        for m in item['messages']:
            if m['role'] == 'system':
                if ordinary:
                    raise ValueError('Do not silently relocate mid-conversation system messages')
                system.append(m['content'])
            else:
                ordinary.append(m)
        payload = {'model': 'claude-sonnet-5-5', 'messages': ordinary, 'max_tokens': max(4096, int(params.get('max_tokens', 4096))),
                   'thinking': {'type': 'adaptive', 'display': 'summarized'}}
        if system:
            payload['system'] = '\n\n'.join(system)
        if params.get('stop'):
            payload['stop_sequences'] = params['stop'] if isinstance(params['stop'], list) else [params['stop']]
        return payload, {'direct_provider': 'Anthropic', 'source_params': params,
                         'omitted_sampling': {k: v for k, v in params.items() if k in {'temperature', 'top_p', 'top_k'}},
                         'thinking': 'adaptive, effort default high; summarized thinking saved',
                         'sampling': 'provider default; non-default sampling unsupported by native API',
                         'docs': 'https://platform.claude.com/docs/en/models/sonnet-5-5/migration-guide',
                         'pricing_source': 'https://platform.claude.com/docs/en/models/sonnet-5-5/overview'}

    def estimate(self, model, payload):
        prompt_bound = len(json.dumps(payload['messages'], ensure_ascii=False).encode()) + 128 * len(payload['messages'])
        if 'system' in payload:
            prompt_bound += len(json.dumps(payload['system'], ensure_ascii=False).encode()) + 128
        return 1.25 * (prompt_bound * 0.000002 + payload['max_tokens'] * 0.00001)

    def endpoint(self, model):
        return 'https://api.anthropic.com/v1/messages', {'x-api-key': os.environ.get('ANTHROPIC_API_KEY', ''),
                                                         'anthropic-version': '2023-06-01', 'Content-Type': 'application/json'}

    def summarize(self, model, response, reserve):
        blocks = response.get('content', [])
        content = '\n'.join(b.get('text', '') for b in blocks if b.get('type') == 'text')
        reasoning = '\n'.join(b.get('thinking', '') for b in blocks if b.get('type') == 'thinking')
        u = response.get('usage', {})
        if 'input_tokens' not in u or 'output_tokens' not in u:
            charged, kind = reserve, 'conservative_bound_missing_usage'
        else:
            charged = (u['input_tokens'] * .000002 + u['output_tokens'] * .00001
                       + u.get('cache_read_input_tokens', 0) * .0000002 + u.get('cache_creation_input_tokens', 0) * .0000025)
            kind = 'usage_estimate_official_pricing'
        return dict(content=content, reasoning=reasoning, finish_reason=response.get('stop_reason'),
                    returned_model=response.get('model'), provider='Anthropic-direct', charged_usd=charged, cost_kind=kind)


class InitialOpenAIDirect(Policy):
    """initial-reproduction/harness/direct_openai.py: exact historical snapshots, source parameters passed through."""
    name = 'initial_openai_v1'
    PRICES = {'gpt-4-0613': (.00003, .00006), 'gpt-4o-2024-05-13': (.000005, .000015),
              'gpt-4o-2024-11-20': (.0000025, .00001)}
    pacing = {'openai-direct/gpt-4o-2024-11-20': 120}

    def accepts(self, model):
        return model.removeprefix('openai-direct/') in self.PRICES and model.startswith('openai-direct/')

    def adapt(self, model, item):
        name = model.removeprefix('openai-direct/')
        if name not in self.PRICES:
            raise ValueError('Only verified historical snapshots supported')
        payload = {'model': name, 'messages': item['messages'], **item.get('params', {})}
        unspecified = not payload.get('max_tokens')
        if unspecified:
            if not name.startswith('gpt-4o-'):
                raise ValueError('Explicit source cap required')
            payload['max_tokens'] = 4096
        if payload.get('n', 1) != 1:
            raise ValueError('Only n=1 supported')
        return payload, {'direct_provider': 'OpenAI', 'requested_params': item.get('params', {}), 'omitted_unsupported': {},
                         'source_cap_retained': not unspecified, 'source_cap_unspecified': unspecified,
                         'cap_choice_if_unspecified': 4096 if unspecified else None,
                         'pricing_source': ('https://developers.openai.com/api/docs/models/gpt-4' if name == 'gpt-4-0613' else
                                            'https://developers.openai.com/api/docs/models/gpt-4o' if name == 'gpt-4o-2024-11-20' else
                                            'https://community.openai.com/t/announcing-gpt-4o-in-the-api/744700')}

    def _prices(self, model):
        return self.PRICES[model.removeprefix('openai-direct/')]

    def estimate(self, model, payload):
        pin, pout = self._prices(model)
        prompt_bound = len(json.dumps(payload['messages'], ensure_ascii=False).encode()) + 128 * len(payload['messages'])
        cap = payload.get('max_tokens', payload.get('max_completion_tokens'))
        return 1.25 * (prompt_bound * pin + cap * pout)

    def endpoint(self, model):
        return 'https://api.openai.com/v1/chat/completions', {'Authorization': 'Bearer ' + os.environ.get('OPENAI_API_KEY', ''),
                                                              'Content-Type': 'application/json'}

    def summarize(self, model, response, reserve):
        content, reasoning, finish = _openai_style(response)
        u = response.get('usage', {})
        if 'prompt_tokens' not in u or 'completion_tokens' not in u:
            charged, kind = reserve, 'conservative_bound_missing_usage'
        else:
            pin, pout = self._prices(model)
            charged = u['prompt_tokens'] * pin + u['completion_tokens'] * pout
            kind = 'usage_upper_estimate_official_uncached_pricing' if model.endswith('2024-11-20') else 'usage_estimate_official_pricing'
        return dict(content=content, reasoning=reasoning, finish_reason=finish, returned_model=response.get('model'),
                    provider=response.get('provider'), charged_usd=charged, cost_kind=kind)


class InitialGPT6Direct(InitialOpenAIDirect):
    """initial-reproduction/harness/direct_gpt6.py: native GPT-6.1 Sol, unsupported sampling dropped, cap floor 4096."""
    name = 'initial_gpt6_v1'
    LABEL = 'openai-direct/gpt-6.1-sol'
    pacing = {LABEL: 120}
    SUPPORTED = {'reasoning_effort', 'response_format', 'seed', 'user'}

    def accepts(self, model):
        return model == self.LABEL

    def adapt(self, model, item):
        if model != self.LABEL:
            raise ValueError('Only verified GPT-6.1-Sol supported')
        source = dict(item.get('params', {}))
        if source.get('n', 1) != 1:
            raise ValueError('n>1 unsupported')
        effective = {k: v for k, v in source.items() if k in self.SUPPORTED}
        cap = max(4096, int(source.get('max_tokens', source.get('max_completion_tokens', 4096))))
        payload = {'model': 'gpt-6.1-sol', 'messages': item['messages'], 'max_completion_tokens': cap, **effective}
        return payload, {'direct_provider': 'OpenAI', 'source_params': source,
                         'omitted_unsupported': {k: v for k, v in source.items() if k not in self.SUPPORTED and k not in {'max_tokens', 'max_completion_tokens', 'n'}},
                         'effective_cap': cap, 'reasoning': 'native default medium; no tools or extra prompt',
                         'docs': 'https://developers.openai.com/api/docs/guides/latest-model',
                         'pricing_source': 'https://developers.openai.com/api/docs/models/gpt-6.1-sol'}

    def _prices(self, model):
        return (.000002, .00001)

    def summarize(self, model, response, reserve):
        out = super().summarize(model, response, reserve)
        if out['cost_kind'] != 'conservative_bound_missing_usage':
            out['cost_kind'] = 'usage_upper_estimate_official_uncached_pricing'
        return out


class Panel(Policy):
    """the first-pass panel client: Anthropic direct > OpenAI direct > OpenRouter."""
    name = 'panel_v1'
    retry_http = frozenset({429, 500, 502, 503, 504, 529})

    title = 'LLM claim panel reproduction'

    def __init__(self, capabilities=None):
        self.table = M.PANEL
        self.caps = M.panel_capabilities() if capabilities is None else capabilities

    def accepts(self, model):
        return model in self.table

    def adapt(self, rid, item):
        route, mid, _, _, reasoning = self.table[rid]
        params = dict(item.get('params', {}))
        if int(params.pop('n', 1)) != 1:
            raise ValueError('n>1 must be expanded into repeats by the caller')
        original_cap = params.pop('max_tokens', params.pop('max_completion_tokens', None))
        source_cap = 1024 if original_cap is None else int(original_cap)
        cap = max(source_cap, item.get('min_cap_reasoning', 4096)) if reasoning else source_cap
        rejected = set(self.caps.get(rid, {}).get('rejected_params', []))
        omitted = {k: v for k, v in params.items() if k in rejected}
        kept = {k: v for k, v in params.items() if k not in rejected}
        messages = item['messages']
        if route == 'anthropic':
            system = [m['content'] for m in messages if m['role'] == 'system']
            ordinary = [m for m in messages if m['role'] != 'system']
            if system and messages[0]['role'] != 'system':
                raise ValueError('mid-conversation system message')
            payload = {'model': mid, 'messages': ordinary, 'max_tokens': cap}
            if system:
                payload['system'] = '\n\n'.join(system)
            if 'temperature' in kept and 'top_p' in kept:
                omitted['top_p'] = kept.pop('top_p')
            for k in ('temperature', 'top_p', 'top_k'):
                if k in kept:
                    payload[k] = kept.pop(k)
            if 'stop' in kept:
                s = kept.pop('stop')
                payload['stop_sequences'] = s if isinstance(s, list) else [s]
            if kept:
                omitted.update(kept)
        elif route == 'openai':
            payload = {'model': mid, 'messages': messages, **kept}
            payload['max_completion_tokens' if reasoning else 'max_tokens'] = cap
        else:
            payload = {'model': mid, 'messages': messages, **kept, 'max_tokens': cap,
                       'provider': {'require_parameters': True, 'ignore': ['together']}}
            if reasoning:
                payload['include_reasoning'] = True
        adaptations = {'route': route, 'requested_params': item.get('params', {}), 'omitted_unsupported': omitted,
                       'source_cap': original_cap, 'effective_cap': cap, 'output_cap_changed': cap != source_cap,
                       'reasoning': 'provider default effort; reasoning saved where exposed'}
        return payload, adaptations

    def estimate(self, rid, payload):
        _, _, pin, pout, _ = self.table[rid]
        prompt_bound = len(json.dumps(payload.get('messages'), ensure_ascii=False).encode()) + 128 * len(payload.get('messages', []))
        prompt_bound += len(json.dumps(payload.get('system', ''), ensure_ascii=False).encode())
        cap = payload.get('max_tokens', payload.get('max_completion_tokens'))
        return 1.25 * (prompt_bound * pin + cap * pout) / 1e6 * (1 + OR_FEE)

    def endpoint(self, rid):
        route = self.table[rid][0]
        if route == 'anthropic':
            return 'https://api.anthropic.com/v1/messages', {'x-api-key': os.environ.get('ANTHROPIC_API_KEY', ''),
                                                             'anthropic-version': '2023-06-01', 'Content-Type': 'application/json'}
        if route == 'openai':
            return 'https://api.openai.com/v1/chat/completions', {'Authorization': 'Bearer ' + os.environ.get('OPENAI_API_KEY', ''),
                                                                  'Content-Type': 'application/json'}
        return 'https://openrouter.ai/api/v1/chat/completions', _openrouter_headers(self.title)

    def summarize(self, rid, response, reserve):
        route, _, pin, pout, _ = self.table[rid]
        if route == 'anthropic':
            blocks = response.get('content', [])
            content = '\n'.join(b.get('text', '') for b in blocks if b.get('type') == 'text')
            reasoning = '\n'.join(b.get('thinking', '') for b in blocks if b.get('type') == 'thinking')
            u = response.get('usage', {})
            cost = None if 'output_tokens' not in u else (
                u.get('input_tokens', 0) * pin + u['output_tokens'] * pout
                + u.get('cache_read_input_tokens', 0) * pin * .1 + u.get('cache_creation_input_tokens', 0) * pin * 1.25) / 1e6
            finish, provider, kind = response.get('stop_reason'), 'Anthropic-direct', 'usage_x_list_price'
        else:
            content, reasoning, finish = _openai_style(response)
            u = response.get('usage', {})
            if route == 'openai':
                cost = None if 'completion_tokens' not in u else (u.get('prompt_tokens', 0) * pin + u['completion_tokens'] * pout) / 1e6
                provider, kind = 'OpenAI-direct', 'usage_x_list_price_uncached'
            else:
                raw = u.get('cost')
                cost = None if raw is None else float(raw) * (1 + OR_FEE)
                provider, kind = response.get('provider'), 'provider_reported_plus_fee'
        charged, kind = (cost, kind) if cost is not None else (reserve, 'conservative_bound_missing_usage')
        return dict(content=content, reasoning=reasoning, finish_reason=finish, returned_model=response.get('model'),
                    provider=provider, charged_usd=charged, cost_kind=kind)

    def http_error_cost(self, rid, status, reserve):
        # 4xx validation errors are not billed; other failures keep the conservative reserve.
        if 400 <= status < 500:
            return 0.0, 'rejected_4xx_unbilled'
        return reserve, 'conservative_bound_failed_request'


class PanelV2(Panel):
    """Authoritative panels (the model table, the panel): panel_v1's rules over models.MODELS, same route priority
    (Anthropic direct > OpenAI direct > OpenRouter). Shared models have identical table entries, so their payloads
    match panel_v1's; only the policy name and the OpenRouter title header differ."""
    name = 'panel_v2'
    title = 'LLM claim panel run'

    def __init__(self, capabilities=None):
        super().__init__(M.model_capabilities() if capabilities is None else capabilities)
        self.table = M.MODELS

    def adapt(self, rid, item):
        # earlier pipeline fix (2026-10-04): a variant with stop=None (e.g. C057 paper_base) made panel_v1's Anthropic branch
        # send stop_sequences=[None] (HTTP 400). A null stop means "no stop sequence", so it is dropped and recorded.
        params = item.get('params', {})
        if 'stop' in params and params['stop'] is None:
            item = {**item, 'params': {k: v for k, v in params.items() if k != 'stop'}}
            payload, adaptations = super().adapt(rid, item)
            adaptations['requested_params'] = params
            adaptations['dropped_null_stop'] = True
            return payload, adaptations
        return super().adapt(rid, item)


class OriginalsOpenAIDirect(Policy):
    """Original-model reproductions (earlier pipeline, 2026-10-04): historical OpenAI chat snapshots that the earlier
    policies do not cover, on /v1/chat/completions. Verified reachable on 2026-10-04 by a live probe; davinci-002,
    babbage-002, gpt-3.5-turbo-instruct(-0914) and gpt-3.5-turbo-1106 return model_not_found (retired), so no completion
    route exists. Source parameters pass through, except top_k (not an OpenAI parameter), which is omitted and recorded.
    A missing output cap becomes 4096 (recorded as cap_choice_if_unspecified), unlike initial_openai_v1, which raises.
    gpt-4-0613 and gpt-4o-2024-05-13 are also served here, but policy_for keeps routing them to initial_openai_v1
    (legacy parity); pass --policy originals_v1 to run an uncapped source protocol on them.
    Prices: OpenAI list prices per 1M tokens, 2026-10-04."""
    name = 'originals_v1'
    retry_http = frozenset({429, 500, 502, 503, 504})
    PRICES = {'gpt-3.5-turbo-0125': (.5, 1.5), 'gpt-4-turbo-2024-04-09': (10., 30.), 'gpt-4o-2024-08-06': (2.5, 10.),
              'gpt-4-0613': (30., 60.), 'gpt-4o-2024-05-13': (5., 15.)}
    UNSUPPORTED = ('top_k',)
    DEFAULT_CAP = 4096  # used only when the source sets no cap; recorded in adaptations

    def accepts(self, model):
        return model.startswith('openai-direct/') and model.removeprefix('openai-direct/') in self.PRICES

    def adapt(self, model, item):
        if not self.accepts(model):
            raise ValueError(f'originals_v1 does not serve {model}')
        params = dict(item.get('params', {}))
        if int(params.pop('n', 1)) != 1:
            raise ValueError('n>1 must be expanded into repeats by the caller')
        unspecified = not (params.get('max_tokens') or params.get('max_completion_tokens'))
        if unspecified:
            params['max_tokens'] = self.DEFAULT_CAP
        omitted = {k: params.pop(k) for k in self.UNSUPPORTED if k in params}
        payload = {'model': model.removeprefix('openai-direct/'), 'messages': item['messages'], **params}
        return payload, {'direct_provider': 'OpenAI', 'requested_params': item.get('params', {}), 'omitted_unsupported': omitted,
                         'source_cap_unspecified': unspecified, 'cap_choice_if_unspecified': self.DEFAULT_CAP if unspecified else None,
                         'pricing_source': 'https://platform.openai.com/docs/pricing (2026-10-04)'}

    def _prices(self, model):
        pin, pout = self.PRICES[model.removeprefix('openai-direct/')]
        return pin / 1e6, pout / 1e6

    def estimate(self, model, payload):
        pin, pout = self._prices(model)
        prompt_bound = len(json.dumps(payload['messages'], ensure_ascii=False).encode()) + 128 * len(payload['messages'])
        cap = payload.get('max_tokens', payload.get('max_completion_tokens'))
        return 1.25 * (prompt_bound * pin + cap * pout)

    def endpoint(self, model):
        return 'https://api.openai.com/v1/chat/completions', {'Authorization': 'Bearer ' + os.environ.get('OPENAI_API_KEY', ''),
                                                              'Content-Type': 'application/json'}

    def summarize(self, model, response, reserve):
        content, reasoning, finish = _openai_style(response)
        u = response.get('usage', {})
        if 'prompt_tokens' not in u or 'completion_tokens' not in u:
            charged, kind = reserve, 'conservative_bound_missing_usage'
        else:
            pin, pout = self._prices(model)
            charged, kind = u['prompt_tokens'] * pin + u['completion_tokens'] * pout, 'usage_x_list_price_uncached'
        return dict(content=content, reasoning=reasoning, finish_reason=finish, returned_model=response.get('model'),
                    provider='OpenAI-direct', charged_usd=charged, cost_kind=kind)

    def http_error_cost(self, model, status, reserve):
        if 400 <= status < 500:
            return 0.0, 'rejected_4xx_unbilled'
        return reserve, 'conservative_bound_failed_request'


POLICIES = {cls.name: cls for cls in (InitialOpenRouter, InitialAnthropicDirect, InitialOpenAIDirect, InitialGPT6Direct, Panel,
                                      PanelV2, OriginalsOpenAIDirect)}


def get_policy(name):
    if name not in POLICIES:
        raise KeyError(f'unknown policy {name}; choose from {sorted(POLICIES)}')
    return POLICIES[name]()


def policy_for(model):
    """Default policy for a model key: earlier record IDs keep panel_v1 (unchanged behaviour for earlier
    runs); record IDs only in MODELS use panel_v2. Pass --policy panel_v2 to run a whole new panel under one name."""
    if model in M.PANEL:
        return get_policy('panel_v1')
    if model in M.MODELS:
        return get_policy('panel_v2')
    for name in ('initial_anthropic_v1', 'initial_gpt6_v1', 'initial_openai_v1', 'originals_v1'):
        policy = get_policy(name)
        if policy.accepts(model):
            return policy
    policy = get_policy('initial_openrouter_v1')
    if policy.accepts(model):
        return policy
    raise KeyError(f'no policy accepts model {model!r}')
