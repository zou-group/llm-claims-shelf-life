# Models, routes and API keys

The post uses 20 models released between July 2024 and September 2026. The route column is what
[`code/shelflife/models.py`](../shelflife/models.py) uses today, and so which API key a new run of that model needs.
Prices are the frozen 2026-10-01 list prices, in USD per million input / output tokens.

| ID | Model | Released | Route | Key | API model ID | $/1M in / out |
|---|---|---|---|---|---|---|
| M015 | GPT-4o mini | 2024-07-18 | OpenAI (direct) | `OPENAI_API_KEY` | `gpt-4o-mini-2024-07-18` | 0.15 / 0.6 |
| M017 | Llama 3.1 70B Instruct | 2024-07-23 | OpenRouter | `OPENROUTER_API_KEY` | `meta-llama/llama-3.1-70b-instruct` | 0.4 / 0.4 |
| M025 | Qwen2.5-72B-Instruct | 2024-09-19 | OpenRouter | `OPENROUTER_API_KEY` | `qwen/qwen-2.5-72b-instruct` | 0.36 / 0.4 |
| M045 | DeepSeek V3 | 2024-12-26 | OpenRouter | `OPENROUTER_API_KEY` | `deepseek/deepseek-chat` | 0.2574 / 1.0287 |
| M049 | Mistral Small 3 (2501) | 2025-01-30 | OpenRouter | `OPENROUTER_API_KEY` | `mistralai/mistral-small-24b-instruct-2501` | 0.05 / 0.08 |
| M066 | Llama 4 Maverick | 2025-04-05 | OpenRouter | `OPENROUTER_API_KEY` | `meta-llama/llama-4-maverick` | 0.1875 / 0.6525 |
| M090 | Gemini 2.5 Flash | 2025-05-20 | OpenRouter | `OPENROUTER_API_KEY` | `google/gemini-2.5-flash` | 0.3 / 2.5 |
| M102 | gpt-oss-120b | 2025-08-05 | OpenRouter | `OPENROUTER_API_KEY` | `openai/gpt-oss-120b` | 0.15 / 0.6 |
| M119 | Claude Haiku 4.5 | 2025-10-15 | Anthropic (direct) | `ANTHROPIC_API_KEY` | `claude-haiku-4-5-20251001` | 1 / 5 |
| M133 | GPT-5.2 | 2025-12-11 | OpenAI (direct) | `OPENAI_API_KEY` | `gpt-5.2` | 1.75 / 14 |
| M140 | Claude Opus 4.6 | 2026-02-05 | Anthropic (direct) | `ANTHROPIC_API_KEY` | `claude-opus-4-6` | 5 / 25 |
| M143 | GLM-5 | 2026-02-12 | OpenRouter | `OPENROUTER_API_KEY` | `z-ai/glm-5` | 0.6 / 1.92 |
| M171 | Qwen3.7-Max | 2026-05-20 | OpenRouter | `OPENROUTER_API_KEY` | `qwen/qwen3.7-max` | 1.475 / 4.425 |
| M183 | Qwen3.7-Flash | 2026-07-21 | OpenRouter | `OPENROUTER_API_KEY` | `qwen/qwen3.7-flash` | 0.03 / 0.13 |
| M190 | DeepSeek V4 Pro 0813 | 2026-08-13 | OpenRouter | `OPENROUTER_API_KEY` | `deepseek/deepseek-v4-pro-0813` | 1.32 / 3.96 |
| M196 | GLM-5.3-Flash | 2026-08-26 | OpenRouter | `OPENROUTER_API_KEY` | `z-ai/glm-5.3-flash` | 0.15 / 0.5 |
| M199 | Gemini 3.8 Flash | 2026-09-02 | OpenRouter | `OPENROUTER_API_KEY` | `google/gemini-3.8-flash` | 0.75 / 3.75 |
| M204 | GPT-6 Luna | 2026-09-22 | OpenAI (direct) | `OPENAI_API_KEY` | `gpt-6-luna` | 0.1 / 0.5 |
| M206 | Claude Sonnet 5.5 | 2026-09-28 | Anthropic (direct) | `ANTHROPIC_API_KEY` | `claude-sonnet-5-5` | 2 / 10 |
| M207 | GPT-6.1 Sol | 2026-09-29 | OpenAI (direct) | `OPENAI_API_KEY` | `gpt-6.1-sol` | 2 / 10 |

`--models panel20` runs these 20 models. The full registered table (`--models all`, 50 models) and the earlier model
table are also registered. Earlier runs depend on them, so they are kept unchanged.

## API keys and providers

Only `shelflife.cli run` calls models. It reads one environment variable per provider at call time. The code never
writes keys to run records, logs or the ledger.

| Provider | Environment variable | Endpoint | Panel models routed there |
|---|---|---|---|
| **Anthropic** (direct) | `ANTHROPIC_API_KEY` | `api.anthropic.com/v1/messages` | M119, M140, M206 |
| **OpenAI** (direct) | `OPENAI_API_KEY` | `api.openai.com/v1/chat/completions` | M015, M133, M204, M207 |
| **OpenRouter** | `OPENROUTER_API_KEY` | `openrouter.ai/api/v1/chat/completions` | the other 13 panel models |

No other providers are implemented. Open-weight models such as gpt-oss-120b and GLM-5.3-Flash are called through
OpenRouter, which picks the upstream host. Local inference servers (vLLM, Ollama) are not supported.

**Set your keys.** Export only the keys for the providers you will use:

```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
export OPENAI_API_KEY="your-openai-key"
export OPENROUTER_API_KEY="your-openrouter-key"
```

Or keep them in a `.env` file at the repository root. The package does not read `.env` itself, so load it
into your shell first:

```bash
cp .env.example .env          # then edit .env; it is ignored by git
set -a; source .env; set +a
```

**Choosing a provider and model.** You choose a *model*, not a provider. Each model ID (`M015`, `M206`, …) maps to
one route in [`code/shelflife/models.py`](../shelflife/models.py) as `(route, API model ID, $ per 1M input tokens,
$ per 1M output tokens, reasoning)`. Routes follow one rule: Anthropic direct first, then OpenAI direct, then
OpenRouter. To run a model that is not in the table, or to send a model through a different route, add or change an
entry there with its route, API model ID and price. Adding a model creates a new protocol, which the run records.
`--policy` selects how a request is adapted to each provider (for example caps for reasoning models and
parameters a route rejects); the default is chosen from the model.
