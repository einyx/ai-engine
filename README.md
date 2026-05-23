# airproxy

A Rust gateway for LLM APIs. Drop-in compatible with OpenAI and Anthropic SDKs;
also serves any OpenAI-compatible upstream — Ollama, vLLM, LM Studio, OpenRouter — out of the box.

**Status:** v0.1.0 — Gateway Core (sub-project #1). Stateless proxy with
a typed pipeline architecture. See `docs/superpowers/specs/` for the full design
and `docs/superpowers/plans/` for the implementation plan.

## Why

| Axis | Existing gateways (Python) | airproxy (Rust) |
|---|---|---|
| Per-request overhead | Tens of ms baseline | Sub-millisecond pipeline overhead |
| Concurrent streams | GIL-bound; throughput collapses with middleware | tokio + hyper; thousands of SSE streams on one process |
| Deploy footprint | Interpreter + venv + required DB | Single static binary, no external services in v1 |
| Extension model | Subclass / fork | Trait-based `Stage`s, additive, configured from TOML |

## Features (v1)

- HTTP endpoints: `/v1/chat/completions`, `/v1/messages`, `/v1/embeddings`,
  `/v1/models`, `/healthz`, `/readyz`.
- Upstreams: OpenAI, Anthropic, and any OpenAI-compatible server (Ollama,
  vLLM, LM Studio, OpenRouter — pick `kind = "openai"` and set `base_url`).
- Streaming (SSE) with mid-stream error envelopes.
- Format-pinned routing — `/v1/chat/completions` only routes to OpenAI-shape
  backends; `/v1/messages` only to Anthropic.
- TOML configuration with `${ENV}` interpolation and SIGHUP hot-reload.
- Pipeline architecture with five built-in stages (auth, content_policy,
  model_route, forward, log) — runtime-configurable per route.
- Auth: passthrough and shared-master-key modes.
- Content policy: max request size + regex prompt-injection blocking.
- Observability: one JSON log line per request to stdout.
- Tests: unit, provider mocks (wiremock), wire-compat with real SDKs, load smoke.

## Quickstart

```bash
# Build
cargo build --release

# Generate a config from the example
cp airproxy.toml.example airproxy.toml
$EDITOR airproxy.toml

# Run
OPENAI_API_KEY=sk-... ANTHROPIC_API_KEY=sk-ant-... AIRPROXY_MASTER_KEY=mk-... \
  ./target/release/airproxy --config airproxy.toml

# Validate-and-exit
./target/release/airproxy --check --config airproxy.toml
```

Point any OpenAI SDK at `http://localhost:8080/v1` with the master key:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer mk-..." \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}'
```

## Using with Ollama

Ollama exposes an OpenAI-compatible API at `http://localhost:11434/v1` and
requires no authentication. Configure it as any other OpenAI-kind provider,
just omit the `api_key`:

```toml
[[provider]]
id = "ollama-local"
kind = "openai"
base_url = "http://localhost:11434/v1"
timeout_secs = 600   # local models can be slow
http2 = false

[[route]]
match = { model = "llama3*" }
provider = "ollama-local"
```

airproxy will not send an `Authorization` header to upstreams where no
`api_key` is configured and no inbound bearer is forwarded — exactly what
local OpenAI-compatible servers expect.

The same pattern works for vLLM, LM Studio, llama.cpp's OpenAI shim, and
OpenRouter (which does accept a key — just set `api_key` for that one).

## Architecture

```
HTTP request
   │
   ▼
[axum extractors]  →  RequestCtx
   │
   ▼
[Pipeline.execute(&mut ctx)]
   1. AuthStage           → validates bearer; sets ctx.identity
   2. ContentPolicyStage  → max_request_bytes + injection regexes
   3. ModelRouteStage     → ctx.body.model → ctx.binding
   4. ForwardStage        → Provider::chat / chat_stream / messages / …
   5. LogStage  [terminal] → one JSONL line to stdout
   │
   ▼
[axum response]   → JSON or SSE
```

- **Pipeline semantics.** Stages return `Continue`, `Respond(r)`, or `Err(e)`.
  Short-circuits skip remaining non-terminal stages but terminal stages
  (those returning `is_terminal() = true`) always run. This is what makes
  every request produce exactly one log line.
- **Format-pinning.** OpenAI-shape endpoints route to `kind = "openai"`
  providers; `/v1/messages` routes to `kind = "anthropic"`. Cross-format
  translation is intentionally out of v1 scope.
- **Provider trait.** Lives in `airproxy-provider`. Default methods return
  `Unsupported`, so a concrete provider implements only the methods it
  actually supports.

## Workspace

- `crates/airproxy-core` — `Pipeline`, `Stage` trait, `RequestCtx`, errors
- `crates/airproxy-provider` — `Provider` trait + wire types (OpenAI + Anthropic shapes)
- `crates/airproxy-openai` — OpenAI provider (also serves Ollama, vLLM, etc.)
- `crates/airproxy-anthropic` — Anthropic provider
- `crates/airproxy-stages` — `auth`, `content_policy`, `model_route`, `forward`, `log`
- `crates/airproxy-config` — TOML schema + `${ENV}` interpolation + validation
- `crates/airproxy-http` — axum router + SSE encoding + error envelopes
- `crates/airproxy` — binary; CLI parser, signal handling, hot reload

## Testing

```bash
# Unit + integration tests
cargo test --workspace

# Load smoke (release mode recommended)
cargo test --release -p airproxy --test load_smoke -- --ignored --nocapture
```

The wire-compatibility tests in `crates/airproxy/tests/wire_compat_*.rs`
hit airproxy with the real `async-openai` SDK pointed at wiremock-backed
upstreams. They are the canonical "drop-in compatible" gate.

## Roadmap

Future sub-projects each land as additive `Stage`s + config — never as
edits to the pipeline machinery:

- **#2** — DB-backed auth (users, teams, keys, OIDC).
- **#3** — Rate limits, budgets, response cache.
- **#4** — Prometheus, request-log persistence, Langfuse exporter.
- **#5** — Fallbacks, retries with backoff, circuit breakers.
- **#6** — PII scrubbing (regex + ONNX NER).
- **#7** — RAG / knowledge base over Qdrant.
- **#8** — Admin REST API, Helm chart, container, migrations CLI.

## License

Apache-2.0.
