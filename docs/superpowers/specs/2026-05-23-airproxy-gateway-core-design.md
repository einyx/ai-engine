# airproxy — Gateway Core (sub-project #1) design

**Status:** approved (brainstorming) — pending implementation plan
**Date:** 2026-05-23
**Scope:** Foundational sub-project of `airproxy`, a Rust reimplementation of the Relay AI gateway (https://github.com/geeper-io/relay).
**Author / driver:** alessio

---

## 1. Goals & non-goals

### Goals
`airproxy` aims to reach feature parity with Relay over time, while being measurably better on four axes (priority order when forced to trade): raw performance, operational simplicity, architectural clarity, and extensibility.

Sub-project #1 — **Gateway core** — ships a stateless, drop-in-compatible HTTP proxy for OpenAI and Anthropic that establishes the architectural foundation every later sub-project will plug into.

**In scope for v1 (this spec):**

- HTTP endpoints: `/v1/chat/completions`, `/v1/messages`, `/v1/embeddings`, `/v1/models`, `/healthz`, `/readyz`
- Upstream providers: OpenAI, Anthropic
- Streaming (SSE) for chat and messages, including mid-stream error handling
- Format-pinned routing (no cross-format translation): the OpenAI endpoint only routes to OpenAI-shape backends, ditto Anthropic
- TOML configuration with env-var interpolation and SIGHUP hot-reload
- Pipeline architecture with a stable `Stage` trait, runtime-configurable per route
- Provider abstraction via a `Provider` trait
- Auth (simple): passthrough mode and shared-master-key mode
- Content policy: request-size cap and regex-based prompt-injection blocking
- Observability: per-request JSON log line to stdout, `tracing` spans on stderr in dev mode
- Test suite: unit, provider mocks (`wiremock`), full-pipeline integration, wire-compat with real SDKs, load smoke test

### Non-goals for v1

- DB-backed persistence (no users, teams, sessions, request logs, cache, rate-limit state)
- SSO / OIDC
- Cross-format routing (`/v1/chat/completions` → Anthropic backend, or reverse)
- Rate limiting, budgets, response caching
- Model fallbacks, retries with backoff, circuit breakers
- PII scrubbing
- RAG / knowledge base
- Admin REST API
- Prometheus metrics
- Langfuse / external tracing exporters
- Helm chart, container image build, migrations
- Runtime plugin host (WASM / scripting)
- Providers beyond OpenAI and Anthropic (Azure / Bedrock / Gemini deferred)

These are deferred to sub-projects #2–#8 (stubbed in §11).

---

## 2. Why "better" — what we are actually improving on Relay

| Axis | Relay (Python) | `airproxy` (Rust) target |
|---|---|---|
| Per-request overhead | FastAPI + Python async; tens of ms baseline | Sub-millisecond pipeline overhead measured against a no-op upstream |
| Concurrent streaming sessions | GIL + ASGI worker model; throughput drops with parsed-event middleware | tokio + hyper; thousands of concurrent SSE streams on one process |
| Deploy footprint | Python interpreter + 200+ MB venv + Postgres requirement | Single static binary, no required external services in v1 |
| Architecture | Implicit pipeline: middleware chained inside each route handler in `app/api/v1/{chat,messages,embeddings}.py` | Explicit `Pipeline` of typed `Stage`s, configurable per route, with terminal-stage guarantees |
| Extensibility | Subclass / fork | Trait-based stages are additive — every future sub-project lands as new `Stage` implementations + config, never as edits to the pipeline machinery |

These are the design constraints that justify the decisions throughout this document.

---

## 3. Workspace layout

Cargo workspace rooted at `/home/alessio/aip/airproxy/`:

```
airproxy/
├── Cargo.toml                  # workspace root
├── crates/
│   ├── airproxy/               # binary; main.rs, CLI parser, signal handling
│   ├── airproxy-config/        # TOML schema (serde), env-var interpolation, validation, hot-reload
│   ├── airproxy-core/          # Pipeline, Stage trait, RequestCtx, GatewayError, GatewayResponse
│   ├── airproxy-http/          # axum app, route handlers, SSE encoding, error → HTTP mapping
│   ├── airproxy-provider/      # Provider trait, Capabilities, ProviderError, Credentials, CallCtx
│   ├── airproxy-openai/        # OpenAI provider impl + wire types (ChatRequest, etc.)
│   ├── airproxy-anthropic/     # Anthropic provider impl + wire types
│   └── airproxy-stages/        # Built-in stages: auth, content_policy, model_route, forward, log
├── docs/
│   └── superpowers/specs/      # design docs live here (this file)
├── tests/                      # workspace-level integration tests
└── airproxy.toml.example
```

**Rationale:**

- `airproxy-core` and `airproxy-provider` have minimal dependencies (no axum, no tokio runtime types, just `async-trait`, `serde`, `bytes`, `futures`). Provider crate authors and stage authors can depend on these without pulling in the HTTP server.
- Each provider lives in its own crate, independently publishable on crates.io once stable.
- The `airproxy` binary is thin — it parses CLI args, loads config, builds the pipeline graph, and runs `airproxy-http`.

---

## 4. Request lifecycle

A request flows through this lifecycle, illustrated for `/v1/chat/completions`:

```
HTTP request
   │
   ▼
[axum extractors]
   • parse body into RequestBody::OpenAiChat
   • extract headers, client IP, Authorization
   • mint request_id (uuidv7), capture started_at
   • build RequestCtx
   │
   ▼
[Pipeline.execute(&mut ctx)] — stages from config:
   1. AuthStage           validates bearer; sets ctx.identity
   2. ContentPolicyStage  enforces max_request_bytes; runs prompt-injection regexes
   3. ModelRouteStage     resolves ctx.body.model → ctx.binding (provider id + creds + upstream_model)
   4. ForwardStage        calls Provider::chat / chat_stream; fills ctx.response
   5. LogStage            TERMINAL — emits JSONL log line; always runs
   │
   ▼
[axum response]
   • ctx.response = Full(bytes)      → Json response
   • ctx.response = Stream(events)   → Sse response
   • ctx.error.is_some()             → GatewayError-mapped HTTP error (with OpenAI- or Anthropic-shaped envelope)
```

### Pipeline semantics

Stages return `Result<StageOutcome, StageError>`:

- `Ok(StageOutcome::Continue)` — pipeline proceeds to the next stage.
- `Ok(StageOutcome::Respond(resp))` — pipeline stops calling non-terminal stages, sets `ctx.response`, then runs remaining terminal stages and returns.
- `Err(e)` — `ctx.error = Some(e.into())`, pipeline stops non-terminal stages and runs remaining terminal stages.

**Terminal stages** (`Stage::is_terminal() -> true`) run on every request regardless of short-circuit or error. In v1 the only terminal stage is `LogStage`. This guarantees every request produces exactly one log line, populated with whatever state the pipeline reached.

Per-route pipelines are configured in TOML (§7). Each pipeline must contain `forward` and at least one terminal stage; this is validated at startup.

---

## 5. `Stage` trait & `RequestCtx`

```rust
// crates/airproxy-core/src/stage.rs

#[async_trait::async_trait]
pub trait Stage: Send + Sync + 'static {
    /// Stable identifier used in TOML config and log fields.
    fn name(&self) -> &'static str;

    /// Terminal stages run unconditionally after non-terminal stages
    /// have completed or short-circuited.
    fn is_terminal(&self) -> bool { false }

    async fn process(&self, ctx: &mut RequestCtx) -> Result<StageOutcome, StageError>;
}

pub enum StageOutcome {
    Continue,
    Respond(GatewayResponse),
}

pub struct StageError {
    pub stage: &'static str,
    pub error: GatewayError,
}
```

```rust
// crates/airproxy-core/src/ctx.rs

pub struct RequestCtx {
    pub request_id: Uuid,                       // uuidv7
    pub started_at: Instant,
    pub route: &'static str,                    // "/v1/chat/completions" etc.
    pub headers: HeaderMap,
    pub body: RequestBody,
    pub identity: Option<Identity>,             // set by AuthStage
    pub binding: Option<ProviderBinding>,       // set by ModelRouteStage
    pub response: ResponseSlot,                 // Pending | Full(Bytes) | Stream(BoxStream)
    pub error: Option<GatewayError>,
    pub metadata: HashMap<&'static str, serde_json::Value>,
}

pub enum RequestBody {
    OpenAiChat(openai::ChatRequest),
    AnthropicMessages(anthropic::MessagesRequest),
    OpenAiEmbeddings(openai::EmbeddingsRequest),
    Empty,                                      // for /models, /healthz, /readyz
}

pub enum ResponseSlot {
    Pending,
    Full(GatewayResponse),
    Stream(BoxStream<'static, Result<ProviderEvent, ProviderError>>),
}

pub enum Identity {
    /// Passthrough mode: the raw bearer is forwarded to the upstream provider unchanged.
    Anonymous { raw_bearer: String },
    /// Shared-key mode: the bearer matched a configured master key with this holder name.
    Holder { name: String },
}

pub struct ProviderBinding {
    pub provider_id: String,                    // "openai-prod"
    pub upstream_model: String,                 // possibly different from ctx.body.model
}
```

### Built-in stages (`crates/airproxy-stages`)

| Name (TOML id) | Type | Terminal | Responsibility |
|---|---|:---:|---|
| `auth` | `AuthStage` | no | Validate `Authorization` against config; populate `ctx.identity` or return `Err(Unauthorized)` |
| `content_policy` | `ContentPolicyStage` | no | Enforce `max_request_bytes` against the raw body length. Run prompt-injection regexes against: `RequestBody::OpenAiChat` → every string in `messages[*].content` (string or array-part), plus `system`; `RequestBody::AnthropicMessages` → every string in `messages[*].content`, plus `system`; `RequestBody::OpenAiEmbeddings` → `input` (string or list of strings); other bodies skipped |
| `model_route` | `ModelRouteStage` | no | Resolve `ctx.body.model` against the route table (top-to-bottom, first match wins); populate `ctx.binding`; return `Err(NoRouteForModel)` on miss |
| `forward` | `ForwardStage` | no | Look up the `Provider` for `ctx.binding.provider_id`; dispatch to `chat` / `chat_stream` / `messages` / `messages_stream` / `embeddings` based on `ctx.body` and the request's `stream` flag; fill `ctx.response` |
| `log` | `LogStage` | **yes** | Emit a single JSON line to stdout summarizing the request (see §9) |

Each stage is implemented as a small struct, constructed from config, registered in a `StageRegistry` keyed by stable id. The `Pipeline` for a route is a `Vec<Arc<dyn Stage>>` resolved at startup.

### Auth modes in v1

`AuthStage` supports two modes via config:

- **`passthrough`**: the `Authorization` header is forwarded verbatim to the upstream provider. `Identity::Anonymous { raw_bearer: Some(...) }`. No validation.
- **`shared-key`**: the bearer must exactly match the `key` field of one of the entries in `[auth].master_keys` (each entry is a TOML table with `key` and `name`). On match, `Identity::Holder { name }` is populated using the entry's `name`. On mismatch, `Err(Unauthorized)`.

Full SSO and DB-backed key management arrive with sub-project #2.

---

## 6. `Provider` trait & upstream model

```rust
// crates/airproxy-provider/src/lib.rs

#[async_trait::async_trait]
pub trait Provider: Send + Sync + 'static {
    fn id(&self) -> &str;                             // matches [[provider]].id in config
    fn kind(&self) -> &'static str;                   // "openai" | "anthropic"
    fn capabilities(&self) -> Capabilities;

    async fn chat(
        &self,
        req: openai::ChatRequest,
        creds: &Credentials,
        ctx: &CallCtx,
    ) -> Result<openai::ChatResponse, ProviderError>;

    async fn chat_stream(
        &self,
        req: openai::ChatRequest,
        creds: &Credentials,
        ctx: &CallCtx,
    ) -> Result<BoxStream<'static, Result<openai::ChatStreamEvent, ProviderError>>, ProviderError>;

    async fn messages(
        &self,
        req: anthropic::MessagesRequest,
        creds: &Credentials,
        ctx: &CallCtx,
    ) -> Result<anthropic::MessagesResponse, ProviderError>;

    async fn messages_stream(
        &self,
        req: anthropic::MessagesRequest,
        creds: &Credentials,
        ctx: &CallCtx,
    ) -> Result<BoxStream<'static, Result<anthropic::MessagesEvent, ProviderError>>, ProviderError>;

    async fn embeddings(
        &self,
        req: openai::EmbeddingsRequest,
        creds: &Credentials,
        ctx: &CallCtx,
    ) -> Result<openai::EmbeddingsResponse, ProviderError>;
}

pub struct Capabilities {
    pub chat: bool,
    pub messages: bool,
    pub embeddings: bool,
    pub streaming: bool,
    pub tools: bool,
    pub vision: bool,
}

pub struct CallCtx {
    pub request_id: Uuid,
    pub deadline: Option<Instant>,
    pub upstream_model: String,                       // already substituted from binding
}
```

### Credentials sourcing in v1

In v1 each `[[provider]]` block carries its own `api_key`. The `Provider` instance constructed at startup holds those credentials, and `ForwardStage` passes them into the trait method as the `creds` argument (`Credentials::from_provider_default(&provider)`). The `creds` parameter on the trait method is forward-compat: when sub-project #2 introduces per-user / per-team BYOK keys, `ForwardStage` will choose credentials per request based on `ctx.identity`. No trait change is required.

### Format-pinning

The `ChatRequest` / `ChatResponse` types are the OpenAI wire shapes verbatim. `MessagesRequest` / `MessagesResponse` are the Anthropic wire shapes verbatim. There is **no unified `Message` type in v1.**

Because routing is format-pinned, in practice `Provider::chat` is only ever called for backends whose `kind = "openai"`, and `Provider::messages` only for `kind = "anthropic"`. Methods a provider does not support return `Err(ProviderError::Unsupported)`. The router validates that no route binds a chat-format model to an Anthropic-kind provider (and vice versa) at startup.

Splitting into two traits (`OpenAiProvider`, `AnthropicProvider`) was considered and rejected: one trait gives a single registry and simpler `ForwardStage` dispatch; the runtime `Unsupported` cost is paid at startup-validation time and never on the hot path.

### Streaming details

- Each provider uses a long-lived `reqwest::Client` built from `[[provider]]` config, with HTTP/2 enabled and a per-provider connection pool.
- `*_stream` methods return `BoxStream` over the *parsed event type* for that protocol. SSE framing is decoded once at the provider boundary; downstream stages see typed events.
- `ForwardStage` deposits the stream into `ctx.response` as `ResponseSlot::Stream(...)`. `airproxy-http` re-encodes the stream as SSE on the way out.
- For events where no stage has mutated the payload, the original byte payload is re-emitted unchanged (token-perfect passthrough). Stages that do modify events do so via stream combinators (`.map`) — never by materializing the whole stream.
- Backpressure is enforced by axum's `Sse` responder and the bounded mpsc buffer between upstream and client (default 64 events).

### Error handling on streams

- If `Provider::*_stream` fails **before** the first event reaches the client, `ForwardStage` returns `Err`, and the normal HTTP error path produces a JSON error envelope.
- If the stream fails **mid-flight** (after bytes have been written to the client), `airproxy-http` emits a final SSE `event: error\ndata: {...}\n\n` chunk and closes the connection. No retries; resilience features arrive in sub-project #5.

---

## 7. Configuration

`airproxy.toml`:

```toml
[server]
bind = "0.0.0.0:8080"
shutdown_grace_secs = 30
log_format = "json"                                # "json" | "pretty"
log_level = "info"

[auth]
mode = "shared-key"                                # "passthrough" | "shared-key"
master_keys = [
  { key = "${AIRPROXY_MASTER_KEY}", name = "default" },
]

[content_policy]
max_request_bytes = 1_048_576
prompt_injection_patterns = [
  "ignore (all )?previous instructions",
]

# --- Providers: how to talk to upstream ---
[[provider]]
id = "openai-prod"
kind = "openai"
base_url = "https://api.openai.com/v1"
api_key = "${OPENAI_API_KEY}"
timeout_secs = 120
http2 = true

[[provider]]
id = "anthropic-prod"
kind = "anthropic"
base_url = "https://api.anthropic.com"
api_key = "${ANTHROPIC_API_KEY}"
timeout_secs = 120

# --- Routes: which model names go where ---
[[route]]
match = { model = "gpt-4o" }
provider = "openai-prod"
upstream_model = "gpt-4o-2024-08-06"

[[route]]
match = { model = "gpt-*" }                        # glob; first match wins
provider = "openai-prod"

[[route]]
match = { model = "claude-*" }
provider = "anthropic-prod"

# --- Pipelines: which stages run per endpoint ---
[pipeline."/v1/chat/completions"]
stages = ["auth", "content_policy", "model_route", "forward", "log"]

[pipeline."/v1/messages"]
stages = ["auth", "content_policy", "model_route", "forward", "log"]

[pipeline."/v1/embeddings"]
stages = ["auth", "model_route", "forward", "log"]
```

### Resolution & validation rules

- `${VAR}` interpolation against process env at load time. Missing env var → fatal startup error.
- Routes match top-to-bottom; first match wins.
- Each pipeline must:
  - Reference only known stage ids registered in `StageRegistry`.
  - Contain exactly one `forward` stage.
  - Contain at least one terminal stage (i.e., a stage whose `is_terminal()` returns true).
- Each route's `provider` must reference a known `[[provider]]` id, and the provider's `kind` must be compatible with the endpoint the route is reachable from (chat/embeddings → `openai` kind; messages → `anthropic` kind). Mismatches are startup errors.
- SIGHUP triggers a config reload: the new file is parsed and validated in full; on success the new pipeline graph and provider clients atomically replace the old ones; on failure the old config keeps running and a warning is emitted. The `[server].bind` field is not hot-reloadable.

---

## 8. Error model

```rust
// crates/airproxy-core/src/error.rs

pub enum GatewayError {
    BadRequest(String),
    Unauthorized,
    PayloadTooLarge,
    NoRouteForModel { model: String },
    Provider(ProviderError),
    Internal(anyhow::Error),
}

pub enum ProviderError {
    Connect(reqwest::Error),
    Timeout,
    Status { status: u16, body: Bytes },
    InvalidResponse(serde_json::Error),
    Stream(io::Error),
    Unsupported,
}
```

### HTTP mapping

| `GatewayError` | HTTP status | Body shape |
|---|:---:|---|
| `BadRequest` | 400 | Endpoint-native error envelope |
| `Unauthorized` | 401 | Endpoint-native error envelope |
| `PayloadTooLarge` | 413 | Endpoint-native error envelope |
| `NoRouteForModel` | 502 | Endpoint-native error envelope |
| `Provider(Connect)` | 502 | … |
| `Provider(Timeout)` | 504 | … |
| `Provider(Status { status, body })` | `status` | Pass through upstream body |
| `Provider(InvalidResponse)` | 502 | … |
| `Provider(Stream)` | mid-stream SSE `event: error` | n/a (already streaming) |
| `Provider(Unsupported)` | 502 | … (also a startup-validation failure normally) |
| `Internal` | 500 | … |

"Endpoint-native error envelope" means: requests to OpenAI-shaped endpoints get OpenAI's `{"error": {"message": ..., "type": ..., "code": ...}}` shape; requests to `/v1/messages` get Anthropic's `{"type": "error", "error": {...}}` shape. This preserves SDK compatibility.

---

## 9. Observability (v1 — stdout only)

The `LogStage` emits exactly one JSON object per request, one line:

```json
{
  "ts": "2026-05-23T12:00:01.234Z",
  "request_id": "01939abc-...",
  "route": "/v1/chat/completions",
  "model": "gpt-4o",
  "provider": "openai-prod",
  "upstream_model": "gpt-4o-2024-08-06",
  "status": 200,
  "duration_ms": 2841,
  "ttfb_ms": 312,
  "stream": true,
  "tokens": { "prompt": 523, "completion": 417, "total": 940 },
  "identity": "default",
  "error": null
}
```

Token counts come from:
- Non-streaming responses: the upstream `usage` field, parsed before logging.
- Streaming responses: OpenAI emits a final chunk with `usage` (when `stream_options.include_usage = true`); Anthropic emits `message_delta` events with `usage`. `LogStage` reads from `ctx.metadata["usage"]`, which `ForwardStage` populates by tapping the stream.

`tracing` spans (one per stage, one per upstream call) are emitted on stderr in dev mode via `tracing-subscriber` with a pretty format. In `log_format = "json"` mode, tracing is also emitted as JSON for ingestion by log shippers.

Prometheus metrics, DB-backed log persistence, and external tracing exporters arrive with sub-project #4.

---

## 10. Testing strategy

| Layer | Tooling | Scope |
|---|---|---|
| Stage unit tests | `#[tokio::test]` | Each stage exercised against synthetic `RequestCtx` instances. Covers success, short-circuit, error paths. |
| Provider unit tests | `wiremock` | Each provider exercised against recorded fixtures (sanitized golden request/response pairs from real upstreams). Covers streaming, non-streaming, tool use, error responses, malformed responses. |
| Pipeline integration | `axum::Router::oneshot` | Full pipeline exercised in-process with a mocked `Provider`. Covers auth failure, model-not-routed, content-policy block, streaming success, mid-stream error, short-circuit ordering, terminal-stage guarantee. |
| Wire compatibility | Real `openai` / `anthropic` SDKs in CI | The actual SDKs pointed at a running `airproxy` whose upstream is a `wiremock` server. Verifies our response bytes are parseable by real SDKs — the only test that catches subtle SSE-framing or field-naming bugs. |
| Load smoke | `oha` or `vegeta` in CI | 500 concurrent streaming connections for 60 s against a mock upstream; assertions on TTFB p99, no memory growth, no dropped events. |

Fixtures live in `crates/airproxy-openai/fixtures/` and `crates/airproxy-anthropic/fixtures/`. The wire-compat tests are the most important gate on "drop-in compatible."

---

## 11. Sub-project stubs (#2–#8)

Each lands as additive stages, provider variants, or sidecar binaries — none requires changes to `Pipeline`, `Stage`, or `Provider` shape.

- **#2 — Auth & keys.** Replaces v1's `passthrough` / `shared-key` modes with DB-backed users, teams, API keys (`ap-` prefix), and OIDC SSO for human login. Introduces the storage abstraction (SQLite default, Postgres feature-flagged). `AuthStage` grows; the trait does not.
- **#3 — Limits & quotas.** Adds `RateLimitStage`, `BudgetStage`, `CacheStage`. In-memory + Redis backends. Per-user, per-team, per-model budgets in tokens and requests. Cache hits short-circuit to `Respond(cached)` before `forward`.
- **#4 — Observability.** Adds `MetricsStage` and a `/metrics` endpoint, request-log persistence via the storage layer, Langfuse exporter, usage rollup queries. `LogStage` gains an optional DB sink alongside stdout.
- **#5 — Resilience.** Adds `FallbackStage` (retry on a backup model on 5xx or context overflow), upstream retries with jitter, expanded content-policy rule sets, per-provider circuit breakers.
- **#6 — PII scrubber.** Adds `PiiScrubStage` (pre-forward) and `PiiRestoreStage` (post-forward, terminal). Regex patterns plus ONNX-based NER (English first), shipped in an `airproxy-pii-onnx` crate. Placeholder restoration in both full and streaming responses.
- **#7 — Knowledge base / RAG.** Adds `RagStage` (injects retrieved context before `model_route`). Vector store: Qdrant. Tree-sitter chunking for 15+ languages. GitHub / GitLab sync runs in a separate `airproxy-kb-sync` binary that writes to the same Qdrant.
- **#8 — Admin API & ops.** Admin REST (`/admin/users`, `/admin/usage`, `/admin/keys`), Helm chart, container build, migrations CLI (`airproxy migrate`).

---

## 12. Open questions

None blocking implementation. Items deferred by design (cross-format translation, additional providers, plugin host) are intentionally out of scope and will be re-opened in their respective sub-project specs.

---

## 13. Approval

- Brainstorming sections 1–7 reviewed and approved by user 2026-05-23.
- Pending: user review of this written spec, then transition to implementation plan via the `writing-plans` skill.
