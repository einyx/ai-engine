# ai-engine

A Rust gateway for LLM APIs. Drop-in compatible with OpenAI and Anthropic SDKs;
also serves any OpenAI-compatible upstream — Ollama, vLLM, LM Studio, OpenRouter — out of the box.

**Status:** v0.2.1 — Streaming + concurrency. Stateless proxy with
a typed pipeline architecture, single-node Rust inference, and pipeline-parallel
distributed inference over QUIC. See `docs/superpowers/specs/` for the full design
and `docs/superpowers/plans/` for the implementation plan.

## v0.2.0 — Distributed inference + gateway (this release)

ai-engine v0.2.0 ships three things in one binary:

1. **Drop-in OpenAI / Anthropic / Ollama gateway** — proxy traffic to remote
   API providers with a typed pipeline of `Stage`s (auth, content policy,
   model routing, forwarding, logging). Original v0.1 functionality.

2. **Single-node Rust inference** — load any Llama-3-family safetensors
   checkpoint and serve it directly via burn (CPU / CUDA / Metal / WebGPU).
   Bytes-tolerant gate verifies logits match HF transformers to within 1e-3.

3. **Distributed pipeline-parallel inference** — partition a model across
   multiple nodes connected over QUIC with fingerprint-pinned TLS. The
   leader speaks HTTP; workers expose only `/healthz`. Each node loads its
   assigned layer range from a shared safetensors checkpoint. A 3-node
   loopback test verifies cluster output matches single-node baseline
   exactly under greedy sampling.

### Quickstart: standalone gateway

Same as v0.1. See `ai-engine.toml.example`.

### Quickstart: distributed cluster

On each node, write `ai-engine.toml` describing the cluster. The cert
fingerprint for each node is printed to stderr the first time the node
starts; copy it into the config.

```toml
[[cluster]]
id = "home"
leader = "node-a"
quic_bind = "0.0.0.0:7700"

[cluster.model]
id = "llama-3-70b"
config_path = "/srv/models/llama-3-70b/config.json"
weights_path = "/srv/models/llama-3-70b/model.safetensors"
tokenizer_path = "/srv/models/llama-3-70b/tokenizer.json"

[[cluster.node]]
id = "node-a"
addr = "192.168.1.10:7700"
cert_fingerprint = "sha256:..."
backend = "cuda"

[[cluster.node]]
id = "node-b"
addr = "192.168.1.11:7700"
cert_fingerprint = "sha256:..."
backend = "metal"

[[provider]]
id = "home-cluster"
kind = "local-cluster"
cluster = "home"

[[route]]
match = { model = "llama-3-70b" }
provider = "home-cluster"

[pipeline."/v1/chat/completions"]
stages = ["auth", "model_route", "forward", "log"]
```

On each node run:

```
./ai-engine --config ai-engine.toml --node-id <this-node-id>
```

Send a chat completion to the leader as if it were OpenAI:

```
curl http://node-a:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "llama-3-70b", "messages": [{"role": "user", "content": "hi"}]}'
```

### Known limitations in v0.2.0 (deferred to v0.3+)

- Single concurrent request per cluster (leader serializes via `&mut self`).
- Non-streaming responses only (no SSE chunks during generation).
- Static cluster membership — config-time only, no mDNS discovery.
- No automatic failover when a worker dies mid-request.
- bf16 / f16 / f32 only; no quantization (Q4/Q5/AWQ/GPTQ).
- Workers compute their own layer range via even-split; capability-aware
  partitioning is computed by the leader but Assignment isn't yet wired
  back to workers over QUIC for them to load weights accordingly.
- Pipeline-parallel only (no tensor parallelism).

## Why

| Axis | Existing gateways (Python) | ai-engine (Rust) |
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
cp ai-engine.toml.example ai-engine.toml
$EDITOR ai-engine.toml

# Run
OPENAI_API_KEY=sk-... ANTHROPIC_API_KEY=sk-ant-... AI_ENGINE_MASTER_KEY=mk-... \
  ./target/release/ai-engine --config ai-engine.toml

# Validate-and-exit
./target/release/ai-engine --check --config ai-engine.toml
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

ai-engine will not send an `Authorization` header to upstreams where no
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
- **Provider trait.** Lives in `ai-engine-provider`. Default methods return
  `Unsupported`, so a concrete provider implements only the methods it
  actually supports.

## Workspace

- `crates/ai-engine-core` — `Pipeline`, `Stage` trait, `RequestCtx`, errors
- `crates/ai-engine-provider` — `Provider` trait + wire types (OpenAI + Anthropic shapes)
- `crates/ai-engine-openai` — OpenAI provider (also serves Ollama, vLLM, etc.)
- `crates/ai-engine-anthropic` — Anthropic provider
- `crates/ai-engine-stages` — `auth`, `content_policy`, `model_route`, `forward`, `log`
- `crates/ai-engine-config` — TOML schema + `${ENV}` interpolation + validation
- `crates/ai-engine-http` — axum router + SSE encoding + error envelopes
- `crates/ai-engine` — binary; CLI parser, signal handling, hot reload

## Testing

```bash
# Unit + integration tests
cargo test --workspace

# Load smoke (release mode recommended)
cargo test --release -p ai-engine --test load_smoke -- --ignored --nocapture
```

The wire-compatibility tests in `crates/ai-engine/tests/wire_compat_*.rs`
hit ai-engine with the real `async-openai` SDK pointed at wiremock-backed
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

## Release history

### v0.2.0-alpha.1 — Single-node inference preview

ai-engine v0.2-alpha can load a Llama-3-family safetensors checkpoint
and run inference directly — no cluster yet. See the test fixture at
`crates/ai-engine-runtime/fixtures/toy-llama-3/` for the canonical example,
and the bytes-tolerant correctness gate in
`crates/ai-engine-runtime/tests/reference_logits.rs` that verifies the
burn-based forward pass matches HF transformers to within 1e-3.

Supported model families (safetensors layout):
- Llama 3.x
- Mistral 7B / Mistral Nemo
- Qwen 2.5
- DeepSeek V2 (dense portions only — no MoE in v0.2)

Backends compiled by default: CPU (ndarray) and WebGPU (covers Metal on macOS,
Vulkan on Linux). CUDA available behind the `backend-cuda` feature.

### v0.2.0-alpha.2 — Distributed inference preview

ai-engine v0.2-alpha.2 adds the `ai-engine-cluster` crate: a leader/worker
QUIC-based pipeline-parallel inference coordinator. A 3-node loopback test
in `crates/ai-engine-cluster/tests/inprocess_cluster.rs` verifies that the
cluster path produces logits matching the single-node baseline to within
1e-3 on the toy-llama-3 fixture.

Components:
- `ai-engine-cluster::tls` — self-signed ed25519 cert generation + SHA-256
  fingerprint pinning.
- `ai-engine-cluster::transport` — QUIC over rustls with ALPN
  `ai-engine-cluster/1`.
- `ai-engine-cluster::protocol` — control plane (postcard-framed) and
  data plane (length-prefixed activation frames) over QUIC streams.
- `ai-engine-cluster::partition` — capability-aware DP layer-cut solver
  with manual override and content-addressed manifests.
- `ai-engine-cluster::worker` / `::leader` — state machines.
- `ai-engine-cluster::provider` — implements the existing `Provider`
  trait so the gateway pipeline routes to the cluster without changes.

### v0.2.1 — Streaming + concurrency

ai-engine v0.2.1 closes three v0.2.0 gaps:

- **Per-token SSE streaming** on `/v1/chat/completions` when `stream: true`.
- **Concurrent requests on one leader.** Multiple in-flight chat completions
  are interleaved through the cluster via per-request bidi QUIC streams —
  no more serialization on `&mut self`.
- **Real partition Assignment** over QUIC. Workers wait for the leader's
  manifest before loading weights; partition policy lives entirely on the
  leader, including the optional `[[cluster.partition_override]]` blocks
  in TOML.

Updated known limitations (still deferred to v0.3+): mDNS auto-discovery,
dynamic worker membership, automatic failover, quantization, tensor
parallelism, web playground UI.

### v0.3.0-alpha — Q8 weight quantization

ai-engine v0.3.0-alpha adds Q8 (8-bit symmetric per-tensor) weight
quantization to `ai-engine-runtime`. Each Linear weight is stored as
int8 + an f32 scale; the forward pass dequantizes per call. Memory at
rest is ~2× smaller for the toy fixture (more for real models where
Linear weights dominate the parameter count).

Correctness:
- Q8 forward matches bf16 reference within ~0.023 on the random-init
  toy-llama-3 fixture. Argmax matches exactly. Real models with
  structured weights see substantially smaller per-op error.
- 3-node Q8 cluster generation matches single-node Q8 generation EXACTLY
  under greedy sampling — the QUIC wire serialization preserves Q8
  forward output byte-for-byte.

Generate a Q8 checkpoint from any bf16 safetensors model using the
`crates/ai-engine-runtime/scripts/generate_q8_fixture.py` template.

Known limitations:
- Q4 (4-bit packed) not supported — Plan 6.
- Dequantize-on-forward is unfused; specialized int8 GEMM would be
  substantially faster on GPU backends.
- Loader recognizes only our `<name>.scale` convention, not
  bitsandbytes `<name>.SCB` per-channel scales or AWQ/GPTQ layouts.
- Activations stay f32.

### v0.3.0-alpha.2 — Q4 weight quantization

ai-engine v0.3.0-alpha.2 adds Q4 (4-bit, per-group symmetric, group
size 32) weight quantization. Each Linear weight is stored as packed
nibbles (2 values per byte) plus per-group f32 scales. Memory at rest
is ~3.2× smaller than bf16 for the toy fixture (closer to 4× for
realistic models where Linear weights dominate parameter count).

Correctness:
- Q4 forward drift vs bf16 reference is ~0.28 on the random-init
  toy-llama-3 fixture — argmax does not match because the random toy
  has only 0.14 separation between its top-10 logits. Dispatch-parity
  test proves the Q4 matmul is bit-identical to dequantize-then-Dense
  matmul, so the drift is intrinsic per-group Q4 noise, not a bug.
  Trained models have substantially wider top-1 separation and would
  preserve argmax.
- 3-node Q4 cluster generation matches single-node Q4 generation
  EXACTLY under greedy sampling — the QUIC wire serialization
  preserves Q4 forward output bit-for-bit.

Format: our own per-group symmetric Q4, group size 32, low-nibble-first
packing. Stored pre-transposed (math order [in, out]) so the loader
never has to transpose Q4 weights at load time.

Generate a Q4 checkpoint from any bf16 safetensors model using
`crates/ai-engine-runtime/scripts/generate_q4_fixture.py`.

Known limitations (still deferred):
- External format readers (AWQ / GPTQ / GGUF / bitsandbytes NF4).
- Dequantize-on-forward is unfused; specialized int4 GEMM kernels
  would be substantially faster on GPU backends.
- Activations stay f32.
- Per-group symmetric only; no zero-point variants.

### v0.3.0-alpha.3 — GGUF Q4_0 reader

ai-engine v0.3.0-alpha.3 reads GGUF (the llama.cpp checkpoint format) directly.
Currently supports v3 files with the Q4_0 quantization type for Linear weights,
plus F32 / F16 / BF16 for boundary tensors (embeddings, layernorms).

Implementation:
- Native `LinearWeight::Q4Gguf` variant that preserves GGUF's exact block layout
  (32 weights per block, f16 scale + 16 bytes of biased nibbles, low half =
  block indices 0..16, high half = 16..32).
- GGUF→HF tensor name translation built in (`blk.N.attn_q.weight` → standard HF).
- `load_gguf` entry point alongside the existing safetensors loader.
- Toy fixture compresses 3.5× over bf16.

Use:
```
ai_engine_runtime::loader::load_gguf::<B>(path, &cfg, 0..cfg.n_layers, true, true, &dev)
```

Known limitations:
- Only Q4_0 + F32 + F16 + BF16 are decoded. Q4_1, Q4_K, Q5_*, Q6_K, Q8_0,
  IQ_* are deferred to Plan 8.
- The GGUF reader doesn't yet wire into the TOML config — there's no
  `model.gguf` path in `[cluster.model]`. Operators use `load_gguf` from
  code or extend `build_app_state` themselves. Plan 9 wires this.

### v0.3.0-alpha.4 — mDNS auto-discovery

ai-engine v0.3.0-alpha.4 lets cluster nodes find each other on the LAN
via mDNS. No more pasting cert fingerprints into every `[[cluster.node]]`
block.

How it works:
- Workers announce themselves on startup with TXT records: cluster_id,
  node_id, role=worker, protocol_version, fingerprint, backend.
- The leader, when `[[cluster.discover]]` is set, browses for
  `_ai-engine._tcp.local.` services and TOFU-pins the announced
  fingerprints.
- The existing static `[[cluster.node]]` path is unchanged.

Config:

```toml
[[cluster]]
id = "home-lab"
leader = "leader"
quic_bind = "0.0.0.0:7700"

[cluster.discover]
expected_workers = 2
timeout_secs = 30

[cluster.model]
id = "llama-3-70b"
# ...
```

Known limitations:
- TOFU only on first announcement; later contradictory announcements
  for the same node_id are ignored.
- Dynamic membership not supported — workers joining a running cluster
  still require restart.
- `cert_fingerprint` is still required on `[[cluster.node]]` entries
  even when `[[cluster.discover]]` is set (placeholder zeros suffice).
  Cleanup is a future TODO.
- mDNS multicast may be unavailable on some restricted networks /
  Docker setups. The `multiproc_smoke_mdns` test is `#[ignore]`d for
  portability.

### v0.3.0-alpha.5 — GGUF binary wiring

ai-engine v0.3.0-alpha.5 loads `.gguf` checkpoints through the binary
path. Just point `weights_path` at a GGUF file:

```toml
[cluster.model]
id = "llama-3-70b"
config_path = "/srv/models/llama-3-70b/config.json"
weights_path = "/srv/models/llama-3-70b/model.gguf"     # <-- .gguf, not .safetensors
tokenizer_path = "/srv/models/llama-3-70b/tokenizer.json"
```

The new `load_weights` function dispatches on file extension; everything
else (workers, leader, partitioning, generation) is unchanged.

Known limitations (still deferred):
- `config_path` + `tokenizer_path` still required even when the GGUF
  embeds them. Pulling these from GGUF metadata is a future cleanup.
- Only Q4_0 + F32 + F16 + BF16 GGUF tensor types decoded.

### v0.3.0-alpha.6 — GGUF self-describing checkpoints

ai-engine v0.3.0-alpha.6 drops the requirement for separate `config_path`
and `tokenizer_path` when `weights_path` is a `.gguf` file. The GGUF
metadata already carries both — extract them at load time:

```toml
[cluster.model]
id = "llama-3-70b"
weights_path = "/srv/models/llama-3-70b/model.gguf"
# config_path + tokenizer_path no longer required for GGUF
```

Internals:
- `ModelConfig::from_gguf_file` extracts hyperparams from `llama.*` keys.
- `load_tokenizer_from_gguf` rebuilds the HF tokenizer from
  `tokenizer.ggml.tokens` + `.merges` (Llama-3-style byte-level BPE).
- Both are dispatched automatically by `build_app_state` and the
  worker entrypoint when the corresponding TOML path is absent.

Known limitations:
- Only Llama-3-family (`general.architecture = "llama"`) supported.
- Only byte-level BPE tokenizers (`tokenizer.ggml.model = "gpt2"`/`"llama"`).
  SentencePiece-based GGUF tokenizers deferred.
- `tie_word_embeddings` defaults to `true` (the Llama-3 norm).
