# CandleProvider — Native-Quantized Local GPU Inference

**Date:** 2026-05-24
**Status:** Design approved, pending implementation plan

## Problem

ai-engine decode runs at 0.338 tok/s on CPU for Llama-3.2-1B Q4_0. Six GPU
probes established that every burn GPU backend (wgpu, cuda, candle-via-burn)
dequantizes Q4_0 weights to f32 because burn has no native quantized-matmul
abstraction. The f32 inflation (0.7 GiB packed → 4.6 GiB) makes the workload
memory-bandwidth-bound and OOMs wgpu (f32 weights + parallel-stream scratch
exceed 12 GiB VRAM).

`candle-core` (HuggingFace's ML library) has native Q4_0/Q4_K/Q6_K matmul
kernels for CUDA/Metal/CPU that operate on packed quantized blocks without f32
dequantization. `candle-transformers::models::quantized_llama` is a complete
GGUF-loaded quantized Llama-3 implementation (correct GQA, `rope_freq_base`,
interleaved RoPE — matching the q/k permutation ai-engine had to fix manually
for its burn path). This gives native-quantized GPU inference (realistic
50-150 tok/s for a 1B model) by reusing proven kernels rather than writing or
forking any.

## Goal

Add a `CandleProvider` — a single-node, GPU-capable, native-quantized inference
provider — as a new provider `kind` alongside the existing burn-based
distributed cluster. Strictly additive: the burn cluster is unchanged and
remains the path for multi-node distribution. candle is the fast local path.

## Non-Goals

- Distributed/multi-node candle inference (the burn cluster owns distribution;
  candle-transformers has no layer-sharding concept).
- Non-Llama architectures (Qwen, Mistral, etc.) — deferred; v1 is Llama-3 family
  via `quantized_llama`.
- Sharing model weights across replicas (candle-transformers bundles
  weights+KV-cache with no sharing API; replicas load independently).
- Making candle part of the default build (its CUDA deps require `nvcc`).

## Architecture

### New crate: `ai-engine-candle`

A dedicated crate isolates candle's heavy, platform-specific dependencies
(candle-core, candle-transformers, optional CUDA/Metal) behind a Cargo feature
so the default workspace build stays burn-only and needs no CUDA toolkit.

```
crates/ai-engine-candle/
  Cargo.toml          # candle-core, candle-transformers; feature flags for device families
  src/
    lib.rs            # crate root, re-exports CandleProvider
    provider.rs       # CandleProvider: implements ai_engine_provider::Provider
    device.rs         # device auto-detection (CUDA > Metal > CPU) + TOML override parsing
    model.rs          # thin wrapper over quantized_llama::ModelWeights + per-replica state
    pool.rs           # replica pool (Vec<Mutex<...>>) + acquire-free-replica logic
```

### Feature flags

In `ai-engine-candle/Cargo.toml`:
- `cpu` (default) — candle CPU backend, no CUDA toolkit needed
- `cuda` — candle CUDA backend (requires `nvcc` at build time)
- `metal` — candle Metal backend (macOS)

In the top-level `ai-engine` binary crate: a `backend-candle` feature that
pulls in `ai-engine-candle`. Off by default. Building with GPU support is an
explicit opt-in: `cargo build --features backend-candle,ai-engine-candle/cuda`.

### Provider registration

`CandleProvider` implements `ai_engine_provider::provider::Provider`:
- `kind()` → `"candle-local"`
- `id()` → the model id from config
- `capabilities()` → `chat: true, streaming: true`, everything else false
  (matches the cluster provider's capability surface; messages/embeddings/
  tools/vision out of scope)
- `chat(req, creds, ctx)` and `chat_stream(req, creds, ctx)` as below

`crates/ai-engine/src/app.rs` gains a match arm for `kind = "candle-local"`
that constructs a `CandleProvider` (gated behind `#[cfg(feature =
"backend-candle")]`; when the feature is off, the arm returns a clear
configuration error telling the user to rebuild with the feature).

## Configuration

Extends the existing model-entry TOML pattern. A model with
`kind = "candle-local"`:

```toml
[[model]]
id = "llama-3.2-1b-gpu"
kind = "candle-local"
weights_path = "/srv/models/Llama-3.2-1B-Instruct-Q4_0.gguf"
device = "auto"        # auto | cpu | cuda:0 | cuda:N | metal
pool_size = 2          # number of model replicas (default 2)
```

Schema additions in `ai-engine-config`:
- The model/provider entry enum gains a `CandleLocal` variant (or the existing
  struct gains `device: Option<String>` and `pool_size: Option<usize>` fields
  used only when `kind == "candle-local"`).
- `device` parsing: `"auto"` (default) → try CUDA(0), then Metal, then CPU.
  `"cpu"` → CPU. `"cuda:N"` → CUDA device N. `"metal"` → Metal. Invalid →
  config validation error.
- `pool_size` default 2; must be ≥ 1.

Validation rules (in `ai-engine-config/src/validate.rs`):
- `candle-local` requires `weights_path` to exist and end in `.gguf`.
- `pool_size >= 1`.
- `device` matches one of the accepted forms.

## Components

### `device.rs` — device resolution

`fn resolve_device(spec: &str) -> anyhow::Result<candle_core::Device>`:
- `"auto"`: attempt `Device::cuda_if_available(0)`; if that yields CPU and
  Metal is compiled, try Metal; else CPU. Log which device was selected.
- `"cpu"` → `Device::Cpu`
- `"cuda:N"` → `Device::new_cuda(N)`
- `"metal"` → `Device::new_metal(0)`
- Unknown → error.

Device-family availability is gated by the crate's `cuda`/`metal` features;
requesting `cuda:0` without the `cuda` feature is a clear runtime error.

### `model.rs` — single-replica wrapper

`struct CandleModel { weights: quantized_llama::ModelWeights, tokenizer:
Arc<HfTokenizer>, device: Device, eos_token_id: u32 }`.

- `CandleModel::load(gguf_path, device, tokenizer) -> Result<Self>`: open the
  GGUF via `candle_core::quantized::gguf_file::Content::read`, build
  `ModelWeights::from_gguf(content, &mut reader, &device)`. Extract the EOS
  token id from GGUF metadata (`tokenizer.ggml.eos_token_id`).
- `CandleModel::generate(&mut self, prompt_tokens, params) -> impl Iterator /
  callback`: the autoregressive loop. `forward(input, pos)` → logits →
  `ai_engine_runtime::sample::sample(logits, params)` → next token. Maintains
  `index_pos`. Stops on EOS or `max_tokens`. Yields tokens as they are produced
  (for streaming) and accumulates for the non-streaming path.

The tokenizer is loaded once (shared `Arc` across replicas — it is immutable
and `Send + Sync`). Only the candle `ModelWeights` (with its internal KV cache)
is per-replica.

### `pool.rs` — replica pool

`struct ReplicaPool { replicas: Vec<tokio::sync::Mutex<CandleModel>> }`.

- `ReplicaPool::new(gguf_path, device, tokenizer, n) -> Result<Self>`: load the
  GGUF `n` times into `n` `CandleModel`s. (Independent loads; weights not
  shared.)
- `ReplicaPool::acquire(&self) -> MutexGuard<CandleModel>`: iterate replicas,
  `try_lock` each; return the first that succeeds. If all are busy, `await` on
  the lock of replica `request_index % n` (round-robin fallback so requests
  don't all pile on replica 0).

KV cache reset: candle's `ModelWeights` accumulates KV state across `forward`
calls. Each generation must start from a clean cache. `quantized_llama`
exposes cache state per-instance; the wrapper resets/recreates it at the start
of each `generate` (verify the exact reset API during implementation — if
`ModelWeights` has no public cache reset, recreate the replica's `ModelWeights`
from the cached GGUF content, or track `index_pos` from 0 and rely on causal
masking; the implementation plan must pin this down).

### `provider.rs` — the Provider impl

`struct CandleProvider { id: String, pool: ReplicaPool }`.

- `chat`: convert `ChatRequest` messages → a single prompt string (reuse the
  same chat-templating the cluster provider uses, or a minimal Llama-3 chat
  template). Tokenize, `pool.acquire()`, `generate` to completion, build a
  `ChatResponse` with usage (`prompt_tokens`, `completion_tokens`).
- `chat_stream`: same, but return a stream that emits OpenAI-style SSE chunks
  per generated token. On completion emit the final `[DONE]`-equivalent per the
  existing streaming convention in `ai-engine-provider`.

Concurrency: a request holds its replica's mutex guard for the duration of the
generation. With `pool_size = N`, up to N generations run concurrently; the
(N+1)th awaits a free replica.

## Data Flow

```
HTTP request → gateway → pipeline stages (auth, model_route) →
  Provider::chat(req) on CandleProvider →
    pool.acquire() → CandleModel::generate →
      loop: ModelWeights::forward (native Q4 matmul on GPU) → sample → token
    → detokenize → ChatResponse → wire types → HTTP response
```

Streaming replaces the final accumulation with per-token SSE emission.

## Error Handling

- GGUF load failure (missing file, bad format, non-Llama arch): surfaced at
  provider construction (startup), not per-request. Fail fast with a clear
  message.
- Device-unavailable (e.g. `cuda:0` requested, no CUDA): error at provider
  construction.
- Per-request errors (tokenization failure, forward-pass panic): mapped to
  `ProviderError`, returned as an HTTP error, replica released (mutex guard
  dropped on the error path).
- Feature-off: the `app.rs` match arm for `candle-local` without the
  `backend-candle` feature returns a config error directing the user to rebuild.

## Testing

- **Unit** (`ai-engine-config`): parse a TOML with `kind = "candle-local"`,
  assert device/pool_size defaults and validation rules (bad device string,
  pool_size 0, non-.gguf weights_path).
- **Unit** (`ai-engine-candle`, CPU feature): `resolve_device` for each spec
  form; pool `acquire` returns distinct replicas under concurrent access.
- **Integration** (env-gated, `#[ignore]`, mirrors `real_model_smoke`): with
  `AI_ENGINE_REAL_GGUF` set, load the real Llama-3.2-1B Q4_0 via CandleProvider
  (CPU device for CI determinism; CUDA when available), run a chat completion
  for "Hello, who are you?", assert non-empty coherent output, and print
  tok/s. A second variant runs on CUDA if the `cuda` feature + hardware are
  present and reports the GPU tok/s for the perf claim.

## Open Implementation Details (to pin down in the plan)

1. Exact `quantized_llama::ModelWeights` KV-cache reset API between
   generations (recreate vs reset vs index_pos tracking).
2. Whether the tokenizer comes from the GGUF (`load_tokenizer_from_gguf`,
   already in `ai-engine-runtime`) or a sidecar `tokenizer.json`. Prefer
   GGUF-embedded for the self-describing-checkpoint story established in
   v0.3.0-alpha.6.
3. The chat-template applied to messages before tokenization (Llama-3 instruct
   format vs raw concatenation) — match whatever the cluster provider does for
   consistency.

## Effort

5-8 days. Most of the work is the `CandleProvider`/pool/device glue and the
config plumbing; the transformer itself is candle-transformers'.
