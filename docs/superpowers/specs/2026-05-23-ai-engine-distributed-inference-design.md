# ai-engine — Distributed Inference Coordinator (sub-project #2) design

**Status:** approved (brainstorming) — pending implementation plan
**Date:** 2026-05-23
**Scope:** Sub-project #2 of `ai-engine` (formerly `airproxy`). v0.2 turns the v0.1 gateway into a real inference engine that distributes model execution across a user-defined cluster of nodes, while preserving every v0.1 gateway feature.
**Author / driver:** alessio
**License:** Apache-2.0

---

## 1. Goals & non-goals

### Goals

`ai-engine` v0.2 ships sub-project #2 — **Distributed Inference Coordinator**. The project gains the ability to *serve* models directly, with inference distributed across multiple nodes of a user-defined cluster. The pure-gateway functionality from v0.1 is preserved untouched — distributed inference is an *additional* provider kind, not a replacement.

This sub-project also includes a prerequisite rename of the project from `airproxy` to `ai-engine`. v0.1 was framed as a proxy; v0.2 makes it a full engine. The rename happens in one atomic commit before v0.2 implementation starts; everything in this spec uses the new name.

**In scope for v0.2:**

- A new crate, `ai-engine-runtime`, implementing a parameterized transformer in [burn](https://burn.dev/) supporting the Llama-3-style architecture family: RoPE positional embeddings, GQA, SwiGLU FFN, RMSNorm. Configurable hidden size, layer count, attention heads, KV heads, head dim, vocabulary size.
- A `safetensors` weight loader that ingests Hugging Face checkpoints and maps them into our blocks via a small config-driven naming registry. Covers Llama-3 / Mistral / Qwen-2.5 / DeepSeek-V2 (dense parts) — anything that fits the parameterized arch.
- burn backends compiled in: CPU (ndarray), CUDA (cubecl), Metal (cubecl), WebGPU (wgpu). Single binary, runtime-selected.
- A new "local-cluster" provider kind that exposes the cluster behind the existing `/v1/chat/completions` endpoint — drop-in compatible with the v0.1 plumbing.
- **Cluster topology**: persistent leader elected at startup via deterministic config-order. Each request is handled by the leader, which orchestrates worker shards via QUIC.
- **Transport**: QUIC (quinn) between nodes, mandatory TLS with self-signed certs auto-provisioned at startup, multiplexed control + data streams.
- **Partitioning**: capability-aware auto-partition. Each node advertises (RAM/VRAM, backend kind, link bandwidth); leader computes a layer-to-node assignment at cluster join.
- **KV cache distribution**: each worker owns the KV cache for its assigned layers, persisted across the autoregressive loop.
- **Token streaming**: SSE out to the client; activations stream node-to-node via QUIC.
- **Configuration**: TOML schema gains `[[cluster]]`, `[[cluster.node]]`, `[[cluster.model]]`, `[[cluster.partition_override]]`, plus `[[provider]] kind = "local-cluster"`. Cluster definition is static in v0.2.
- Testing: layered strategy — pure unit tests, single-node ML correctness against an HF reference, in-process distributed integration over loopback QUIC, real-SDK wire compatibility, multi-process smoke (one box, three processes), and a manual multi-machine release gate.

### Non-goals for v0.2

- **mDNS / Bonjour auto-discovery** — deferred to v0.3 (sub-project #4 in the renumbered roadmap).
- **Dynamic membership** — nodes joining or leaving a running cluster. v0.2 cluster is fixed at startup; topology change requires restart.
- **Fault tolerance** — a worker dying mid-request fails the whole request. No retry, no rebalancing, no checkpointing.
- **Tensor parallelism** — only pipeline parallelism (each node owns a contiguous range of layers). Tensor parallelism is deferred.
- **Speculative decoding, paged attention, continuous batching** — single-stream serving only in v0.2.
- **Quantization** — fp16 / bf16 only. Q4/Q5/AWQ/GPTQ deferred to v0.4+ (sub-project #8 in the renumbered roadmap).
- **MoE expert routing** — DeepSeek-V2 included only if its dense components fit our generic arch; MoE routing deferred indefinitely.
- **Training, fine-tuning, LoRA serving** — inference only.
- **Cross-format routing changes** — same format-pinning as v0.1.
- **Web playground UI** — deferred to v0.4+.
- **Heterogeneous backend per node beyond what burn supports out of the box** — no ROCm-specific code, no TPU.
- **Per-cluster multi-model serving** — each cluster serves exactly one model id in v0.2.
- **Weight distribution** — operator pre-stages identical model files on every node (rsync / NFS / S3 sync is the operator's job).

---

## 2. Architecture overview

### Roles

Two node roles, runtime-determined from config:

- **Leader.** Exactly one per cluster. Receives client HTTP requests via the existing axum router. Owns the tokenizer, the global request lifecycle, the autoregressive loop, and the worker-coordination protocol. The leader also hosts a layer range, so "leader" and "worker that happens to be first" are physically the same binary doing more bookkeeping.
- **Worker.** Owns a contiguous range of transformer layers + the KV cache for those layers + an embedding or output projection if assigned the boundary positions. Speaks QUIC inbound from the leader (and only the leader); never accepts client HTTP traffic.

Both roles run the same `ai-engine` binary with the same TOML config. Role per node is derived at startup from `[cluster].leader` (a node id) and the running node's id (hostname or `--node-id` override).

### Two planes

- **Control plane.** QUIC bidirectional stream, leader ↔ worker, postcard-framed binary messages. Used for: cluster join handshake (capability advertisement + partition assignment), KV cache lifecycle (allocate / free a request's slot), shutdown, health pings.
- **Data plane.** QUIC unidirectional streams, one per layer-boundary per request. Carries raw tensor activations in burn's native layout (bf16 by default). Length-prefixed binary frames so per-token overhead stays minimal.

Why two planes: control messages are bursty + small (kilobytes) and benefit from a long-lived bidi stream; data is steady + large (megabytes per request) and benefits from dedicated short-lived streams that QUIC can schedule independently without head-of-line blocking.

### Request lifecycle (single request, single token generation)

```
[client SDK] ──HTTP POST /v1/chat/completions──> [leader axum]
                                                       │
                                                       ▼
                                          [ForwardStage with kind=local-cluster]
                                                       │
                                          tokenize prompt → token IDs
                                                       │
                          ┌────────────── allocate request slot ──────────────┐
                          │  (control plane: leader broadcasts a `Begin`      │
                          │   message to all workers; each worker reserves    │
                          │   KV cache for its layer range)                   │
                          └────────────────────────────────────────────────────┘
                                                       │
                              ┌─── prefill ───┐  ┌─ generation loop ─┐
                              │ (full prompt) │  │ (one token at a time)
                              ▼               ▼  ▼
   leader (layers L0..Lk) → activations → worker A (layers Lk+1..Lm) →
        activations → worker B (layers Lm+1..Ln) → ... → leader (output proj)
                                                       │
                                              sample next token
                                                       │
                                              SSE chunk → client
                                                       │
                                          (loop until EOS or max_tokens)
                                                       │
                              ┌──── release request slot (control plane) ────┐
                              │  workers free KV cache for this request_id   │
                              └────────────────────────────────────────────────┘
```

Activations flow strictly forward: each worker is "between" two QUIC streams (one inbound, one outbound), forwards activations through its layer range, writes outbound. The leader is special only because it bookends — it takes the embedding step and the output projection / sampling step.

### Failure semantics in v0.2

- **Worker dies mid-request:** the leader sees the QUIC stream error; `ctx.error` is populated with `Provider(Stream(...))`; the existing v0.1 mid-stream SSE error path emits `event: error` and closes. **No retry, no failover.** The cluster as a whole continues serving subsequent requests; only the in-flight ones die.
- **Leader dies:** clients see connection reset; no recovery. When the operator restarts the leader, workers re-handshake (no persistent state on workers beyond the static layer assignment).
- **Network partition:** the leader's QUIC streams to the unreachable worker time out → in-flight requests fail as above. No quorum, no split-brain handling — v0.2 assumes a healthy LAN.

### Integration with v0.1

The cluster surfaces as a single entry in `[[provider]]` with a new `kind = "local-cluster"`. The existing `ai-engine-stages::ForwardStage` learns one branch: when the binding's provider is a cluster, it calls into `ai-engine-cluster` instead of `ai-engine-openai` / `ai-engine-anthropic`. Everything else (auth, content policy, model routing, log) is identical.

A node can simultaneously be a cluster member AND serve as a normal gateway to remote OpenAI/Anthropic upstreams. The two coexist in the same binary with the same pipeline. A user's `gpt-4o` request still goes to OpenAI; a `llama-3-70b` request goes to the cluster.

---

## 3. Workspace additions

Three new crates land in `crates/`. Each has a single responsibility; together they layer cleanly on top of v0.1.

```
crates/
├── ai-engine-runtime/           # NEW — burn-based parameterized transformer + safetensors loader
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs               # public surface: Model, ModelConfig, Backend selection
│       ├── arch/
│       │   ├── mod.rs
│       │   ├── transformer.rs   # generic decoder block: RMSNorm → attn → residual → RMSNorm → SwiGLU → residual
│       │   ├── attention.rs     # MHA / GQA with RoPE, KV cache slot interface
│       │   ├── ffn.rs           # SwiGLU
│       │   ├── embedding.rs     # token embedding + (tied or untied) output projection
│       │   └── rope.rs          # precomputed cos/sin tables, applied per-layer
│       ├── config.rs            # ModelConfig (hidden_size, n_layers, n_heads, n_kv_heads, head_dim, vocab_size, …)
│       ├── kv_cache.rs          # per-layer KV cache abstraction; backend-generic
│       ├── loader.rs            # safetensors → Tensor mapping via a name registry
│       ├── name_map.rs          # one entry per model family (Llama-3, Mistral, Qwen-2.5, DeepSeek-V2)
│       └── backend.rs           # burn backend selection at runtime (ndarray / cuda / metal / wgpu)
│
├── ai-engine-cluster/           # NEW — QUIC transport + control protocol + partitioner
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs               # ClusterProvider (implements ai_engine_provider::Provider)
│       ├── transport/
│       │   ├── mod.rs
│       │   ├── quic.rs          # quinn setup, TLS bootstrap, connection pool
│       │   └── frame.rs         # length-prefixed binary framing for activations
│       ├── protocol/
│       │   ├── mod.rs
│       │   ├── control.rs       # serde types: Join, Capability, Assignment, Begin, End, Health
│       │   ├── data.rs          # activation frame headers (request_id, layer_idx, seq_pos, dtype, shape)
│       │   └── codec.rs         # encode/decode helpers (postcard)
│       ├── partition.rs         # capability-aware layer-to-node assignment algorithm
│       ├── leader.rs            # request orchestration, generation loop, KV slot mgmt
│       ├── worker.rs            # accepts QUIC inbound, runs assigned layers, manages local KV cache
│       └── tls.rs               # self-signed cert provisioning, peer pinning by hash
│
└── ai-engine-tokenizer/         # NEW — tokenizer abstraction
    ├── Cargo.toml
    └── src/
        ├── lib.rs               # Tokenizer trait: encode(&str) -> Vec<u32>, decode(&[u32]) -> String
        └── hf.rs                # wraps huggingface/tokenizers; loads tokenizer.json from HF dump
```

### Modifications to existing crates (renamed)

- **`ai-engine-config`** — new schema blocks: `[[cluster]]`, `[[cluster.node]]`, `[[cluster.model]]`, `[[cluster.partition_override]]`. The `[[provider]] kind = "local-cluster"` variant gains a `cluster = "<id>"` field that links to a `[[cluster]]`. Validation: leader id must match a node id; node QUIC addresses must be unique; partition_override must form a contiguous, non-overlapping, complete cover of `0..n_layers`.
- **`ai-engine-stages::forward`** — `ForwardStage` learns one branch: when binding resolves to a cluster-kind provider, dispatch to `ai-engine-cluster::ClusterProvider`. The `Provider` trait does not change — `ClusterProvider` implements `Provider::chat` / `chat_stream` like any other.
- **`ai-engine`** binary — `build_app_state` learns to construct `ClusterProvider`s from `[[cluster]]` config. On startup, if this node's id matches `[cluster].leader`, it joins as leader (sets up the QUIC listener + handles HTTP); otherwise it boots in worker mode (QUIC listener only, no chat routes — health endpoints stay exposed).
- **Workspace `Cargo.toml`** — new workspace deps: `burn` (with feature flags per backend), `safetensors`, `tokenizers`, `quinn`, `rustls`, `rcgen`, `postcard`, `petgraph` (for partitioner graph).

### Crate dependency shape

```
ai-engine-runtime    depends on:  burn (+ backend features), safetensors
ai-engine-tokenizer  depends on:  tokenizers
ai-engine-cluster    depends on:  ai-engine-runtime, ai-engine-tokenizer,
                                  ai-engine-provider, ai-engine-core,
                                  quinn, rustls, rcgen, postcard
ai-engine-stages     gains dep on:  ai-engine-cluster
ai-engine            gains dep on:  ai-engine-cluster, ai-engine-runtime, ai-engine-tokenizer
```

The trait-only crates (`ai-engine-core`, `ai-engine-provider`) remain dependency-light. The trait surface from v0.1 is stable: adding distributed inference required zero changes to `Provider`, `Stage`, `Pipeline`, or `RequestCtx`.

### Why three crates, not one

- **`ai-engine-runtime`** is reusable on its own. Someone who wants single-node inference (no cluster) can depend on just this — a future v0.3+ might expose a "local" provider kind that uses `ai-engine-runtime` directly with no cluster wrapping.
- **`ai-engine-tokenizer`** is reusable on its own. Tokenizer concerns are orthogonal to model architecture or transport.
- **`ai-engine-cluster`** is the only piece that knows about both ML and networking — and it should stay focused. Depending on `ai-engine-runtime` and `ai-engine-tokenizer` rather than absorbing them lets the transport / protocol / partitioner work be exercised against mock implementations of the runtime, which dramatically improves testability without a GPU.

---

## 4. Model layer (`ai-engine-runtime`)

The hardest single piece of v0.2. Most of the design risk lives here because burn doesn't ship a Llama-class transformer we can lift.

### What we build

One parameterized decoder-only transformer. Forward pass shape:

```
input: [batch, seq] token IDs
  ↓ embedding lookup
  ↓ for each of n_layers:
      x_norm = RMSNorm(x)
      q, k, v = Linear(x_norm) split into n_heads / n_kv_heads
      apply RoPE to q, k
      append k, v to layer's KV cache; read full cached k, v
      attn_out = scaled_dot_product_attention(q, k, v, causal_mask)
      x = x + Linear(attn_out)               # residual
      x_norm = RMSNorm(x)
      gate, up = Linear(x_norm) (SwiGLU has two projections)
      x = x + Linear(SiLU(gate) * up)        # residual + SwiGLU FFN
  ↓ RMSNorm
  ↓ Linear(hidden_size → vocab_size) — tied or untied with embedding per config
output: [batch, seq, vocab]
```

This covers Llama-3, Mistral-7B/8x7B (dense parts only), Qwen-2.5, DeepSeek-V2 (dense parts only). Architectures with sliding window attention, Mamba blocks, or MoE routing are out of scope; their configs reject at load time with a clear message.

### `ModelConfig`

```rust
pub struct ModelConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,        // FFN inner dim
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,               // < n_heads enables GQA
    pub head_dim: usize,                 // usually hidden_size / n_heads, but DeepSeek decouples
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub tie_word_embeddings: bool,
    pub family: ModelFamily,             // for tokenizer + weight-name lookup
}

pub enum ModelFamily { Llama3, Mistral, Qwen25, DeepSeekV2 }
```

Loaded from `config.json` next to the safetensors files (standard HF dump layout). The loader validates required fields, defaults the optional ones, and rejects configs naming an architecture we don't support yet (e.g., `mixtral` errors with "MoE not supported in v0.2").

### Per-layer weight load (`load_range`)

The crucial bit for distributed inference: every weight tensor is tagged with its layer index (for per-layer tensors) or `Embedding` / `OutputProjection` for boundary tensors. The loader produces:

```rust
pub struct LoadedWeights<B: Backend> {
    pub embedding: Option<Tensor<B, 2>>,         // None on workers not hosting the embed
    pub layers: Vec<LayerWeights<B>>,            // index relative to assigned range start
    pub final_norm: Option<Tensor<B, 1>>,
    pub output_proj: Option<Tensor<B, 2>>,       // None if tied or not hosted here
}
```

A worker that owns layers 12..24 deserializes only those entries from the safetensors mmap. The loader has a `load_range(path, start_layer, end_layer, hosts_embedding, hosts_output)` constructor that walks the safetensors header, picks matching tensors, and memory-maps only the relevant byte ranges. Workers never load embedding or output_proj unless explicitly assigned.

### Backend abstraction

burn parameterizes everything on `B: Backend`. Concrete `Backend` choice at startup based on `[[cluster.node]].backend` config:

```rust
pub enum BackendKind { Cpu, Cuda, Metal, Wgpu }
```

Models are loaded onto the chosen backend; tensors moved between backends only at the QUIC boundary (transfer to CPU before serializing, then onto the target backend on the receiving side). To keep model code shared across backends, `Model<B: Backend>` is generic; per-backend factory functions are feature-gated:

```rust
#[cfg(feature = "backend-cuda")]
pub fn cuda_model(cfg: &ModelConfig, weights_path: &Path, layer_range: Range<usize>) -> Model<burn::backend::Cuda> { … }
```

The binary is larger when all backends are compiled in. v0.2 ships them all by default; an `embedded` feature flag for cherry-picking backends is deferred.

### KV cache

Per-layer, per-request:

```rust
pub struct KvCacheSlot<B: Backend> {
    pub k: Tensor<B, 4>,    // [batch, n_kv_heads, max_seq, head_dim]
    pub v: Tensor<B, 4>,
    pub current_len: usize, // how many tokens written
}
```

Each worker maintains a `HashMap<RequestId, Vec<KvCacheSlot<B>>>` — one slot vec per active request, one slot per layer the worker owns. Slots allocated on `ControlMessage::Begin { request_id, max_tokens }`, freed on `End { request_id }`. Slot size is bounded by `max_tokens + prompt_len`, so memory per slot is deterministic at allocation time.

### Sampling

On the leader (it runs the output projection). v0.2 supports:

- **greedy** (argmax)
- **temperature + top-p (nucleus)**
- **top-k**

Configurable per request via the existing OpenAI `temperature`, `top_p`, `top_k` request fields. Mirostat, typical-p, DRY are deferred.

### Streaming generation

The leader's generation loop:

```rust
for token_idx in 0..max_tokens {
    let activations = forward_through_local_layers(token);
    for worker in workers_in_pipeline_order {
        // SEND activations on the request's data stream
        // RECV next-stage activations on the inbound stream
        activations = worker.exchange(request_id, activations).await?;
    }
    let logits = output_projection(activations);
    let next_token = sample(logits);
    emit_sse_chunk(next_token);
    if next_token == eos { break; }
    // next iteration: only feed the single new token to the embedding
}
```

Backpressure is bounded by quinn's per-stream window; if the client SSE consumer falls behind, axum's SSE responder pauses, propagating back through the in-process channel into this loop.

### Risks acknowledged

- **GQA correctness against HF reference.** Llama-3-8B uses GQA (32 heads, 8 KV heads). KV repeat / broadcast indexing is the most common bug site. Mitigation: bytes-exact comparison of logits against `transformers` HF reference on a fixed prompt, tolerance < 1e-3 bf16, as a CI test gate.
- **RoPE off-by-one with KV cache.** The position passed into RoPE must reflect cumulative sequence position, not position within the current forward call. Test suite exercises multi-step generation with a fresh-vs-cached cross-check.
- **safetensors name maps** drift between families and versions. Each `ModelFamily` has a small declarative name-map module with tests against a recorded fixture from each canonical HF dump.

---

## 5. Partitioning

How layers get assigned to nodes. Approach is "capability-aware auto-partition."

### Node capability advertisement

At cluster join, every node publishes a `Capability` over the control plane:

```rust
pub struct Capability {
    pub node_id: String,
    pub backend: BackendKind,           // Cpu | Cuda | Metal | Wgpu
    pub device_index: usize,
    pub available_memory_bytes: u64,    // VRAM for GPU; system RAM for CPU
    pub compute_score: u32,             // synthetic — see below
    pub link_mbps_to_leader: u32,       // measured via brief QUIC throughput probe at join
}
```

- **available_memory_bytes**: queried from the burn backend at startup. `nvml` for CUDA, Metal device info for macOS, `sysinfo` for CPU. Conservative — subtracts a safety margin (default 512 MiB) before reporting.
- **compute_score**: one-time microbenchmark at startup — run a fixed 1024×1024 matmul, time it, normalize so a baseline CPU = 100. Dimensionless relative ordering, not TFLOPs.
- **link_mbps_to_leader**: measured during join handshake — leader sends a 4 MiB payload, worker echoes, RTT and throughput recorded. Refreshed only at join.

### The partitioning problem

Given:
- A model with `n_layers` layers, each with known per-layer memory cost (deterministic function of `ModelConfig`) and per-layer compute cost (assumed uniform — layers in Llama-class models are identical).
- N nodes with `(memory_i, compute_i, link_i)` from capability messages.
- Total memory headroom: leader also hosts embedding + output_proj, which adds `vocab_size × hidden_size × 2` bytes on top of its layer assignment.

Find a **contiguous** layer-to-node assignment that:
1. **Respects memory constraints**: each node's assigned-layer-bytes (+ KV cache budget at default `max_tokens=4096, batch=1`) ≤ that node's `available_memory_bytes`.
2. **Minimizes the maximum stage latency** (slowest pipeline stage dominates token throughput): `max_i (assigned_layers_i / compute_i + (link_to_next_i)⁻¹)`.

### The algorithm

A **dynamic-programming layer-cut solver** over a fixed pipeline-order of nodes:

1. **Order nodes by topology.** v0.2 uses config order — the order in which nodes appear in `[[cluster.node]]` is the pipeline order. v0.3 may auto-optimize.
2. **DP state**: `cost[i][k]` = minimum max-stage-cost of assigning the first `i` layers across the first `k` nodes, subject to memory constraints.
3. **Transition**: `cost[i][k] = min over j < i of  max(cost[j][k-1], stage_cost(j..i, node_k))` where `stage_cost(range, node) = layers_in_range / node.compute_score + transport_overhead(node)`.
4. **Feasibility filter**: any assignment violating a node's memory cap returns `∞`, pruning it.
5. **Recover assignment** by backtracking through the DP table.

Complexity: `O(n_layers² × n_nodes)`. For Llama-3-70B (80 layers) on 5 nodes, ~32,000 comparisons — microseconds.

If no feasible assignment exists, the leader fails cluster startup with:

```
ERROR: model llama-3-70b (134 GiB at bf16, plus 28 GiB embed/output, plus KV budget)
       does not fit any partition across this cluster (total available memory: 96 GiB).
       Options: add another node, reduce max_tokens, or use a smaller model.
```

### Why contiguous-only

Pipeline parallelism requires contiguous layer ranges per node — non-contiguous assignment forces activations to bounce, which is worse than no parallelism. Contiguous is the only sensible shape under the pipeline-parallel-only constraint adopted in §1.

### Partition manifest

Once computed, the leader broadcasts the assignment as part of the `Assignment` control message:

```rust
pub struct PartitionManifest {
    pub model_id: String,
    pub model_config_hash: [u8; 32],     // SHA-256 of ModelConfig — workers verify
    pub assignments: Vec<NodeAssignment>,
}

pub struct NodeAssignment {
    pub node_id: String,
    pub layer_range: Range<usize>,
    pub hosts_embedding: bool,
    pub hosts_output: bool,
    pub previous_node: Option<String>,
    pub next_node: Option<String>,
}
```

The manifest is content-addressed by `(model_id, model_config_hash, capabilities_hash)`. Same cluster + same capabilities + same model ⇒ identical deterministic assignment. Useful for warm restarts, partition reproducibility in tests, and debugging.

### Manual override

`[[cluster.partition_override]]` in TOML accepts an explicit assignment, bypassing the solver:

```toml
[[cluster.partition_override]]
node = "node-a"
layers = "0..27"
```

The solver still runs for validation (memory feasibility check), but layer ranges come from config. Useful for testing, benchmarks, and operator intuition during early experimentation.

### Determinism property

Given the same `(ModelConfig, [Capability])` tuple, the solver returns the same `PartitionManifest`. Tie-breaks in the DP use lexicographic order on node IDs. Required for:
- Warm-restart equivalence.
- Integration tests asserting exact partition shapes.
- Debugging: a partition that worked yesterday and fails today implies a capability change.

---

## 6. Inter-node protocol

The wire shape between nodes.

### Connection model

- **QUIC endpoint per node.** Each worker listens on a configured `quic_bind` address. The leader opens one persistent `quinn::Connection` per worker at cluster join. Connections are pooled and survive across requests.
- **TLS is mandatory.** v0.2 ships with **auto-provisioned self-signed certs**: on first startup, each node generates a long-lived ed25519 keypair stored in `~/.ai-engine/node.{key,crt}`. The cluster's `[[cluster.node]]` config lists every node's certificate SHA-256 fingerprint, so peers pin by hash — no CA, no manual cert exchange beyond copying fingerprints into config (operator does this once, at cluster creation; CLI subcommand `ai-engine node fingerprint` prints it).
- **ALPN**: `ai-engine-cluster/1`. Defends against accidental connection from non-cluster clients and reserves room for protocol versioning.

### Two-plane stream layout per connection

```
Leader ↔ Worker connection (one per (leader, worker) pair)
│
├── Bidirectional stream #0 — Control plane (long-lived, postcard frames)
│       leader → worker:  Join, Assignment, Begin{request_id, …}, End{request_id}, Health
│       worker → leader:  Capability, JoinAck, BeginAck, Heartbeat, FaultReport
│
└── Many unidirectional streams — Data plane (one stream per direction per request)
        leader → worker (request_id=R, direction=forward):
            binary frames carrying activations for R as the leader produces them
        worker → leader (request_id=R, direction=backward):
            binary frames carrying activations from R after the worker's layers
```

For an N-node pipeline, each adjacency `(i, i+1)` in the pipeline gets one stream per request, opened lazily on the first token and closed when the leader sends `End`.

### Control plane: message types

Encoded with `postcard` (compact binary, serde-driven).

```rust
pub enum LeaderToWorker {
    Join {
        cluster_id: String,
        protocol_version: u16,
        leader_node_id: String,
    },
    Assignment {
        manifest: PartitionManifest,
        model_id: String,
    },
    Begin {
        request_id: Uuid,                  // uuidv7 from leader
        max_tokens: u32,
        prompt_len: u32,                   // for KV cache size hint
    },
    End {
        request_id: Uuid,
        reason: EndReason,                 // Completed | ClientCancelled | Error
    },
    HealthPing { nonce: u64 },
}

pub enum WorkerToLeader {
    Capability(Capability),                // §5, sent immediately after Join
    JoinAck { node_id: String, certificate_sha256: [u8; 32] },
    BeginAck { request_id: Uuid },         // sent once KV slot is allocated
    Heartbeat { nonce: u64 },              // echo of HealthPing
    FaultReport {
        request_id: Option<Uuid>,
        kind: FaultKind,                   // OutOfMemory | BackendError | Internal
        detail: String,
    },
}
```

Control messages are length-prefixed (u32 LE) on the bidi stream.

### Data plane: activation frames

Per-token, per-stage activations. Each frame:

```
┌──────────────────────────────────────────────────────────────┐
│ u32 LE  payload_len                                           │
│ ActivationHeader (postcard)                                   │
│   request_id:      Uuid                                       │
│   seq_pos:         u32      // position of this token in the  │
│                              //  generation (0 = first token) │
│   shape:           [u32; 3] // [batch, seq, hidden] — seq=1   │
│                              //  during generation, ≥1 prefill│
│   dtype:           Dtype    // Bf16 | F16 | F32               │
│   is_terminal:     bool     // true on the last activation    │
│                              //  for this request             │
│ tensor_bytes: [u8; …]       // raw, little-endian, dense      │
└──────────────────────────────────────────────────────────────┘
```

- **Dtype is per-frame, not per-stream.** Encoding it on the frame means future heterogeneous-dtype mixes don't require a protocol revision.
- **No compression** in v0.2. Activations are bf16 dense; for `hidden_size=8192, seq=1, batch=1` that's 16 KiB per layer-boundary per token. At 50 tok/s that's 800 KiB/s — modest. Not worth CPU cost until measurements demand it.
- **No batching across requests.** v0.2 is single-stream. Multiple concurrent requests get independent streams; QUIC's per-stream flow control isolates them naturally.

### Lifecycle of one request

```
T+0     leader → all workers (control bidi): Begin { request_id, max_tokens, prompt_len }
T+1     workers → leader (control bidi):     BeginAck   (each worker allocates KV slot)
T+2     leader → worker A (data uni #N):     ActivationFrame  (prefill: shape=[1, prompt_len, hidden])
T+3     worker A → worker B (data uni #M):   ActivationFrame  (after A's layer range)
T+4     worker B → leader (data uni #K):     ActivationFrame  (after B's layer range)
T+5     leader: output_proj + sample → first token → SSE chunk → client
T+6     (generation loop: every subsequent token = one round-trip through the pipeline)
…
T+last  leader → all workers (control bidi): End { request_id, reason: Completed }
        workers free KV slot for request_id
```

Data-uni streams are short-lived per request, closed by sending a frame with `is_terminal = true`.

### Backpressure

Native to QUIC: each unidirectional stream has independent flow control. A slow worker fills its inbound stream's window; the previous node's send buffer hits its limit; sender awaits — propagating backpressure upstream all the way to the leader, then to the SSE writer, then to the client.

### Latency budget (illustrative)

For a 5-node pipeline on a wired LAN (sub-millisecond RTT):
- Per-token cost = `Σ (layer_compute_time_per_node) + (n_hops − 1) × hop_overhead`
- Hop overhead ≈ 200–500 μs on QUIC over wired LAN (RTT + serde + memcpy).
- For Llama-3-70B distributed 4 ways, 20 layers/node ≈ 8 ms compute/node @ A100; 4 nodes = ~32 ms compute + ~1.5 ms transport. ~30 tok/s wall — competitive with single-node serving at this size; the win is **fitting** the model.

### Protocol versioning

`protocol_version: u16` in `Join`. v0.2 ships version `1`. On mismatch the worker sends `FaultReport { kind: Internal, detail: "protocol version mismatch: leader=1, worker=2" }` and cluster startup fails. We commit to bumping the version on any wire-incompatible change.

### Acknowledged risks

- **TLS bootstrap UX.** Asking the operator to copy SHA-256 fingerprints across nodes is friction. Mitigated by `ai-engine node fingerprint` CLI. v0.3 will likely automate via mDNS + TOFU.
- **Stream proliferation.** N concurrent requests × M pipeline stages = N×M open data streams. QUIC handles thousands per connection (designed for HTTP/3), but ops should monitor `quinn::Connection::open_streams` and warn at thresholds.
- **postcard schema drift.** Adding a field to a `Capability` could break a worker compiled against an older `ai-engine-cluster`. Mitigated by `#[serde(default)]` on every additive field and the `protocol_version` hard gate.

---

## 7. Configuration extensions

TOML schema grows to express a cluster. Three new blocks, plus one new `[[provider]]` kind. All v0.1 schema continues to work unchanged.

### `[[cluster]]` block

Defines one logical cluster. A node binary can be a member of at most one cluster in v0.2.

```toml
[[cluster]]
id = "home-lab"
leader = "node-a"                  # references a [[cluster.node]].id below
quic_bind = "0.0.0.0:7700"         # this node's QUIC listener
protocol_version = 1
join_timeout_secs = 30             # how long the leader waits for all workers to join
heartbeat_interval_secs = 5

# A model to serve. Multiple models per cluster are NOT supported in v0.2 —
# each cluster serves exactly one model. (Multiple clusters in one process IS supported,
# each with its own [[cluster]] block and its own model.)
[cluster.model]
id = "llama-3-70b"
config_path = "/srv/models/llama-3-70b/config.json"
weights_path = "/srv/models/llama-3-70b"       # safetensors shards live here on every node
tokenizer_path = "/srv/models/llama-3-70b/tokenizer.json"
# Operator pre-stages identical model files on every node.

# Optional explicit partition (overrides the auto-partitioner).
# Omit this to let ai-engine compute the assignment from capabilities.
[[cluster.partition_override]]
node = "node-a"
layers = "0..27"

[[cluster.partition_override]]
node = "node-b"
layers = "27..54"

[[cluster.partition_override]]
node = "node-c"
layers = "54..80"
```

### `[[cluster.node]]` block

One per node in the cluster. Every node's config file lists every node — config is symmetric across the cluster.

```toml
[[cluster.node]]
id = "node-a"
addr = "192.168.1.10:7700"              # QUIC dial address (peers connect here)
cert_fingerprint = "sha256:a3f9c2…"     # SHA-256 of this node's TLS cert
backend = "cuda"                        # cpu | cuda | metal | wgpu
device_index = 0
# Optional override of the auto-detected memory ceiling (in MiB).
max_memory_mib = 79000

[[cluster.node]]
id = "node-b"
addr = "192.168.1.11:7700"
cert_fingerprint = "sha256:8b1c4e…"
backend = "metal"
device_index = 0
```

### `[[provider]]` extension

The cluster is exposed via the existing provider system:

```toml
[[provider]]
id = "home-cluster"
kind = "local-cluster"
cluster = "home-lab"                    # references [[cluster]].id
# base_url / api_key / http2 fields silently rejected with a warning if present.
```

Routes bind to it like any other provider:

```toml
[[route]]
match = { model = "llama-3-70b" }
provider = "home-cluster"

[[route]]
match = { model = "llama3*" }           # fallback
provider = "home-cluster"
upstream_model = "llama-3-70b"          # cluster only serves one model id
```

### Role derivation at startup

Role is derived from `[[cluster]].leader` matching one of the node IDs in `[[cluster.node]]`, and the running node identifying itself via **hostname → node-id lookup** (each node's `[[cluster.node]].id` should match `hostname()`, or `--node-id <id>` overrides on the CLI).

```bash
# Most common case: hostname matches the id field
ai-engine --config ai-engine.toml

# Override (useful for containers, multiple instances per host, tests)
ai-engine --config ai-engine.toml --node-id node-c
```

If the resolved id matches `[[cluster]].leader`, this binary boots as **leader**: starts axum router (HTTP, port from `[server].bind`) + QUIC listener (port from `[cluster].quic_bind`).

Otherwise **worker**: starts QUIC listener only. `[server].bind` is reused but only exposes `/healthz` and `/readyz` so operators can health-check without going through QUIC.

### Validation rules (additive to v0.1's)

- `[[cluster]].leader` must reference an existing `[[cluster.node]].id`.
- `[[cluster.node]].id` must be unique within the cluster.
- `[[cluster.node]].addr` must be unique (no two nodes on the same `host:port`).
- `[[cluster.node]].cert_fingerprint` must be a valid SHA-256 hex string with `sha256:` prefix.
- `[[cluster.node]].backend` must be a known kind.
- `[[provider]] kind = "local-cluster"` must have `cluster` referencing a `[[cluster]].id`.
- `[[cluster.partition_override]]` entries must form a contiguous, non-overlapping, complete cover of `0..n_layers` once the model config is loaded (two-phase validation: TOML parse first, then post-model-load).
- Each `[[provider]] kind = "local-cluster"`'s referenced cluster must define exactly one `[cluster.model]`.

### Hot reload behavior

The existing SIGHUP reload from v0.1 explicitly **does not** apply to `[[cluster]]` or `[[cluster.node]]` changes — those require a full restart. Validation:

```
SIGHUP received
  ↓
parse new config
  ↓
diff against running config
  ↓
if any [[cluster]] / [[cluster.node]] field changed → reject reload, warn:
   "cluster topology changes require restart; old config retained"
  ↓
otherwise (only [[route]], [[pipeline]], stage params changed) → atomic-swap pipelines as in v0.1
```

Keeps the hot-reload contract honest without closing the door on it for non-cluster changes.

---

## 8. Integration with v0.1 gateway

Zero changes to `ai-engine-core` or `ai-engine-provider` — the trait surface from v0.1 was designed for this exact extension.

### `ClusterProvider` implements the existing `Provider` trait

```rust
// ai-engine-cluster/src/lib.rs
pub struct ClusterProvider {
    id: String,
    leader: Arc<leader::LeaderState>,    // populated only if this node is the leader
    is_leader: bool,
}

#[async_trait::async_trait]
impl Provider for ClusterProvider {
    fn id(&self) -> &str { &self.id }
    fn kind(&self) -> &'static str { "local-cluster" }
    fn capabilities(&self) -> Capabilities {
        Capabilities { chat: true, streaming: true, tools: false, vision: false, messages: false, embeddings: false }
    }

    async fn chat(&self, req: openai::ChatRequest, _creds: &Credentials, ctx: &CallCtx)
        -> Result<openai::ChatResponse, ProviderError>
    {
        if !self.is_leader { return Err(ProviderError::Unsupported); }
        self.leader.complete_chat(req, ctx).await
    }

    async fn chat_stream(&self, req: openai::ChatRequest, _creds: &Credentials, ctx: &CallCtx)
        -> Result<EventStream<openai::ChatStreamEvent>, ProviderError>
    {
        if !self.is_leader { return Err(ProviderError::Unsupported); }
        self.leader.stream_chat(req, ctx).await
    }

    // messages, messages_stream, embeddings → default Unsupported.
}
```

`ForwardStage` doesn't learn anything new at the trait level: it already calls `provider.chat()` / `chat_stream()` and handles `Err(ProviderError::Unsupported)`.

### Credentials are deliberately ignored

The cluster has no upstream credentials concept. The `_creds` parameter is accepted to satisfy the trait but discarded. Per-user auth gating lives in `AuthStage`, same as for any provider.

### Pipeline configuration is unchanged

A cluster route uses exactly the same pipeline as any other route:

```toml
[pipeline."/v1/chat/completions"]
stages = ["auth", "content_policy", "model_route", "forward", "log"]
```

Users adopting v0.2 see **zero changes** to their request flow. They configure a new provider, add a new route, the same OpenAI-shape HTTP API works. The cluster is functionally indistinguishable from "a really fast OpenAI-compatible upstream" from the route table's perspective.

### Mixed routing in one process

A single `ai-engine` instance can simultaneously be:
1. A gateway to remote OpenAI (`gpt-4o` requests).
2. A gateway to remote Anthropic (`claude-*` requests).
3. A gateway to a local Ollama (`llama3.2:1b`).
4. A leader of a cluster serving `llama-3-70b`.

All four coexist because they're four entries in `[[provider]]` + four `[[route]]` rules.

### Worker mode runs the SAME binary

A worker node:
- Skips axum routing (no `/v1/*` endpoints exposed).
- Skips `[[provider]]` construction beyond its own cluster's `ClusterProvider` (with `is_leader = false`).
- Skips pipeline construction entirely (workers have no pipeline — they're called by the leader over QUIC, not by axum).
- Hosts a tiny axum router with just `/healthz` and `/readyz` for ops visibility.
- Spins up the QUIC listener, registers with the leader, and waits.

`app::build_app_state` grows a worker-mode branch returning a stripped `AppState` that carries only health-endpoint state + a QUIC worker handle.

### `/v1/models` exposure

The cluster provider contributes `[cluster.model].id` to `AppState::openai_models`. SDKs enumerating available models see local-cluster and remote-gateway models in one list — exactly the abstraction we want.

### The one ForwardStage change (acknowledged)

`build_app_state` gains a one-line construct-ClusterProvider branch when it sees `kind = "local-cluster"`. The stage code itself looks up providers by id and dispatches polymorphically through the trait — genuinely unchanged.

---

## 9. Testing strategy

Testing distributed inference without a multi-machine cluster. Layered, each layer runnable on a single workstation.

### Layer 1 — Pure unit tests (no ML, no network)

| Crate | Coverage |
|---|---|
| `ai-engine-cluster::partition` | DP solver determinism, feasibility, manual-override validator, edge cases |
| `ai-engine-cluster::protocol::codec` | Round-trip every control message via postcard; activation header encode/decode |
| `ai-engine-cluster::tls` | Self-signed cert generation, fingerprint computation, peer pinning validator |
| `ai-engine-runtime::config` | safetensors `config.json` parse, family detection, rejection messages |
| `ai-engine-runtime::name_map` | Golden test asserting weight-name mapping per `ModelFamily` |

### Layer 2 — Single-node ML correctness (bytes-exact gate)

**Without this, the rest doesn't matter.**

Test harness: a tiny safetensors fixture of a deliberately-small model (e.g., Llama-3-style 32M-param toy with 4 layers, hidden 256, 4 heads, 2 KV heads) checked into the repo at `crates/ai-engine-runtime/fixtures/toy-llama-3/`. ~40 MB at bf16.

Tests:

- `forward_matches_reference_logits.rs` — run a forward pass on a fixed prompt, compare logits element-wise against `reference_logits.bin` precomputed from `transformers` on the same fixture. Tolerance: `max |a - b| < 1e-3` in bf16. **Canonical correctness gate.**
- `multi_step_generation_with_kv_cache.rs` — generate 16 tokens, then re-run from scratch as a single 16-token forward, compare token-by-token. Exposes RoPE / KV cache off-by-ones.
- `gqa_correctness.rs` — fixture uses GQA explicitly; targeted assertions.
- `safetensors_load_range.rs` — load only layers 2..4 of the toy; assert tensor shapes and that layers 0,1 are not memory-mapped.
- `backend_parity.rs` — same forward pass on `Cpu` and `Wgpu`; assert tensors match within wgpu tolerance. CUDA/Metal parity tests gated by `#[cfg]` on CI runners with those backends.

### Layer 3 — In-process distributed integration

Spin up "cluster" of N workers + 1 leader inside one process, communicating over loopback QUIC. Catches actual distributed bugs without multiple machines.

```rust
// crates/ai-engine-cluster/tests/inprocess_cluster.rs

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn three_node_pipeline_generates_correct_tokens() {
    let toy = fixtures::toy_llama();
    let workers = spawn_workers(&toy, vec![(0..2), (2..4)]);
    let leader = spawn_leader(&toy, 0..0, &workers).await;
    let resp = http_client.post(format!("{}/v1/chat/completions", leader.base_url))
        .json(&json!({"model": "toy-llama", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 5}))
        .send().await.unwrap();
    let chat: openai::ChatResponse = resp.json().await.unwrap();
    assert_eq!(chat.choices[0].message.content, fixtures::expected_completion());
}
```

Additional tests:
- `streaming_generation.rs` — `stream: true`; SSE chunks arrive in order with correct token deltas.
- `worker_dies_midstream.rs` — kill one worker's task; leader emits `event: error` and request fails cleanly without hanging.
- `partition_override.rs` — manual partition produces identical output to auto-partitioned (determinism cross-check).
- `kv_cache_isolation.rs` — interleave two concurrent requests; tokens don't leak across.
- `backpressure.rs` — slow SSE consumer; leader's generation loop blocks rather than buffering unboundedly.

### Layer 4 — End-to-end with `async-openai` SDK

Wire-compat tests, same shape as v0.1's `wire_compat_*.rs` but pointed at an in-process cluster.

```rust
#[tokio::test(flavor = "multi_thread")]
async fn sdk_works_against_inprocess_cluster() {
    let cluster = spawn_3_node_cluster(toy_llama()).await;
    let client = async_openai::Client::with_config(
        async_openai::config::OpenAIConfig::new()
            .with_api_base(format!("{}/v1", cluster.leader_url))
            .with_api_key("test-key")
    );
    let req = async_openai::types::CreateChatCompletionRequestArgs::default()
        .model("toy-llama").messages([…]).stream(true)
        .build().unwrap();
    let mut stream = client.chat().create_stream(req).await.unwrap();
    while let Some(chunk) = stream.next().await { let _ = chunk.unwrap(); }
}
```

Proves SDK compatibility extends to the cluster path.

### Layer 5 — Multi-process smoke (one machine, multiple OS processes)

Bridges in-process tests and a real cluster. CI runs this on a single Linux runner.

```bash
ai-engine --config tests/fixtures/cluster_3node.toml --node-id worker-1 &
ai-engine --config tests/fixtures/cluster_3node.toml --node-id worker-2 &
ai-engine --config tests/fixtures/cluster_3node.toml --node-id leader &
sleep 2
curl http://127.0.0.1:8080/v1/chat/completions -d '{…}' | jq '.choices[0].message.content'
```

Run via `#[ignore]`'d Rust test that shells out. Catches bugs missed by in-process tests because they share an allocator and tokio runtime — static-state assumptions, double-init of logging, port reuse, certificate file paths.

### Layer 6 — Manual multi-machine validation

Documented in `docs/superpowers/notes/v0.2-multimachine-validation.md`: runbook the developer follows once before tagging v0.2.0. Two physical/cloud nodes, exchange fingerprints, run Llama-3-8B partition across them, compare latency and output to single-node baseline. Not in CI; explicitly a release gate.

### Load smoke (the v0.1 pattern)

`crates/ai-engine-cluster/tests/load_smoke.rs` — `#[ignore]`'d. 100 concurrent SSE streams against an in-process cluster using the toy model. Assert: p99 TTFT, no dropped streams, memory stable.

### What we deliberately don't test

- Real-money model correctness on real hardware (Layer 6 is the manual gate).
- Cross-OS networking in CI (Linux only; QUIC libs are mature enough).
- Adversarial network conditions — out of scope for v0.2 (fault tolerance is a non-goal).

### Test count target

| Layer | New tests added |
|---|---|
| 1 — Pure unit | ~40 |
| 2 — Single-node ML | 5 |
| 3 — In-process cluster | 6 |
| 4 — SDK wire-compat | 2 |
| 5 — Multi-process smoke | 1 (`#[ignore]`) |
| Load smoke | 1 (`#[ignore]`) |
| **Total new tests** | **~55** |

v0.2 ships at ~133 tests workspace-wide (v0.1's 78 + ~55 new).

---

## 10. Sub-project boundaries

### Prerequisites (single atomic step before v0.2 work)

**P0** — Rename v0.1 from `airproxy` to `ai-engine`. Mechanical pass:
- Every crate `airproxy-*` → `ai-engine-*`.
- Binary `airproxy` → `ai-engine`.
- `airproxy.toml(.example)` → `ai-engine.toml(.example)`.
- Every `use airproxy_*::*` import across 78 tests.
- TLS storage path `~/.airproxy/` (none yet) → planned `~/.ai-engine/` in v0.2 design.
- README, CLI help strings, log fields, file headers.
- Single atomic commit; clippy + full test suite must pass at HEAD.

### Core v0.2 deliverables

1. **`ai-engine-runtime`** — burn-based parameterized transformer + safetensors loader. Llama-3 + Mistral + Qwen-2.5 + DeepSeek-V2 (dense parts). bf16/f16/f32. CPU + CUDA + Metal + WebGPU backends compiled in.
2. **`ai-engine-tokenizer`** — wraps `tokenizers` crate; loads `tokenizer.json` from HF dumps.
3. **`ai-engine-cluster`** — QUIC transport (quinn + rustls + auto-self-signed certs pinned by SHA-256), control + data plane protocol, capability-aware DP partitioner with manual override, leader + worker state machines, `ClusterProvider` impl of the existing `Provider` trait.
4. **Config schema extension** in `ai-engine-config`: `[[cluster]]`, `[[cluster.node]]`, `[[cluster.model]]`, `[[cluster.partition_override]]`, plus `[[provider]] kind = "local-cluster"`.
5. **Binary integration** — `build_app_state` learns cluster mode; worker mode skips axum routes beyond `/healthz`+`/readyz`; CLI gains `--node-id` override and `ai-engine node fingerprint` subcommand.
6. **Test suite** — ~55 new tests across 6 layers per §9, with toy-llama fixture, in-process distributed correctness gate, multi-process smoke (`#[ignore]`), load smoke (`#[ignore]`).
7. **Documentation** — README section on running a cluster, runbook for multi-machine validation, updated architecture diagram, this design spec.

### Deferred

**To v0.3 (sub-project #4 — Cluster operations & ergonomics):**
- mDNS / Bonjour auto-discovery.
- TOFU certificate exchange.
- Dynamic membership.
- Worker failover with KV cache reconstruction.
- Hot-reload of cluster topology.
- Per-cluster multi-model support.
- Web playground UI.

**To v0.4+ (sub-project #8 — Inference performance):**
- Quantization (Q4/Q5/Q8/AWQ/GPTQ).
- Continuous batching.
- Paged attention.
- Speculative decoding.
- Tensor parallelism.

**Indefinitely deferred:**
- Training, fine-tuning, LoRA serving.
- MoE expert routing.
- Multimodal.
- ROCm, TPU, custom accelerator backends.

### Renumbered roadmap

| Old # | New # | Sub-project |
|---|---|---|
| — | #2 | **Distributed inference coordinator (v0.2 — this spec)** |
| #2 | #3 | Auth & keys (DB-backed users, teams, OIDC) |
| #3 | #4 | Cluster operations (mDNS, dynamic membership, failover) |
| #4 | #5 | Limits & quotas (rate limit, budgets, cache) |
| #5 | #6 | Observability (Prometheus, request-log persistence, Langfuse) |
| #6 | #7 | Resilience (fallbacks, retries, circuit breakers) |
| #7 | #8 | Inference performance (quantization, batching, paged attention, speculative) |
| #8 | #9 | PII scrubber |
| #9 | #10 | RAG / knowledge base |
| #10 | #11 | Admin API & ops |

### Acceptance criteria — when is v0.2 "done"?

v0.2.0 is tagged when **all** of the following are true:

1. Toy-llama bytes-exact gate passes on `cargo test --workspace`.
2. In-process 3-node cluster generates a chat completion identical to single-node baseline.
3. Multi-process smoke test passes (`cargo test --release -- --ignored`).
4. Load smoke test passes (≥99% success, ≥4 events/stream on the toy model).
5. Manual multi-machine validation (Layer 6) performed once and recorded in the runbook.
6. `ai-engine --check` validates a real-world cluster config containing Llama-3-70B across 3+ nodes.
7. README documents how an operator stands up a cluster end-to-end.
8. All v0.1 tests still pass (no regressions in the gateway story).
9. Apache-2.0 `LICENSE` + `NOTICE` remain in place.
10. Clippy clean across the full workspace with `--all-targets`.

### Estimated effort

- **Prerequisite rename**: 1 day (mechanical).
- **`ai-engine-tokenizer`**: 1–2 days (mostly wrapping).
- **`ai-engine-runtime` model layer + safetensors loader**: 6–10 weeks (heaviest piece — GQA, RoPE, KV cache, family adapters, bytes-exact correctness gate).
- **`ai-engine-cluster` protocol + partitioner + QUIC**: 4–6 weeks.
- **Integration + leader/worker state machines + generation loop**: 3–5 weeks.
- **Test suite (esp. multi-machine validation runbook)**: 2–3 weeks.
- **Hardening + bug-bashing + docs**: 3–4 weeks.

**Total estimate: 5–7 months of focused work.**

---

## 11. Open questions

None blocking implementation. Items intentionally deferred (mDNS discovery, dynamic membership, quantization, fault tolerance, paged attention, MoE, multimodal) will be reopened in their respective sub-project specs.

---

## 12. Approval

- Brainstorming sections 1–10 reviewed and approved by user 2026-05-23.
- Pending: user review of this written spec, then transition to implementation plan via the `writing-plans` skill.
