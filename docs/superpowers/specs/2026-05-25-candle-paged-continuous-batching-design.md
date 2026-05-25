# Candle Paged Continuous-Batching Engine — Design

Date: 2026-05-25
Status: Approved (brainstorming complete)
Crate: `ai-engine-candle`

## Goal

Add true continuous batching to the `kind = "candle-local"` provider via a
vLLM-style paged KV cache, so one model instance time-shares many in-flight
sequences instead of locking one of N independent replicas per request.

Covers the three currently-supported architectures: **llama, qwen2, qwen3**.
Architectures the engine cannot load fall back to the existing pool-of-N path.

## Background & Constraint

candle-transformers 0.10.2 `quantized_*::ModelWeights::forward(x, index_pos)`
takes a **single** `index_pos` shared across the whole batch and owns an
internally-bundled batch=1 KV cache. RoPE is applied with
`cos.narrow(0, index_pos, seq_len)` — one position for every row — so the stock
API physically forces all sequences in a batch to share one position. There is
no PagedAttention / continuous-batch / batched-generate helper anywhere in
candle-transformers 0.10.2.

Therefore true continuous batching requires reimplementing the forward stack:
we cannot reuse `ModelWeights::forward`, because it owns the KV cache and the
shared `index_pos` we must replace.

## Decisions (locked during brainstorming)

- **KV architecture: paged (vLLM-style).** Block-table allocation, no contiguous
  padding waste in the memory model.
- **Compute: gather + candle ops.** No fused paged kernel (candle lacks one).
  Each step gathers KV blocks into contiguous padded tensors via `index_select`
  and runs standard candle attention. "Paged allocation, dense compute."
  Portable across CPU/CUDA/Metal, no FFI.
- **Arch scope: all three** (llama, qwen2 with QKV bias, qwen3 with q/k-norm).
- **One engine per model**, not a pool of engines.

## Architecture

### Concurrency-model shift

Today: `ReplicaPool` = N independently-loaded `CandleModel`s, one `Mutex`-locked
per request; throughput scales with replica count (and duplicated weights).

New: **one model instance + a paged KV pool, driven by a single-threaded
scheduler step loop**; throughput scales with batch size. The forward is not
safe to run concurrently on one device, so the step loop is the serialization
point. The existing `CandleModel` + `ReplicaPool` path is retained as a
fallback (`engine = "pool"`) and as the correctness oracle.

### New module layout (`crates/ai-engine-candle/src/`)

| Unit | Responsibility | Depends on |
|------|---------------|------------|
| `paged/block_table.rs` | Fixed-size KV block pool: alloc/free/reuse, per-seq block lists, OOM signaling | candle tensors |
| `paged/attention.rs` | Gather KV blocks → padded contiguous tensors; per-row RoPE; per-row causal+padding mask; scaled-dot-product attention (with GQA expansion) | block_table |
| `paged/transformer.rs` | Generic quantized forward (embed → RMSNorm → attn → SwiGLU → final norm → lm_head), parameterized by arch config | attention, candle `QMatMul` |
| `paged/arch.rs` | Arch config: llama / qwen2 (QKV bias) / qwen3 (q/k-norm); rope_theta, dims, norm kind, GGUF tensor-name mapping | gguf metadata |
| `paged/engine.rs` | Scheduler: waiting queue, running set, admission/eviction, step loop, per-seq token channels | transformer, block_table |
| `provider.rs` (edit) | Route `candle-local` through the engine; new config knobs; stream tokens to SSE | engine |

A single generic `transformer.rs` driven by `arch.rs` avoids three
near-duplicate model files; families differ only in a few attention/norm hooks.

## Paged KV cache & gather attention

### Block pool

KV stored as fixed-size blocks of `block_size` tokens (default 16). Per layer,
two f32 tensors `K_pool`, `V_pool` of shape
`(num_blocks, block_size, n_kv_head, head_dim)`. (Activations are full
precision; only *weights* are quantized.) A free-list of block ids; each
sequence owns an ordered `Vec<u32>` block table. Appending token `T` to a
sequence writes into `block_table[T / block_size]` at slot `T % block_size`.
Allocation pulls from the free-list; eviction returns blocks. Block-pool
exhaustion → the request stays queued (backpressure); it never corrupts a
running batch.

### Gather → dense compute (per decode step)

1. `L_max` = longest active sequence in the batch.
2. For each sequence, `index_select` its KV blocks from the pool into a
   contiguous `(seq, n_kv_head, head_dim)` slice, right-pad to `L_max`.
3. Stack into `(batch, L_max, …)`.
4. Build a **per-row key-padding + causal mask** so each row attends only to
   its own real keys (and, at prefill, causally within its prompt).
5. Standard scaled-dot-product attention with GQA head expansion.

The block table provides zero-fragmentation allocation and dynamic join/evict;
compute materializes padded tensors per step (the accepted cost vs a fused
kernel).

### Per-row RoPE (the crux of the fork)

Instead of candle's `cos.narrow(0, index_pos, seq_len)` (one position), build a
per-row position index `[pos_0, pos_1, …]` and `index_select` rows from the
precomputed `cos`/`sin` tables, so a sequence joining at position 0 and one at
position 200 rotate correctly within the same batch.

### Prefill (v1 simplification)

A newly-admitted request gets one prefill forward (`seq_len > 1`, writing its
KV blocks), **then** joins the uniform decode batch (`seq_len = 1` for all
running sequences). Mixing prefill with decode in one step (chunked prefill) is
explicitly deferred. Cost: a long prefill stalls that single step.

## Scheduler, engine loop & provider integration

The engine owns one model + the block pool and runs a single step loop on a
dedicated task. Public async interface:

```
engine.submit(prompt, GenParams) -> impl Stream<Item = Result<u32>>
```

### Step loop (one iteration = one model forward)

1. **Admit**: while `running.len() < max_num_seqs` and enough free blocks exist
   for the prompt, pull a waiting request → prefill forward → first token
   sampled/emitted → move to `running`.
2. **Decode**: one batched forward over all `running` sequences (uniform
   `seq_len = 1`), per-row positions + per-row mask.
3. **Sample** per row (reuse `ai_engine_runtime::sample`, temperature from each
   sequence's `GenParams`); emit each token on that sequence's channel.
4. **Evict**: any sequence hitting EOS or `max_tokens` → free its blocks, close
   its channel.

A per-sequence error fails only that sequence's channel; the batch keeps
running.

### Provider integration (`provider.rs`)

`kind = "candle-local"` constructs one `Engine` instead of a `ReplicaPool`.
`chat_stream` renders the prompt (existing chat-template path), calls
`engine.submit`, and forwards tokens as SSE deltas. New config knobs:

```toml
[[model]]
kind = "candle-local"
weights_path = "..."
device = "auto"
engine = "paged"        # "paged" (new, default for supported arches) | "pool"
max_num_seqs = 32       # max concurrent sequences in a batch
block_size = 16         # tokens per KV block
kv_cache_blocks = 4096  # block pool size (caps total KV memory)
```

`engine = "pool"` keeps today's `ReplicaPool` (and `pool_size`); arches the
paged engine cannot load also fall back to it.

### Flow control

Block-pool exhaustion produces **backpressure** (requests wait in the queue),
not rejection — the scheduler admits them as running sequences free their
blocks. `kv_cache_blocks` is the memory-bounding knob.

## Correctness & testing

The correctness gate is **greedy token-parity** against the existing path: at
`temperature = 0`, the paged engine running a single sequence must produce the
exact same token IDs as today's `CandleModel` for the same prompt, per arch.
`CandleModel`/`ReplicaPool` is shipped and validated, so token-for-token
equality proves the forward reimplementation and paged KV are correct. Greedy
decoding is deterministic, so this is exact (not 1e-3-tolerant).

| Layer | What it proves |
|-------|---------------|
| Unit: block table | alloc/free/reuse, free-list integrity, OOM returns cleanly (no panic) |
| Unit: per-row RoPE gather | a 2-row batch at positions {0, 200} rotates each row identically to candle's single-position `narrow` for that position |
| Unit: per-row mask + gather attention | padded-gather attention equals a dense reference for known small K/V |
| Integration: greedy parity (the gate) | paged engine single-seq output == `CandleModel` output, token-for-token, per arch |
| Integration: continuous batching | N concurrent varied-length prompts each yield the same greedy tokens as run-alone; logs confirm interleaved admission/eviction |
| Smoke: throughput | aggregate tok/s under concurrent load vs pool-of-N, on real GGUF fixtures |

## Out of scope (v1, deferred)

- Chunked prefill (long prefill stalls its step — accepted).
- Fused CUDA paged-attention kernel (gather path is portable; kernel is a later
  optional feature flag).
- Multi-engine / multi-GPU sharding (one engine per model).
- Prefix / KV-cache sharing across sequences (block table makes this possible
  later, not in v1).
- Weight sharing beyond what one engine gives (one engine already loads weights
  once, solving the old per-replica duplication for the batched path).
