# Plan 4 — v0.2.1: streaming, concurrency, real partition Assignment

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the three concrete v0.2.0 limitations: (1) per-token SSE streaming on `chat_stream`, (2) concurrent requests through one leader, (3) workers loading their layer range from a leader-broadcast `Assignment` instead of the current even-split self-computation.

**Architecture:** Three independent refactors riding the existing wire protocol — no new transport, no new control messages (Assignment is already defined in §6 of the spec; we just wire it for real). Concurrency comes from cloning the per-worker `quinn::Connection` so each request task gets its own stream pool; the leader's per-block KV state moves from `&mut self` into per-request `RequestSession` objects. Streaming wires `ClusterLeader::generate` to yield tokens through an mpsc channel that `ClusterProvider::chat_stream` adapts into the existing `EventStream<openai::ChatStreamEvent>` type.

**Tech Stack:** No new external crates. Uses tokio mpsc, futures::Stream, the existing quinn / postcard infrastructure.

**Scope rule:** v0.2.1 ships these three improvements only. mDNS discovery, dynamic membership, web playground, quantization, tensor parallelism are all sub-project #4+ work and explicitly NOT in this plan.

**Baseline:** Branch `main` at `v0.2.0`. 150 tests + 3 ignored (multiproc smoke, load smoke, backend parity), all green.

---

## File structure

```
crates/
├── ai-engine-cluster/
│   └── src/
│       ├── worker.rs                    # MODIFY: run_worker_full waits for Assignment
│       ├── leader.rs                    # MODIFY: start() broadcasts Assignment; add per-request session API
│       ├── session.rs                   # NEW: RequestSession holding per-request leader-side state
│       ├── partition.rs                 # MODIFY: extract NodeAssignment::for_node(node_id) helper
│       └── provider.rs                  # MODIFY: chat_stream impl over autoregressive generation
└── ai-engine/
    └── tests/
        └── streaming_smoke.rs           # NEW: multi-process SSE smoke test
```

No new crates. Refactor surface area:
- `ai-engine-cluster::worker::run_worker_full` — replace `layer_range` parameter with `model_path` + `cfg` only; load weights AFTER receiving Assignment from leader.
- `ai-engine-cluster::leader::ClusterLeader::start` — after computing manifest, send `Assignment { manifest, model_id }` to each worker on the control bidi stream.
- `ai-engine-cluster::leader::ClusterLeader::generate` — change `&mut self` → `&self`; create a per-call `RequestSession` that owns the cloned worker connections + the per-request leader-side state (KV caches, current_pos).
- `ai-engine-cluster::session::RequestSession<B>` — new type bundling: `Arc<Vec<DecoderBlock<B>>>` (shared leader blocks, immutable), `Vec<KvCacheSlot<B>>` (per-request, mutable), `current_pos: usize`, `worker_connections: Vec<quinn::Connection>` (cheap clones).
- `ai-engine-cluster::provider::ClusterProvider::chat_stream` — replace the `Unsupported` stub with a real impl that spawns the generation loop in a task, yields ChatStreamEvent per token, terminates with `[DONE]` semantics.

---

## Important pre-flight notes

- **`quinn::Connection::clone` is cheap** (Arc internally). Each request task gets its own clone — same underlying connection, independent stream pool. quinn handles concurrent `open_uni()` / `accept_uni()` correctly across clones.
- **Per-request state goes in `RequestSession`.** What was in `ClusterLeader::generate`'s body (leader blocks, KV caches, current_pos) moves into `RequestSession`. The leader holds shared, immutable artifacts (the loaded weights, embedding, output projection) that can be `Arc`-shared across sessions.
- **Workers already key KV caches by `request_id: Uuid`** — that part requires no change. The leader simply mints distinct request_ids for concurrent requests, and the workers' `HashMap<Uuid, Vec<KvCacheSlot>>` keeps them isolated.
- **Assignment-over-QUIC is mostly already wired** — `LeaderToWorker::Assignment` exists; `ClusterLeader::start` collects capabilities; only the *send* + *worker-side receive-then-load* steps are missing.
- **The toy fixture's even-split happens to match the auto_partition output** in the 3-node test (4 layers / 2 workers = 2 each). To prove Assignment really works, the test should use a partition that DIFFERS from even-split — e.g., manual_partition override with `[(w1, 0..3), (w2, 3..4)]`, asymmetric on purpose. The worker that gets the smaller range proves it loaded the correct layers.

---

### Task 1: Extract `NodeAssignment::for_node` helper

**Files:**
- Modify: `crates/ai-engine-cluster/src/partition.rs`
- Modify: `crates/ai-engine-cluster/tests/partition_solver.rs` (add one test)

A small helper that lets a worker pull its assigned `Range<usize>` out of a `PartitionManifest` given its node id. Used by both Task 2 (worker) and Task 3 (test verifying the right layers loaded).

- [ ] **Step 1: Failing test**

Add to `crates/ai-engine-cluster/tests/partition_solver.rs`:

```rust
#[test]
fn for_node_returns_assignment_for_known_node() {
    let caps = vec![cap("a", 16, 100), cap("b", 16, 100)];
    let m = ai_engine_cluster::partition::auto_partition(
        "m", &caps, 12, 1024*1024*1024, 1024*1024*1024, 256*1024*1024,
    ).unwrap();
    let a = m.for_node("a").unwrap();
    assert_eq!(a.node_id, "a");
    let b = m.for_node("b").unwrap();
    assert_eq!(b.node_id, "b");
    assert!(m.for_node("missing").is_none());
}
```

- [ ] **Step 2: Confirm fails**

```bash
cd /home/alessio/aip/airproxy
cargo test -p ai-engine-cluster --test partition_solver
# Expected: compile error — for_node not defined.
```

- [ ] **Step 3: Implement**

Add to `crates/ai-engine-cluster/src/partition.rs`:

```rust
impl PartitionManifest {
    /// Find the assignment for a given node id. Returns `None` if the node
    /// isn't in the manifest.
    pub fn for_node(&self, node_id: &str) -> Option<&NodeAssignment> {
        self.assignments.iter().find(|a| a.node_id == node_id)
    }
}
```

- [ ] **Step 4: Verify + commit**

```bash
cargo test -p ai-engine-cluster --test partition_solver
cargo clippy --workspace --all-targets -- -D warnings
git add -A
git commit -m "feat(cluster): PartitionManifest::for_node helper"
```

NO Co-Authored-By footer (global preference).

---

### Task 2: Worker waits for Assignment before loading weights

**Files:**
- Modify: `crates/ai-engine-cluster/src/worker.rs`
- Modify: `crates/ai-engine-cluster/src/leader.rs` (send Assignment after partition)
- Modify: `crates/ai-engine-cluster/tests/leader_partition.rs` (verify Assignment is received by workers)

The current `run_worker_full(endpoint, node_id, backend, model_path, cfg, layer_range)` takes the layer_range as a parameter — that's the v0.2.0 workaround. v0.2.1 makes the worker WAIT for an `Assignment` control message and pull the range from there.

The leader currently computes the manifest in `ClusterLeader::start` but doesn't send it. v0.2.1 sends `LeaderToWorker::Assignment { manifest, model_id }` on each worker's control bidi stream right after collecting capabilities.

- [ ] **Step 1: Failing test**

Modify `crates/ai-engine-cluster/tests/leader_partition.rs` to spawn workers that explicitly check the Assignment they receive:

```rust
use ai_engine_cluster::protocol::codec::decode;
use ai_engine_cluster::protocol::control::LeaderToWorker;
use ai_engine_cluster::transport::frame::read_frame;
use tokio::sync::oneshot;

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn workers_receive_correct_assignment_from_leader() {
    let w1_id = ai_engine_cluster::tls::generate_node_identity("w1").unwrap();
    let w1_ep = ai_engine_cluster::transport::quic::server_endpoint(
        &w1_id, "127.0.0.1:0".parse().unwrap()
    ).unwrap();
    let w1_addr = w1_ep.local_addr().unwrap();
    let w2_id = ai_engine_cluster::tls::generate_node_identity("w2").unwrap();
    let w2_ep = ai_engine_cluster::transport::quic::server_endpoint(
        &w2_id, "127.0.0.1:0".parse().unwrap()
    ).unwrap();
    let w2_addr = w2_ep.local_addr().unwrap();

    // For this test we replicate just the handshake-and-Assignment-receive portion
    // of run_worker_full, so we can capture the assigned range without loading
    // any model weights.
    let (w1_tx, w1_rx) = oneshot::channel();
    let (w2_tx, w2_rx) = oneshot::channel();
    tokio::spawn(handshake_and_capture_assignment(w1_ep, "w1".into(), w1_tx));
    tokio::spawn(handshake_and_capture_assignment(w2_ep, "w2".into(), w2_tx));

    let leader_id = ai_engine_cluster::tls::generate_node_identity("leader").unwrap();
    let cfg = ai_engine_cluster::leader::LeaderConfig {
        cluster_id: "test".into(),
        leader_node_id: "leader".into(),
        model_id: "toy".into(),
        n_layers: 4,
        layer_bytes: 1024 * 1024,
        embed_output_bytes: 1024 * 1024,
        per_node_overhead: 256 * 1024,
        workers: vec![
            ai_engine_cluster::leader::WorkerEndpoint {
                node_id: "w1".into(), addr: w1_addr,
                fingerprint: w1_id.fingerprint.clone()
            },
            ai_engine_cluster::leader::WorkerEndpoint {
                node_id: "w2".into(), addr: w2_addr,
                fingerprint: w2_id.fingerprint.clone()
            },
        ],
    };
    let leader = ai_engine_cluster::leader::ClusterLeader::start(leader_id, cfg).await.unwrap();
    let manifest = leader.manifest();

    // Each worker should have received an Assignment whose embedded manifest
    // contains its expected range. With auto-partition on 4 layers / 2 nodes,
    // each worker gets 2 layers contiguously.
    let assn_w1 = tokio::time::timeout(std::time::Duration::from_secs(5), w1_rx)
        .await.unwrap().unwrap();
    let assn_w2 = tokio::time::timeout(std::time::Duration::from_secs(5), w2_rx)
        .await.unwrap().unwrap();

    let w1_range = manifest.for_node("w1").unwrap().layer_range.clone();
    let w2_range = manifest.for_node("w2").unwrap().layer_range.clone();
    assert_eq!(assn_w1, w1_range);
    assert_eq!(assn_w2, w2_range);
}

async fn handshake_and_capture_assignment(
    endpoint: quinn::Endpoint,
    node_id: String,
    out: oneshot::Sender<std::ops::Range<usize>>,
) -> anyhow::Result<()> {
    let incoming = endpoint.accept().await
        .ok_or_else(|| anyhow::anyhow!("no incoming"))?;
    let conn = incoming.await?;
    let (mut send, mut recv) = conn.accept_bi().await?;

    // Read Join, write JoinAck + Capability.
    let _join: LeaderToWorker = decode(&read_frame(&mut recv).await?)?;
    let ack = ai_engine_cluster::protocol::control::WorkerToLeader::JoinAck {
        node_id: node_id.clone(),
        certificate_sha256: [0u8; 32],
    };
    ai_engine_cluster::transport::frame::write_frame(
        &mut send, &ai_engine_cluster::protocol::codec::encode(&ack)?
    ).await?;
    let cap = ai_engine_cluster::capability::detect_capability(
        &node_id, ai_engine_cluster::capability::BackendKind::Cpu, 0, None,
    )?;
    ai_engine_cluster::transport::frame::write_frame(
        &mut send,
        &ai_engine_cluster::protocol::codec::encode(
            &ai_engine_cluster::protocol::control::WorkerToLeader::Capability(cap)
        )?
    ).await?;

    // Read Assignment, extract our range.
    let assn: LeaderToWorker = decode(&read_frame(&mut recv).await?)?;
    let range = if let LeaderToWorker::Assignment { manifest, .. } = assn {
        manifest.for_node(&node_id)
            .ok_or_else(|| anyhow::anyhow!("no assignment for {node_id}"))?
            .layer_range.clone()
    } else {
        anyhow::bail!("expected Assignment, got {:?}", assn)
    };
    let _ = out.send(range);
    Ok(())
}
```

- [ ] **Step 2: Confirm fails**

The test should hang (the leader currently doesn't send Assignment, so the worker's `read_frame` for it blocks forever). The 5s timeouts inside the test fire and the assertions fail.

- [ ] **Step 3: Modify `ClusterLeader::start` to send Assignment**

In `crates/ai-engine-cluster/src/leader.rs`, after the manifest is computed inside `start`:

```rust
// Phase 3: distribute Assignment to each worker.
for (i, wc) in connections.iter_mut().enumerate() {
    let assignment = LeaderToWorker::Assignment {
        manifest: manifest.clone(),
        model_id: cfg.model_id.clone(),
    };
    crate::transport::frame::write_frame(
        &mut wc.control_send,
        &crate::protocol::codec::encode(&assignment)?,
    ).await?;
    let _ = i;   // index unused; kept for future per-worker variation
}
```

Add the `use crate::protocol::control::LeaderToWorker;` import at the top of leader.rs if not present.

`PartitionManifest` is already `Clone + Serialize + Deserialize` from Task 5 of Plan 2 — no changes needed.

- [ ] **Step 4: Modify `run_worker_full` to wait for Assignment**

Change the signature in `crates/ai-engine-cluster/src/worker.rs`:

```rust
pub async fn run_worker_full<B>(
    endpoint: quinn::Endpoint,
    node_id: String,
    backend: crate::capability::BackendKind,
    model_path: std::path::PathBuf,
    cfg: ai_engine_runtime::config::ModelConfig,
    // layer_range removed; learned from Assignment.
) -> anyhow::Result<()>
where
    B: burn::tensor::backend::Backend,
    B::Device: Default,
{
    // ... existing handshake (Join → JoinAck → Capability) ...

    // NEW: wait for Assignment.
    let assn_bytes = crate::transport::frame::read_frame(&mut recv).await?;
    let assn: crate::protocol::control::LeaderToWorker =
        crate::protocol::codec::decode(&assn_bytes)?;
    let layer_range = match assn {
        crate::protocol::control::LeaderToWorker::Assignment { manifest, .. } => {
            manifest.for_node(&node_id)
                .ok_or_else(|| anyhow::anyhow!("no assignment for {node_id} in manifest"))?
                .layer_range.clone()
        }
        other => anyhow::bail!("expected Assignment, got {other:?}"),
    };

    // ... rest: load_range, build blocks, serving loop ... (existing code)
}
```

Note: `LeaderToWorker::Assignment.manifest` carries `PartitionManifest`. `for_node` from Task 1 gives us the right `&NodeAssignment`, whose `layer_range: Range<usize>` is what we need.

- [ ] **Step 5: Update callers of `run_worker_full`**

In `crates/ai-engine/src/worker_main.rs`, drop the `compute_my_layer_range` call (which becomes dead code) and the `layer_range` argument to `run_worker_full`. The new call is:

```rust
ai_engine_cluster::worker::run_worker_full::<burn_ndarray::NdArray>(
    endpoint, node_id.to_string(), backend, model_path, model_cfg,
).await
```

Also delete the `compute_my_layer_range` function and its `ai_engine_config::Cluster` / `ClusterNode` imports if they become unused.

In `crates/ai-engine-cluster/tests/inprocess_cluster.rs` (the existing 3-node tests from Plan 2 + Plan 3), update the `run_worker_full` calls to remove the `layer_range` argument. The leader will tell each worker its range. Since the leader uses `auto_partition` on the deterministic config-ordered capabilities, and capability microbenchmarks may produce slightly different scores per run, the manifest could differ between runs. For test stability, swap to `manual_partition` in those tests:

Actually, simpler: the leader's `LeaderConfig` doesn't expose a partition override. Add one:

```rust
pub struct LeaderConfig {
    // ... existing fields ...
    /// Optional explicit partition. When Some, bypasses auto_partition.
    pub partition_override: Option<Vec<(String, std::ops::Range<usize>)>>,
}
```

`ClusterLeader::start` uses `manual_partition` when `partition_override` is provided, otherwise `auto_partition` as today.

In the existing tests that hard-code worker layer ranges (e.g., `inprocess_cluster.rs`'s `three_node_cluster_logits_match_single_node`), populate `partition_override = Some(vec![("w1".into(), 1..3), ("w2".into(), 3..4)])` so the test continues to use the same partition that previously worked.

- [ ] **Step 6: Verify all cluster tests pass**

```bash
cargo test -p ai-engine-cluster
cargo clippy --workspace --all-targets -- -D warnings
```

Particularly verify `tests/inprocess_cluster.rs` tests still pass — those exercise the full forward + generation paths with the new Assignment flow.

The multi-process smoke (`tests/multiproc_smoke.rs`) also needs adjustment: workers now learn their range from Assignment, so the leader's `LeaderConfig` either needs `partition_override` or uses auto_partition (auto's deterministic given matching capabilities, which 3 CPU nodes on one box should give since the microbenchmark variance is small but non-zero). Safer: add `partition_override` to the smoke config.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat(cluster): workers receive layer range via Assignment over QUIC; partition_override in LeaderConfig"
```

NO Co-Authored-By.

---

### Task 3: End-to-end test — asymmetric partition through Assignment

**Files:**
- Modify: `crates/ai-engine-cluster/tests/inprocess_cluster.rs` (add asymmetric partition test)

The existing 3-node test uses an even-ish split (leader 0..1, w1 1..3, w2 3..4). To prove the Assignment path really works (vs accidentally matching a default), add a test with a deliberately asymmetric override.

- [ ] **Step 1: Test**

```rust
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn asymmetric_partition_via_assignment_matches_single_node() {
    // Toy llama has 4 layers. Use asymmetric partition: w1 takes 0..3 (three layers),
    // w2 takes 3..4 (one layer). Leader hosts no layers (0..0).
    //
    // Baseline = single-node forward. Cluster output must match.

    let fix = fixture();
    let cfg = ai_engine_runtime::config::ModelConfig::from_file(&fix.join("config.json")).unwrap();
    let tok = ai_engine_tokenizer::HfTokenizer::from_path(fix.join("tokenizer.json")).unwrap();
    let prompt = "The quick brown fox";
    let ids: Vec<u32> = ai_engine_tokenizer::Tokenizer::encode(&tok, prompt).unwrap();
    let ids_i32: Vec<i32> = ids.iter().map(|x| *x as i32).collect();

    let baseline_tokens = single_node_greedy_5::<burn_ndarray::NdArray>(&fix, &cfg, &ids_i32);

    // Spin up 2 workers with no preset layer_range — they'll get it from Assignment.
    let w1_id = ai_engine_cluster::tls::generate_node_identity("w1").unwrap();
    let w1_ep = ai_engine_cluster::transport::quic::server_endpoint(
        &w1_id, "127.0.0.1:0".parse().unwrap()
    ).unwrap();
    let w1_addr = w1_ep.local_addr().unwrap();
    let w2_id = ai_engine_cluster::tls::generate_node_identity("w2").unwrap();
    let w2_ep = ai_engine_cluster::transport::quic::server_endpoint(
        &w2_id, "127.0.0.1:0".parse().unwrap()
    ).unwrap();
    let w2_addr = w2_ep.local_addr().unwrap();

    let model_path = fix.join("model.safetensors");
    let cfg_for_w1 = cfg.clone();
    let mp1 = model_path.clone();
    tokio::spawn(async move {
        ai_engine_cluster::worker::run_worker_full::<burn_ndarray::NdArray>(
            w1_ep, "w1".to_string(),
            ai_engine_cluster::capability::BackendKind::Cpu,
            mp1, cfg_for_w1,
        ).await
    });
    let cfg_for_w2 = cfg.clone();
    let mp2 = model_path.clone();
    tokio::spawn(async move {
        ai_engine_cluster::worker::run_worker_full::<burn_ndarray::NdArray>(
            w2_ep, "w2".to_string(),
            ai_engine_cluster::capability::BackendKind::Cpu,
            mp2, cfg_for_w2,
        ).await
    });

    let leader_id = ai_engine_cluster::tls::generate_node_identity("leader").unwrap();
    let lcfg = ai_engine_cluster::leader::LeaderConfig {
        cluster_id: "test".into(),
        leader_node_id: "leader".into(),
        model_id: "toy".into(),
        n_layers: cfg.n_layers,
        layer_bytes: 256 * 1024,
        embed_output_bytes: 256 * 1024,
        per_node_overhead: 64 * 1024,
        workers: vec![
            ai_engine_cluster::leader::WorkerEndpoint {
                node_id: "w1".into(), addr: w1_addr,
                fingerprint: w1_id.fingerprint.clone()
            },
            ai_engine_cluster::leader::WorkerEndpoint {
                node_id: "w2".into(), addr: w2_addr,
                fingerprint: w2_id.fingerprint.clone()
            },
        ],
        // ASYMMETRIC: w1 takes 0..3, w2 takes 3..4.
        partition_override: Some(vec![
            ("w1".to_string(), 0..3),
            ("w2".to_string(), 3..4),
        ]),
    };

    let leader = ai_engine_cluster::leader::ClusterLeader::start(leader_id, lcfg).await.unwrap();
    let cluster_tokens = leader.generate::<burn_ndarray::NdArray>(
        &model_path, &cfg,
        0..0,    // leader hosts no layers; all 4 live on workers
        &ids_i32,
        /*max_tokens=*/5,
        ai_engine_runtime::sample::SamplingConfig {
            temperature: 0.0, top_p: None, top_k: None, seed: 0,
        },
    ).await.unwrap();

    assert_eq!(
        cluster_tokens, baseline_tokens,
        "asymmetric partition via Assignment must match single-node baseline"
    );
}
```

Note: this test deliberately uses `leader_layers = 0..0` (leader hosts no model layers; it just embeds + does output projection). w1 gets 3 layers, w2 gets 1. The Assignment path is the only thing telling the workers what to load — if it's broken, w1 loads layers 1..3 (the old hard-coded value) and produces wrong output.

- [ ] **Step 2: Implement (already done by Task 2)**

Just run the test:

```bash
cargo test -p ai-engine-cluster --test inprocess_cluster
```

The test should pass: with the new Assignment flow, w1 loads layers 0..3 and w2 loads layer 3..4 from disk based on what the leader tells them.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "test(cluster): asymmetric partition via Assignment matches single-node baseline"
```

NO Co-Authored-By.

---

### Task 4: `RequestSession` extraction for per-request leader state

**Files:**
- Create: `crates/ai-engine-cluster/src/session.rs`
- Modify: `crates/ai-engine-cluster/src/leader.rs` (refactor `generate` to use RequestSession; change `&mut self` → `&self`)
- Modify: `crates/ai-engine-cluster/src/lib.rs` (add `pub mod session;`)

The current `ClusterLeader::generate(&mut self, ...)` body loads weights, builds blocks, allocates KV caches, then drives the forward loop — all on the same call stack. The `&mut self` forces requests to serialize. Refactor:

1. `ClusterLeader` keeps shared, immutable resources: the worker `quinn::Connection`s (cloneable internally), the model weights (loaded once into Arc-shared structures).
2. `RequestSession<B>` holds per-request state: KV caches, current_pos, cloned worker connections (each request opens its own streams).
3. `generate(&self, ...)` builds a `RequestSession` and uses it.

- [ ] **Step 1: Build the new session type**

`crates/ai-engine-cluster/src/session.rs`:

```rust
use ai_engine_runtime::{
    arch::{block::DecoderBlock, embedding::{OutputProjection, TokenEmbedding}, rmsnorm::RmsNorm},
    config::ModelConfig,
    kv_cache::KvCacheSlot,
};
use burn::tensor::backend::Backend;
use quinn::Connection;
use std::sync::Arc;

/// Shared, immutable model artifacts loaded once per leader-cluster pair.
/// Multiple `RequestSession`s share the same `LeaderModel` cheaply (Arc).
pub struct LeaderModel<B: Backend> {
    pub embedding: TokenEmbedding<B>,
    pub blocks: Vec<DecoderBlock<B>>,
    pub final_norm: RmsNorm<B>,
    pub output: OutputProjection<B>,
    pub cfg: ModelConfig,
}

/// Per-request leader-side state. One `RequestSession` per concurrent request.
pub struct RequestSession<B: Backend> {
    pub model: Arc<LeaderModel<B>>,
    pub leader_caches: Vec<KvCacheSlot<B>>,
    pub current_pos: usize,
    pub worker_conns: Vec<Connection>,
    pub request_id: uuid::Uuid,
}

impl<B: Backend> RequestSession<B>
where
    B::Device: Default,
{
    pub fn new(
        model: Arc<LeaderModel<B>>,
        worker_conns: Vec<Connection>,
        device: &B::Device,
    ) -> Self {
        let cfg = &model.cfg;
        let leader_caches = (0..model.blocks.len()).map(|_| {
            KvCacheSlot::<B>::new(
                1, cfg.n_kv_heads, cfg.max_position_embeddings, cfg.head_dim, device,
            )
        }).collect();
        Self {
            model,
            leader_caches,
            current_pos: 0,
            worker_conns,
            request_id: uuid::Uuid::now_v7(),
        }
    }
}
```

- [ ] **Step 2: Refactor `ClusterLeader` to load model once + expose `LeaderModel`**

In `crates/ai-engine-cluster/src/leader.rs`, add a field:

```rust
pub struct ClusterLeader {
    manifest: PartitionManifest,
    pub connections: Vec<WorkerConnection>,
    /// Loaded once after the manifest is known (knows leader's layer range
    /// from `manifest.for_node(leader_node_id)`).
    /// `None` if this cluster has no leader-hosted layers in v0.2.0 — but for
    /// v0.2.1 we always load embedding + final_norm + output (boundary tensors)
    /// here regardless of whether the leader hosts any *layers*.
    pub model: Option<Arc<dyn std::any::Any + Send + Sync>>,
}
```

`model` is `Arc<dyn Any>` because `LeaderModel<B>` is generic and `ClusterLeader` can't be. The actual `LeaderModel<NdArray>` is downcast at use time. This is the price of mixing generic ML types with concrete networking types — Plan 5 might re-architect, but v0.2.1 accepts this pragmatically.

Alternative cleaner approach: lazy-load the model on first `generate` call rather than during `start`. This avoids the dyn Any dance entirely:

```rust
pub struct ClusterLeader {
    manifest: PartitionManifest,
    pub connections: Vec<WorkerConnection>,
    leader_node_id: String,
    model_path_for_lazy_load: Option<std::path::PathBuf>,
}

impl ClusterLeader {
    /// Lazy-load the model for the given backend B at first `generate` call.
    /// Subsequent calls reuse the same Arc.
    /// (Implementer: use `OnceCell<Arc<LeaderModel<B>>>` or a thread-local map,
    /// or just rebuild per session if the model is small enough — toy fixture
    /// is fast to load. v0.2.1 accepts the simpler "build per session" path;
    /// memoization is a future v0.3 optimization.)
    pub async fn build_session<B>(
        &self,
        model_path: &std::path::Path,
        cfg: &ai_engine_runtime::config::ModelConfig,
        leader_layers: std::ops::Range<usize>,
    ) -> anyhow::Result<crate::session::RequestSession<B>>
    where
        B: burn::tensor::backend::Backend,
        B::Device: Default,
    {
        let leader_model = build_leader_model::<B>(model_path, cfg, leader_layers).await?;
        let worker_conns: Vec<quinn::Connection> = self.connections.iter()
            .map(|wc| wc.conn.clone()).collect();
        let device = B::Device::default();
        Ok(crate::session::RequestSession::new(
            std::sync::Arc::new(leader_model), worker_conns, &device,
        ))
    }
}

async fn build_leader_model<B>(
    model_path: &std::path::Path,
    cfg: &ai_engine_runtime::config::ModelConfig,
    leader_layers: std::ops::Range<usize>,
) -> anyhow::Result<crate::session::LeaderModel<B>>
where
    B: burn::tensor::backend::Backend,
    B::Device: Default,
{
    use ai_engine_runtime::{
        arch::{
            attention::Attention, block::DecoderBlock,
            embedding::{OutputProjection, TokenEmbedding},
            ffn::SwiGluFfn, rmsnorm::RmsNorm, rope::RotaryEmbedding,
        },
        loader::load_range,
    };

    let device = B::Device::default();
    let weights = load_range::<B>(model_path, cfg, leader_layers.clone(), true, true, &device)?;
    let embedding = TokenEmbedding::new(weights.embedding.unwrap());
    let final_norm = RmsNorm::new(weights.final_norm.unwrap(), cfg.rms_norm_eps);
    let output = OutputProjection::new(embedding.weight.clone().swap_dims(0, 1));

    let mut blocks: Vec<DecoderBlock<B>> = Vec::with_capacity(leader_layers.len());
    for layer in weights.layers {
        let attn_norm = RmsNorm::new(layer.attn_norm, cfg.rms_norm_eps);
        let ffn_norm = RmsNorm::new(layer.ffn_norm, cfg.rms_norm_eps);
        let rope = RotaryEmbedding::new(
            cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta, &device,
        );
        let attn = Attention::new(
            layer.q_proj.swap_dims(0, 1), layer.k_proj.swap_dims(0, 1),
            layer.v_proj.swap_dims(0, 1), layer.o_proj.swap_dims(0, 1),
            rope, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim,
        );
        let ffn = SwiGluFfn::new(
            layer.ffn_gate.swap_dims(0, 1),
            layer.ffn_up.swap_dims(0, 1),
            layer.ffn_down.swap_dims(0, 1),
        );
        blocks.push(DecoderBlock { attn_norm, attn, ffn_norm, ffn });
    }

    Ok(crate::session::LeaderModel {
        embedding, blocks, final_norm, output,
        cfg: cfg.clone(),
    })
}
```

- [ ] **Step 3: Refactor `generate` to use RequestSession via `&self`**

Replace `ClusterLeader::generate(&mut self, ...)` with:

```rust
impl ClusterLeader {
    pub async fn generate<B>(
        &self,    // <-- &self, not &mut self
        model_path: &std::path::Path,
        cfg: &ai_engine_runtime::config::ModelConfig,
        leader_layers: std::ops::Range<usize>,
        prompt_ids: &[i32],
        max_tokens: usize,
        sampling: ai_engine_runtime::sample::SamplingConfig,
    ) -> anyhow::Result<Vec<u32>>
    where
        B: burn::tensor::backend::Backend,
        B::Device: Default,
    {
        let mut session: crate::session::RequestSession<B> = self.build_session(
            model_path, cfg, leader_layers,
        ).await?;

        // Inline the existing token loop, but operating on `session` instead of
        // local variables. The `step_through_cluster` helper (already extracted
        // in Plan 3 Task 4) takes `&mut [WorkerConnection]`; refactor it to take
        // `&[quinn::Connection]` (cloned in the session) so it doesn't need
        // mutable access to ClusterLeader.

        // Prefill
        let last_logits = step_through_cluster_session(&mut session, prompt_ids, false).await?;
        let mut produced: Vec<u32> = Vec::with_capacity(max_tokens);
        session.current_pos = prompt_ids.len();
        produced.push(ai_engine_runtime::sample::sample(&last_logits, &sampling));

        // Token loop
        for _ in 1..max_tokens {
            let last_token = *produced.last().unwrap() as i32;
            let last_logits = step_through_cluster_session(&mut session, &[last_token], false).await?;
            session.current_pos += 1;
            produced.push(ai_engine_runtime::sample::sample(&last_logits, &sampling));
        }

        Ok(produced)
    }
}

async fn step_through_cluster_session<B>(
    session: &mut crate::session::RequestSession<B>,
    token_ids: &[i32],
    is_terminal: bool,
) -> anyhow::Result<Vec<f32>>
where B: burn::tensor::backend::Backend, B::Device: Default
{
    use crate::protocol::data::{ActivationHeader, Dtype};
    use crate::tensor_io::{tensor_from_bytes, tensor_to_bytes};
    use burn::tensor::{Tensor, TensorData, Int};

    let device = B::Device::default();
    let cfg = &session.model.cfg;
    let seq = token_ids.len();
    let ids = Tensor::<B, 2, Int>::from_data(
        TensorData::new(token_ids.to_vec(), [1, seq]), &device,
    );
    let mut x = session.model.embedding.forward(ids);
    let positions: Vec<i32> = ((session.current_pos as i32)..((session.current_pos + seq) as i32)).collect();
    for (block, cache) in session.model.blocks.iter().zip(session.leader_caches.iter_mut()) {
        x = block.forward(x, &positions, cache);
    }

    // Send through each worker.
    for conn in &session.worker_conns {
        let (bytes, shape) = tensor_to_bytes(x)?;
        let header = ActivationHeader {
            request_id: session.request_id,
            seq_pos: session.current_pos as u32,
            shape: [shape[0] as u32, shape[1] as u32, shape[2] as u32],
            dtype: Dtype::F32,
            is_terminal,
        };
        let mut send_uni = conn.open_uni().await?;
        crate::transport::frame::write_frame(
            &mut send_uni,
            &crate::protocol::codec::encode(&header)?,
        ).await?;
        crate::transport::frame::write_frame(&mut send_uni, &bytes).await?;
        send_uni.finish()?;

        let mut recv_uni = conn.accept_uni().await?;
        let _hdr: ActivationHeader = crate::protocol::codec::decode(
            &crate::transport::frame::read_frame(&mut recv_uni).await?,
        )?;
        let payload = crate::transport::frame::read_frame(&mut recv_uni).await?;
        x = tensor_from_bytes::<B>(&payload, shape, &device)?;
    }

    let x = session.model.final_norm.forward(x);
    let logits = session.model.output.forward(x);
    let last = logits.slice([0..1, (seq - 1)..seq, 0..cfg.vocab_size])
        .reshape([cfg.vocab_size]);
    Ok(last.to_data().to_vec().map_err(|e| anyhow::anyhow!("to_vec: {e:?}"))?)
}
```

- [ ] **Step 4: Update existing `full_forward_for_test` similarly**

Either delete `full_forward_for_test` (it was test-only and its behavior is now covered by `generate` + the bytes-tolerant test) OR refactor it to use a `RequestSession` the same way. Deleting is simpler — check if any tests still use it; if so, update them to call `generate` with `max_tokens=1`.

- [ ] **Step 5: Update callers**

`crates/ai-engine-cluster/src/provider.rs` — `ClusterProvider::chat` calls `st.leader.generate(...)` but now `generate` is `&self`. The Mutex around `LeaderState` can stay (it serializes access to `LeaderState` itself, which is fine), or be replaced with `Arc<LeaderState>` since `generate` no longer needs `&mut self`. For v0.2.1 keep the Mutex — it's a small refactor.

Actually, removing the Mutex is the WHOLE POINT of this task. Change `LeaderState` to be held behind `Arc<LeaderState>` instead of `Arc<Mutex<LeaderState>>`:

```rust
pub struct ClusterProvider {
    id: String,
    is_leader: bool,
    state: Option<Arc<LeaderState>>,    // no Mutex!
}

impl ClusterProvider {
    pub fn new_leader_with_state(id: impl Into<String>, state: Arc<LeaderState>) -> Self {
        Self { id: id.into(), is_leader: true, state: Some(state) }
    }
}

#[async_trait::async_trait]
impl Provider for ClusterProvider {
    async fn chat(&self, req: openai::ChatRequest, _creds: &Credentials, ctx: &CallCtx)
        -> Result<openai::ChatResponse, ProviderError>
    {
        if !self.is_leader { return Err(ProviderError::Unsupported); }
        let st = self.state.as_ref().ok_or_else(|| {
            ProviderError::InvalidResponse("no leader state".into())
        })?;
        // No more `.lock().await` — st is an Arc<LeaderState>.
        let prompt_ids = ai_engine_tokenizer::Tokenizer::encode(&st.tokenizer, &render_prompt(&req))
            .map_err(|e| ProviderError::InvalidResponse(format!("tokenize: {e}")))?;
        let prompt_ids_i32: Vec<i32> = prompt_ids.iter().map(|x| *x as i32).collect();
        let max_tokens = req.max_tokens.unwrap_or(256) as usize;
        let sampling = ai_engine_runtime::sample::SamplingConfig {
            temperature: req.temperature.unwrap_or(1.0),
            top_p: None, top_k: None,
            seed: ctx.request_id.as_u128() as u64,
        };
        let tokens = st.leader.generate::<burn_ndarray::NdArray>(
            &st.model_path, &st.model_cfg, st.leader_layers.clone(),
            &prompt_ids_i32, max_tokens, sampling,
        ).await.map_err(|e| ProviderError::InvalidResponse(format!("generate: {e}")))?;

        let content = ai_engine_tokenizer::Tokenizer::decode(&st.tokenizer, &tokens)
            .map_err(|e| ProviderError::InvalidResponse(format!("decode: {e}")))?;
        Ok(/* same ChatResponse build */)
    }
    // chat_stream, messages, etc. — keep Unsupported for now (Task 6 wires chat_stream).
}
```

- [ ] **Step 6: Update build_app_state to construct Arc<LeaderState> (no Mutex)**

In `crates/ai-engine/src/app.rs`, the leader branch now does:

```rust
let provider_arc: Arc<dyn Provider> = Arc::new(
    ai_engine_cluster::provider::ClusterProvider::new_leader_with_state(
        cluster_id.clone(),
        Arc::new(state),    // no Mutex!
    ),
);
```

- [ ] **Step 7: Verify**

```bash
cargo test -p ai-engine-cluster
cargo test -p ai-engine
cargo clippy --workspace --all-targets -- -D warnings
```

All 150 baseline tests + 2 new ones from Tasks 1-3 should pass.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor(cluster): RequestSession extraction; ClusterLeader::generate is now &self"
```

NO Co-Authored-By.

---

### Task 5: Concurrent request test

**Files:**
- Create: `crates/ai-engine-cluster/tests/concurrent_requests.rs`

Now that `generate` is `&self`, multiple requests can fire in parallel. The QUIC connection clones share underlying state but quinn handles concurrent streams. Workers' per-request KV map (`HashMap<Uuid, Vec<KvCacheSlot>>`) keeps state isolated.

- [ ] **Step 1: Test**

```rust
//! Concurrent requests through a single cluster must produce isolated outputs.

mod common {
    // ... pull in helpers from inprocess_cluster.rs OR duplicate the cluster-spawn
    // template here. (For Plan 4 simplicity, duplicate — the alternative is
    // factoring a shared helper, which is more refactoring than this task warrants.)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn three_concurrent_requests_produce_isolated_outputs() {
    // 1. Spin up a 3-node cluster (leader hosts 0..0; w1 = 0..3; w2 = 3..4 via partition_override).
    // 2. Compute single-node baselines for three distinct prompts.
    // 3. Run all three through the cluster CONCURRENTLY via tokio::try_join!.
    // 4. Assert each matches its respective baseline.

    let fix = fixture();
    let cfg = ai_engine_runtime::config::ModelConfig::from_file(&fix.join("config.json")).unwrap();
    let tok = ai_engine_tokenizer::HfTokenizer::from_path(fix.join("tokenizer.json")).unwrap();
    let prompts = ["The quick brown fox", "Hello world", "ai engine cluster test"];

    let baselines: Vec<Vec<u32>> = prompts.iter().map(|p| {
        let ids: Vec<i32> = ai_engine_tokenizer::Tokenizer::encode(&tok, p).unwrap()
            .iter().map(|x| *x as i32).collect();
        single_node_greedy_5::<burn_ndarray::NdArray>(&fix, &cfg, &ids)
    }).collect();

    // ... cluster spawn (template) ...

    let leader_arc = std::sync::Arc::new(leader);   // generate now takes &self

    let model_path = fix.join("model.safetensors");
    let mut futures = Vec::new();
    for (i, prompt) in prompts.iter().enumerate() {
        let leader = leader_arc.clone();
        let cfg = cfg.clone();
        let model_path = model_path.clone();
        let ids: Vec<i32> = ai_engine_tokenizer::Tokenizer::encode(&tok, prompt).unwrap()
            .iter().map(|x| *x as i32).collect();
        futures.push(async move {
            let res = leader.generate::<burn_ndarray::NdArray>(
                &model_path, &cfg, 0..0, &ids, 5,
                ai_engine_runtime::sample::SamplingConfig {
                    temperature: 0.0, top_p: None, top_k: None, seed: 0,
                },
            ).await;
            (i, res)
        });
    }

    let results: Vec<(usize, anyhow::Result<Vec<u32>>)> =
        futures::future::join_all(futures).await;

    for (i, res) in results {
        let tokens = res.unwrap_or_else(|e| panic!("request {i} failed: {e}"));
        assert_eq!(
            tokens, baselines[i],
            "concurrent request {i} (prompt {:?}) did not match single-node baseline",
            prompts[i]
        );
    }
}
```

- [ ] **Step 2: Run + fix issues**

```bash
cargo test -p ai-engine-cluster --test concurrent_requests -- --nocapture
```

Likely first-time issues + fixes:
- **Hang**: `accept_uni` calls between concurrent requests interleave. The leader expects the next uni stream it accepts to be a reply for the request it just sent — but with concurrency, the workers' replies for different requests arrive in arbitrary order. **Solution**: the activation header has `request_id`; the leader's accept loop should match incoming streams to outstanding requests by header. For a minimal v0.2.1 fix: send + receive on each worker happen serialized within ONE request, but multiple requests can interleave between workers. As long as each request's `step_through_cluster_session` does its open-uni-then-accept-uni dance atomically (without yielding to another request), the streams remain paired. With tokio task scheduling, await points let other tasks run, so this assumption breaks. **Proper fix**: the leader buffers incoming activations into a `HashMap<Uuid, oneshot::Sender<Bytes>>` keyed by request_id; the worker echoes the request_id in the response header. The leader's step opens an open_uni, sends, then awaits on a `oneshot::Receiver` matched to its own request_id. A background dispatch task per worker connection reads incoming uni streams in a loop and routes them by request_id.

For v0.2.1 simplicity, an alternative: serialize requests using a per-worker `tokio::sync::Mutex<()>` that's held during the full `open_uni → accept_uni` exchange for one worker hop. This loses some concurrency (workers process requests one at a time) but maintains correctness. Concurrency between workers can still happen.

Actually the cleanest pragmatic fix: each request opens its own bidi stream pair for activations (not uni), so the response is on the same stream as the request. Then there's no demultiplexing problem. Change `tensor_io` flow to use bidi streams:

```rust
// In step_through_cluster_session, replace open_uni + accept_uni with open_bi:
let (mut send_bi, mut recv_bi) = conn.open_bi().await?;
// ... write header + payload ...
// ... read response header + payload from same stream ...
```

This is the right answer. quinn handles concurrent bidi streams from the same connection without interference. The protocol design already supports this (uni and bidi were used interchangeably in some places).

- [ ] **Step 3: Refactor `tensor_io` flow to use bidi per request hop**

Update `step_through_cluster_session` (Task 4) and the worker's accept loop in `run_worker_full` to use `open_bi()` / `accept_bi()` for activation exchanges instead of `open_uni()` / `accept_uni()`. The worker's loop changes from:

```rust
loop {
    let uni_in = conn.accept_uni().await?;
    // ... process ...
    let uni_out = conn.open_uni().await?;
    // ... reply ...
}
```

to:

```rust
loop {
    let (mut send, mut recv) = conn.accept_bi().await?;
    // ... process inbound + reply on same stream ...
}
```

This naturally handles concurrent requests because each request gets its own bidi stream.

- [ ] **Step 4: Verify**

```bash
cargo test -p ai-engine-cluster --test concurrent_requests -- --nocapture
cargo test -p ai-engine-cluster --test inprocess_cluster -- --nocapture     # existing tests still pass
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(cluster): bidi streams per activation exchange enable concurrent requests"
```

NO Co-Authored-By.

---

### Task 6: ClusterProvider::chat_stream implementation

**Files:**
- Modify: `crates/ai-engine-cluster/src/provider.rs`
- Modify: `crates/ai-engine-cluster/src/leader.rs` (add `generate_stream` that yields tokens)
- Create: `crates/ai-engine-cluster/tests/streaming.rs`

The leader's existing `generate` accumulates all tokens then returns the vec. For SSE we need to yield tokens as they're produced. Add `generate_stream` that returns `impl Stream<Item = Result<u32, anyhow::Error>>` and have `ClusterProvider::chat_stream` adapt that into the existing `EventStream<openai::ChatStreamEvent>` shape.

- [ ] **Step 1: Add `generate_stream` to ClusterLeader**

In `crates/ai-engine-cluster/src/leader.rs`:

```rust
use tokio::sync::mpsc;

impl ClusterLeader {
    /// Streaming variant of `generate`. Yields one token per chunk over an
    /// mpsc channel; returns immediately with a Receiver. The generation loop
    /// runs in a spawned tokio task.
    pub fn generate_stream<B>(
        self: std::sync::Arc<Self>,
        model_path: std::path::PathBuf,
        cfg: ai_engine_runtime::config::ModelConfig,
        leader_layers: std::ops::Range<usize>,
        prompt_ids: Vec<i32>,
        max_tokens: usize,
        sampling: ai_engine_runtime::sample::SamplingConfig,
    ) -> mpsc::Receiver<anyhow::Result<u32>>
    where
        B: burn::tensor::backend::Backend,
        B::Device: Default,
    {
        let (tx, rx) = mpsc::channel(64);
        tokio::spawn(async move {
            let mut session: crate::session::RequestSession<B> = match self.build_session(
                &model_path, &cfg, leader_layers,
            ).await {
                Ok(s) => s,
                Err(e) => { let _ = tx.send(Err(e)).await; return; }
            };

            // Prefill
            let last_logits = match step_through_cluster_session(&mut session, &prompt_ids, false).await {
                Ok(l) => l,
                Err(e) => { let _ = tx.send(Err(e)).await; return; }
            };
            session.current_pos = prompt_ids.len();
            let first = ai_engine_runtime::sample::sample(&last_logits, &sampling);
            if tx.send(Ok(first)).await.is_err() { return; }

            let mut last_token = first as i32;
            for _ in 1..max_tokens {
                let logits = match step_through_cluster_session(&mut session, &[last_token], false).await {
                    Ok(l) => l,
                    Err(e) => { let _ = tx.send(Err(e)).await; return; }
                };
                session.current_pos += 1;
                let t = ai_engine_runtime::sample::sample(&logits, &sampling);
                last_token = t as i32;
                if tx.send(Ok(t)).await.is_err() { return; }
            }
        });
        rx
    }
}
```

- [ ] **Step 2: Implement `ClusterProvider::chat_stream`**

In `crates/ai-engine-cluster/src/provider.rs`:

```rust
use futures::stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

#[async_trait::async_trait]
impl Provider for ClusterProvider {
    // ... existing chat impl ...

    async fn chat_stream(
        &self,
        req: openai::ChatRequest,
        _creds: &Credentials,
        ctx: &CallCtx,
    ) -> Result<EventStream<openai::ChatStreamEvent>, ProviderError> {
        if !self.is_leader { return Err(ProviderError::Unsupported); }
        let st = self.state.as_ref().ok_or_else(|| {
            ProviderError::InvalidResponse("no leader state".into())
        })?;

        let prompt_ids = ai_engine_tokenizer::Tokenizer::encode(&st.tokenizer, &render_prompt(&req))
            .map_err(|e| ProviderError::InvalidResponse(format!("tokenize: {e}")))?;
        let prompt_ids_i32: Vec<i32> = prompt_ids.iter().map(|x| *x as i32).collect();
        let max_tokens = req.max_tokens.unwrap_or(256) as usize;
        let sampling = ai_engine_runtime::sample::SamplingConfig {
            temperature: req.temperature.unwrap_or(1.0),
            top_p: None, top_k: None,
            seed: ctx.request_id.as_u128() as u64,
        };

        // Get the leader as Arc<ClusterLeader> — st.leader is owned by Arc<LeaderState>,
        // so we wrap it for the generate_stream call.
        // generate_stream requires Arc<Self>; we share a fresh Arc that points
        // at the same underlying ClusterLeader through LeaderState.
        let leader_arc = std::sync::Arc::new(/* see below */);

        // Wait — `st.leader` is `ClusterLeader`, not `Arc<ClusterLeader>`.
        // We need to refactor LeaderState to hold `Arc<ClusterLeader>` so we can
        // share it into the spawned generation task. Update Task 4's LeaderState
        // accordingly (this is consistent with the no-Mutex direction).

        // For Plan 4 simplicity, change `pub leader: ClusterLeader` -> `pub leader: Arc<ClusterLeader>`
        // in LeaderState. Then:
        let mut rx = st.leader.clone().generate_stream::<burn_ndarray::NdArray>(
            st.model_path.clone(),
            st.model_cfg.clone(),
            st.leader_layers.clone(),
            prompt_ids_i32,
            max_tokens,
            sampling,
        );

        let model_id = req.model.clone();
        let request_id_str = format!("chatcmpl-{}", ctx.request_id);
        let tokenizer_arc = st.tokenizer.clone();  // HfTokenizer must be Clone; if not, wrap in Arc<HfTokenizer> in LeaderState.

        // Adapt mpsc::Receiver<Result<u32>> into a Stream<Item = Result<ChatStreamEvent, ProviderError>>.
        let stream = async_stream::stream! {
            while let Some(result) = rx.recv().await {
                match result {
                    Ok(token) => {
                        let text = match ai_engine_tokenizer::Tokenizer::decode(&tokenizer_arc, &[token]) {
                            Ok(t) => t,
                            Err(e) => {
                                yield Err(ProviderError::InvalidResponse(format!("decode: {e}")));
                                return;
                            }
                        };
                        // Build a minimal OpenAI-style streaming chunk.
                        let raw = serde_json::json!({
                            "id": request_id_str,
                            "object": "chat.completion.chunk",
                            "created": 0,
                            "model": model_id,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": text},
                                "finish_reason": null,
                            }],
                        });
                        yield Ok(openai::ChatStreamEvent { raw });
                    }
                    Err(e) => {
                        yield Err(ProviderError::InvalidResponse(format!("generate: {e}")));
                        return;
                    }
                }
            }
            // Final chunk with finish_reason.
            let final_raw = serde_json::json!({
                "id": request_id_str,
                "object": "chat.completion.chunk",
                "created": 0,
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }],
            });
            yield Ok(openai::ChatStreamEvent { raw: final_raw });
        };

        Ok(Box::pin(stream))
    }
}
```

If `HfTokenizer` isn't `Clone`, either:
- Add `#[derive(Clone)]` to `HfTokenizer` in `ai-engine-tokenizer/src/hf.rs` (its inner `tokenizers::Tokenizer` is Clone — check). Tokenizers in HF's Rust impl are Clone via Arc internally.
- Or wrap `tokenizer: Arc<HfTokenizer>` in `LeaderState`.

The cleanest approach: change `LeaderState::tokenizer` to `Arc<HfTokenizer>`. Update construction in `build_app_state` accordingly. (`Arc::new(tokenizer)` once on load.)

Add `async-stream.workspace = true` and `tokio-stream` if not in ai-engine-cluster Cargo.toml. (`async-stream` is already in workspace deps from Plan 1.)

- [ ] **Step 3: Streaming test**

`crates/ai-engine-cluster/tests/streaming.rs`:

```rust
//! ClusterProvider::chat_stream returns an SSE-compatible token stream.

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn chat_stream_yields_tokens_then_finish() {
    use futures::StreamExt;
    use ai_engine_provider::provider::Provider;
    // ... spin up 3-node cluster, build LeaderState, ClusterProvider ...

    let req = ai_engine_provider::openai::ChatRequest {
        model: "toy".into(),
        messages: vec![ai_engine_provider::openai::ChatMessage {
            role: "user".into(),
            content: ai_engine_provider::openai::ChatContent::Text("hi".into()),
            extras: Default::default(),
        }],
        stream: Some(true),
        temperature: Some(0.0),
        max_tokens: Some(3),
        stream_options: None,
        extras: Default::default(),
    };
    let ctx = ai_engine_provider::provider::CallCtx {
        request_id: uuid::Uuid::now_v7(),
        deadline: None,
        upstream_model: "toy".into(),
    };
    let creds = ai_engine_provider::provider::Credentials::none();
    let mut stream = provider.chat_stream(req, &creds, &ctx).await.unwrap();

    let mut chunks = Vec::new();
    while let Some(item) = stream.next().await {
        chunks.push(item.unwrap());
    }
    // 3 content chunks + 1 finish chunk = 4 minimum (could be more if model emits short tokens).
    assert!(chunks.len() >= 4, "expected at least 4 chunks (3 content + 1 finish), got {}", chunks.len());
    // Last chunk should have finish_reason = "stop".
    let last = chunks.last().unwrap();
    assert_eq!(last.raw["choices"][0]["finish_reason"], "stop");
    // First content chunk should have a content field.
    assert!(chunks[0].raw["choices"][0]["delta"]["content"].is_string());
}
```

- [ ] **Step 4: Verify + commit**

```bash
cargo test -p ai-engine-cluster --test streaming
cargo clippy --workspace --all-targets -- -D warnings
git add -A
git commit -m "feat(cluster): chat_stream — per-token SSE streaming via ClusterLeader::generate_stream"
```

NO Co-Authored-By.

---

### Task 7: Multi-process SSE smoke test

**Files:**
- Create: `crates/ai-engine/tests/streaming_smoke.rs`

Same pattern as the existing `multiproc_smoke.rs` but sends a `stream: true` request and verifies a sequence of SSE events.

- [ ] **Step 1: Test (copy multiproc_smoke.rs template, modify the request + response handling)**

```rust
#[test]
#[ignore = "multi-process SSE smoke; requires release build; run with --ignored"]
fn three_process_cluster_serves_streaming_chat() {
    // ... copy 3-process spawn template from multiproc_smoke.rs ...
    // Then send:
    let response = client.post(format!("{leader_url}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "toy-llama",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 3,
            "stream": true,
        }))
        .send().unwrap();
    assert_eq!(response.status().as_u16(), 200);
    assert_eq!(
        response.headers().get("content-type").map(|v| v.to_str().unwrap_or("")),
        Some("text/event-stream"),
        "response must be SSE"
    );

    // Read body as text and parse data: lines.
    let body = response.text().unwrap();
    let data_lines: Vec<&str> = body.lines()
        .filter(|l| l.starts_with("data: "))
        .collect();
    assert!(
        data_lines.len() >= 4,
        "expected at least 4 SSE data lines (3 chunks + [DONE]), got {}",
        data_lines.len()
    );
    // Last data line should be "[DONE]".
    assert_eq!(
        *data_lines.last().unwrap(), "data: [DONE]",
        "SSE stream must terminate with [DONE]"
    );
    // ... cleanup processes ...
}
```

- [ ] **Step 2: Verify + commit**

```bash
cargo build --release
cargo test -p ai-engine --test streaming_smoke -- --ignored --nocapture
git add -A
git commit -m "test(smoke): 3-process cluster serves a streaming chat completion (SSE)"
```

NO Co-Authored-By.

---

### Task 8: README + tag v0.2.1

**Files:**
- Modify: `README.md`
- Tag: `v0.2.1`

- [ ] **Step 1: Update README**

Add a brief section under v0.2.0's "Known limitations" pointing out which are fixed:

```markdown
### v0.2.1 — Streaming + concurrency

ai-engine v0.2.1 closes three v0.2.0 gaps:

- **Per-token SSE streaming** on `/v1/chat/completions` when `stream: true`.
- **Concurrent requests on one leader.** Multiple in-flight chat completions
  are interleaved through the cluster — no more serialization on `&mut self`.
- **Real partition Assignment** over QUIC. Workers wait for the leader's
  manifest before loading weights; partition policy lives entirely on the
  leader, including the optional `partition_override` in TOML.

Updated known limitations (still deferred to later releases): mDNS auto-
discovery, dynamic worker membership, automatic failover, quantization,
tensor parallelism, web playground UI.
```

- [ ] **Step 2: Verify everything**

```bash
cd /home/alessio/aip/airproxy
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo build --workspace --release
cargo test -p ai-engine --test multiproc_smoke -- --ignored
cargo test -p ai-engine --test streaming_smoke -- --ignored
```

- [ ] **Step 3: Commit + tag**

```bash
git add README.md
git commit -m "docs: announce v0.2.1 streaming + concurrency release"
git tag v0.2.1
git log --oneline -5
git tag
```

NO Co-Authored-By.

---

## Self-review

**Spec coverage:**
- v0.2.0 limitation "Single concurrent request per cluster" → Tasks 4, 5
- v0.2.0 limitation "Non-streaming responses only" → Tasks 6, 7
- v0.2.0 limitation "Workers compute their own layer range via even-split" → Tasks 1, 2, 3

**Placeholder scan:**
- Task 5 Step 1 references "common helpers from inprocess_cluster.rs OR duplicate" — the implementer either factors a shared fixture helper or copy-pastes; the contract is clear. Acceptable as a one-line direction.
- Task 7 has "// ... copy 3-process spawn template from multiproc_smoke.rs ..." which the implementer fills in from the existing file. Acceptable since the template is checked-in code.
- No "TBD" / "fill in later" / "add error handling".

**Type consistency:**
- `RequestSession<B>` (Task 4) → consumed by `step_through_cluster_session` (Task 4) and `generate_stream` (Task 6). Fields consistent. ✓
- `LeaderModel<B>` (Task 4) → wrapped in `Arc` and held by `RequestSession`. ✓
- `LeaderConfig::partition_override` (Task 2) → consumed by `ClusterLeader::start` for `manual_partition` dispatch. ✓
- `PartitionManifest::for_node` (Task 1) → called in Task 2's worker code and Task 2's leader test. ✓
- `LeaderState::leader: Arc<ClusterLeader>` (Task 6) → required by `generate_stream(self: Arc<Self>, ...)`. Update `build_app_state` accordingly. ✓
- `LeaderState::tokenizer: Arc<HfTokenizer>` (Task 6) → required for clone-into-spawn pattern. ✓

**Acknowledged risks:**

1. **bidi-stream refactor (Task 5)** is the heaviest change. Worker's serving loop changes from `accept_uni() → process → open_uni()` to `accept_bi() → process → reply on same stream`. Both work; bidi is cleaner for concurrent requests but the existing inprocess_cluster tests assume uni. They'll need to keep working — the refactor preserves the per-hop semantics, just changes the stream type.
2. **`LeaderState` ownership refactor (Tasks 4, 6)** ripples through `build_app_state`. Expect 2-3 spots that need updating to pass `Arc<...>` instead of `Arc<Mutex<...>>`.
3. **`HfTokenizer` Clone-ability** — the underlying `tokenizers::Tokenizer` in HF's crate IS `Clone` (Arc-backed). If our wrapper isn't, add `#[derive(Clone)]`.

---

## Execution Handoff

Plan 4 saved to `docs/superpowers/plans/2026-05-23-plan-4-v021-streaming-concurrency.md`.

Two execution options:

**1. Subagent-Driven (recommended)** — 8 tasks, all bounded. Tasks 1, 3, 7, 8 are small. Tasks 2, 4, 5, 6 are substantive but well-specified.

**2. Inline Execution** — also reasonable for this size.

After v0.2.1 ships, sub-project #4 (mDNS, dynamic membership, web playground) is the natural next step.
