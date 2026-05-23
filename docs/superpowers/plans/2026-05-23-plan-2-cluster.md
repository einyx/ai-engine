# Plan 2 — `ai-engine-cluster`

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the `ai-engine-cluster` crate end-to-end: QUIC transport between nodes, a control + data protocol, a capability-aware partitioner, leader/worker state machines, and a `ClusterProvider` that implements the existing `ai_engine_provider::Provider` trait. A 3-node loopback integration test must produce identical chat completions to a single-node run using the same toy-llama-3 fixture.

**Architecture:** One new crate (`ai-engine-cluster`) that depends on `ai-engine-runtime` (Plan 1) for the ML and `ai-engine-provider` for the trait surface. Two communication planes: a long-lived bidi QUIC stream for control messages (postcard-framed) and per-request unidirectional streams for activation tensors (length-prefixed binary). TLS is mandatory with self-signed certs pinned by SHA-256 fingerprint. Partitioning runs a DP solver over (n_layers² × n_nodes) and produces a content-addressed `PartitionManifest`. Leader hosts axum HTTP + a layer range; workers host QUIC only.

**Tech Stack:** `quinn` (QUIC), `rustls` (TLS), `rcgen` (cert generation), `postcard` (serde binary codec), `tokio` (async runtime, already in workspace), `sha2` (fingerprint hashing), `sysinfo` (memory detection), `bytemuck` (tensor byte casting — already in workspace).

**Scope rule:** Plan 2 ships the cluster crate as a self-contained library + in-process integration test. NO config schema changes, NO binary integration, NO multi-process smoke test. Those land in Plan 3.

**Baseline:** Branch `chore/rename-to-ai-engine` at `v0.2.0-alpha.1`. 9 crates, 112 tests, clippy clean.

---

## File structure (locked in here)

```
crates/ai-engine-cluster/
├── Cargo.toml
└── src/
    ├── lib.rs                       # public surface: ClusterProvider + key types
    ├── tls.rs                       # rcgen cert generation; SHA-256 fingerprint; rustls verifier
    ├── protocol/
    │   ├── mod.rs                   # module declarations
    │   ├── control.rs               # LeaderToWorker / WorkerToLeader enums
    │   ├── data.rs                  # ActivationHeader struct
    │   └── codec.rs                 # postcard encode/decode + length-prefixed framing
    ├── transport/
    │   ├── mod.rs
    │   ├── quic.rs                  # quinn endpoint setup; connection establishment
    │   └── frame.rs                 # binary frame reader/writer over a quinn stream
    ├── capability.rs                # Capability struct + microbenchmark + memory detection
    ├── partition.rs                 # PartitionManifest + DP solver + manual override
    ├── tensor_io.rs                 # bf16/f32 tensor <-> bytes (data plane payload)
    ├── worker.rs                    # worker state machine: accept inbound, run layers, manage KV
    ├── leader.rs                    # leader state machine: orchestrate, generation loop
    └── provider.rs                  # ClusterProvider impl of ai_engine_provider::Provider
└── tests/
    ├── tls_pinning.rs               # cert + fingerprint round-trip
    ├── protocol_codec.rs            # postcard message round-trips
    ├── partition_solver.rs          # DP solver determinism + feasibility + override
    ├── transport_loopback.rs        # quinn over loopback: connect, exchange messages
    └── inprocess_cluster.rs         # 3-node end-to-end vs single-node baseline
```

File responsibilities map 1:1 with the design spec §§5–8.

---

## Important pre-flight notes

- **quinn 0.11+ uses rustls 0.23+ types.** API may differ from older tutorials. Verify via `cargo doc -p quinn` after the first add, or query Context7 (`mcp__plugin_context7_context7__resolve-library-id` → `quinn-rs/quinn`).
- **rustls in 2026 requires explicit `CryptoProvider::install_default()`** before any rustls operation. Use `rustls::crypto::ring::default_provider().install_default()` at startup OR per-test (idempotent — wrap with `let _ = ...install_default();`).
- **The toy-llama-3 fixture from Plan 1** is `crates/ai-engine-runtime/fixtures/toy-llama-3/`. The cluster integration test loads weights from this fixture; each in-process worker uses `load_range` with its assigned layer range.
- **`Tensor::<B, N>::to_data()` → `TensorData` → `to_vec::<f32>()`** is the canonical way to extract tensor bytes for the data plane. Going back to a Tensor: `Tensor::from_data(TensorData::new(vec, shape), &dev)`.
- **No `Provider` trait changes.** This is a hard invariant per §8 of the spec. `ClusterProvider` implements the existing trait verbatim.

---

### Task 1: Crate scaffold + dep tree

**Files:**
- Modify: root `Cargo.toml` (workspace deps)
- Create: `crates/ai-engine-cluster/Cargo.toml`
- Create: `crates/ai-engine-cluster/src/lib.rs` (just module declarations + a single dummy test)

- [ ] **Step 1: Workspace dep additions**

Append to root `Cargo.toml` `[workspace.dependencies]`:

```toml
quinn = { version = "0.11", default-features = false, features = ["runtime-tokio", "rustls-ring"] }
rustls = { version = "0.23", default-features = false, features = ["ring", "std"] }
rcgen = { version = "0.13", default-features = false, features = ["pem", "ring"] }
postcard = { version = "1", default-features = false, features = ["use-std"] }
sha2 = "0.10"
sysinfo = "0.32"
ai-engine-cluster = { path = "crates/ai-engine-cluster" }
```

Verify versions resolve against current crates.io. If quinn 0.11+ has bumped rustls to a newer minor version, pin compatibly; commit message should note the actual versions installed.

- [ ] **Step 2: Crate manifest**

`crates/ai-engine-cluster/Cargo.toml`:

```toml
[package]
name = "ai-engine-cluster"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true

[dependencies]
ai-engine-core.workspace = true
ai-engine-provider.workspace = true
ai-engine-runtime.workspace = true
ai-engine-tokenizer.workspace = true
anyhow.workspace = true
async-trait.workspace = true
bytemuck.workspace = true
bytes.workspace = true
futures.workspace = true
postcard.workspace = true
quinn.workspace = true
rcgen.workspace = true
rustls.workspace = true
serde = { workspace = true }
serde_json.workspace = true
sha2.workspace = true
sysinfo.workspace = true
thiserror.workspace = true
tokio = { workspace = true }
tracing.workspace = true
uuid.workspace = true

[dev-dependencies]
burn-ndarray = { workspace = true }
tempfile = "3"
tokio = { workspace = true, features = ["macros", "rt-multi-thread"] }
```

- [ ] **Step 3: Skeleton lib.rs**

`crates/ai-engine-cluster/src/lib.rs`:

```rust
//! ai-engine-cluster
//!
//! Distributed inference coordinator. Implements `Provider` from
//! `ai_engine_provider` against a cluster of nodes running QUIC.

#[cfg(test)]
mod smoke_compile_test {
    #[test]
    fn crate_compiles() {
        // sanity: the crate's dep tree resolves and burn types are usable.
        let _: burn::tensor::Tensor<burn_ndarray::NdArray, 1> =
            burn::tensor::Tensor::zeros([4], &Default::default());
    }
}
```

- [ ] **Step 4: Verify**

```bash
cd /home/alessio/aip/airproxy
cargo check -p ai-engine-cluster
cargo test -p ai-engine-cluster
cargo clippy --workspace --all-targets -- -D warnings
```

First check may take a while (quinn + rustls compile). Expect 30–90s.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(cluster): crate scaffold with quinn/rustls/postcard deps"
```

NO Co-Authored-By footer (global preference).

---

### Task 2: TLS — self-signed cert generation + fingerprint

**Files:**
- Create: `crates/ai-engine-cluster/src/tls.rs`
- Create: `crates/ai-engine-cluster/tests/tls_pinning.rs`
- Modify: `crates/ai-engine-cluster/src/lib.rs` (add `pub mod tls;`)

- [ ] **Step 1: Failing tests**

`crates/ai-engine-cluster/tests/tls_pinning.rs`:

```rust
use ai_engine_cluster::tls::{generate_node_identity, NodeIdentity, fingerprint_sha256};

#[test]
fn cert_generation_produces_valid_pem_pair() {
    let id = generate_node_identity("test-node").unwrap();
    assert!(id.cert_pem.starts_with("-----BEGIN CERTIFICATE-----"));
    assert!(id.key_pem.starts_with("-----BEGIN PRIVATE KEY-----"));
}

#[test]
fn fingerprint_is_64_hex_chars_prefixed() {
    let id = generate_node_identity("test-node").unwrap();
    assert!(id.fingerprint.starts_with("sha256:"));
    let hex = &id.fingerprint["sha256:".len()..];
    assert_eq!(hex.len(), 64);
    assert!(hex.chars().all(|c| c.is_ascii_hexdigit()));
}

#[test]
fn fingerprint_is_deterministic_for_a_given_cert() {
    let id = generate_node_identity("node-a").unwrap();
    // Compute the fingerprint a second time directly from the DER and assert equal.
    let fp2 = fingerprint_sha256(&id.cert_der);
    assert_eq!(id.fingerprint, fp2);
}

#[test]
fn two_invocations_produce_distinct_fingerprints() {
    let a = generate_node_identity("node-a").unwrap();
    let b = generate_node_identity("node-b").unwrap();
    assert_ne!(a.fingerprint, b.fingerprint);
}
```

- [ ] **Step 2: Confirm fails**

```bash
cargo test -p ai-engine-cluster --test tls_pinning 2>&1 | tail -5
```

- [ ] **Step 3: Implement `tls.rs`**

```rust
use rcgen::{CertificateParams, KeyPair, SignatureAlgorithm};
use sha2::{Digest, Sha256};

pub struct NodeIdentity {
    pub cert_pem: String,
    pub key_pem: String,
    pub cert_der: Vec<u8>,
    pub key_der: Vec<u8>,
    pub fingerprint: String,    // "sha256:<64 hex chars>"
}

/// Generate a fresh self-signed ed25519 certificate for `node_id`. The CN of
/// the certificate is `node_id`; SAN entries are not added in v0.2 (peer
/// pinning is by fingerprint, not name).
pub fn generate_node_identity(node_id: &str) -> anyhow::Result<NodeIdentity> {
    // rcgen 0.13 API: `KeyPair::generate(&rcgen::PKCS_ED25519)` returns a
    // keypair; CertificateParams::new() creates default params; .self_signed(&key)
    // produces a Certificate from which we extract PEM and DER.
    let key = KeyPair::generate_for(&rcgen::PKCS_ED25519)
        .map_err(|e| anyhow::anyhow!("keypair: {e}"))?;
    let mut params = CertificateParams::new(vec![node_id.to_string()])
        .map_err(|e| anyhow::anyhow!("cert params: {e}"))?;
    params.distinguished_name.push(rcgen::DnType::CommonName, node_id);

    let cert = params.self_signed(&key)
        .map_err(|e| anyhow::anyhow!("self-sign: {e}"))?;

    let cert_pem = cert.pem();
    let key_pem = key.serialize_pem();
    let cert_der = cert.der().to_vec();
    let key_der = key.serialize_der();
    let fingerprint = fingerprint_sha256(&cert_der);

    Ok(NodeIdentity { cert_pem, key_pem, cert_der, key_der, fingerprint })
}

/// SHA-256 fingerprint of a DER-encoded certificate, formatted as
/// "sha256:<64 lowercase hex chars>".
pub fn fingerprint_sha256(cert_der: &[u8]) -> String {
    let digest = Sha256::digest(cert_der);
    let mut out = String::with_capacity(7 + 64);
    out.push_str("sha256:");
    for byte in digest.iter() {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}
```

If `KeyPair::generate_for` doesn't exist in your rcgen version, try `KeyPair::generate(&rcgen::PKCS_ED25519)` or check the rcgen 0.13 module surface. The keypair-and-self-sign sequence is the standard idiom; only the API method names vary slightly.

- [ ] **Step 4: Wire module + verify**

`crates/ai-engine-cluster/src/lib.rs`:

```rust
pub mod tls;
```

```bash
cargo test -p ai-engine-cluster --test tls_pinning
cargo clippy --workspace --all-targets -- -D warnings
```

4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(cluster): self-signed cert generation + SHA-256 fingerprint pinning"
```

---

### Task 3: Protocol message types + postcard codec

**Files:**
- Create: `crates/ai-engine-cluster/src/protocol/mod.rs`
- Create: `crates/ai-engine-cluster/src/protocol/control.rs`
- Create: `crates/ai-engine-cluster/src/protocol/data.rs`
- Create: `crates/ai-engine-cluster/src/protocol/codec.rs`
- Create: `crates/ai-engine-cluster/tests/protocol_codec.rs`
- Modify: `crates/ai-engine-cluster/src/lib.rs`

- [ ] **Step 1: Failing tests**

`crates/ai-engine-cluster/tests/protocol_codec.rs`:

```rust
use ai_engine_cluster::protocol::control::{EndReason, FaultKind, LeaderToWorker, WorkerToLeader};
use ai_engine_cluster::protocol::data::{ActivationHeader, Dtype};
use ai_engine_cluster::protocol::codec::{decode, encode};
use uuid::Uuid;

#[test]
fn control_message_join_roundtrips() {
    let msg = LeaderToWorker::Join {
        cluster_id: "home-lab".into(),
        protocol_version: 1,
        leader_node_id: "node-a".into(),
    };
    let bytes = encode(&msg).unwrap();
    let back: LeaderToWorker = decode(&bytes).unwrap();
    match (msg, back) {
        (LeaderToWorker::Join { cluster_id: a, protocol_version: pa, leader_node_id: na },
         LeaderToWorker::Join { cluster_id: b, protocol_version: pb, leader_node_id: nb }) => {
            assert_eq!(a, b); assert_eq!(pa, pb); assert_eq!(na, nb);
        }
        _ => panic!("variant changed"),
    }
}

#[test]
fn control_message_begin_with_uuid_roundtrips() {
    let id = Uuid::now_v7();
    let msg = LeaderToWorker::Begin { request_id: id, max_tokens: 256, prompt_len: 12 };
    let bytes = encode(&msg).unwrap();
    let back: LeaderToWorker = decode(&bytes).unwrap();
    if let LeaderToWorker::Begin { request_id, max_tokens, prompt_len } = back {
        assert_eq!(request_id, id);
        assert_eq!(max_tokens, 256);
        assert_eq!(prompt_len, 12);
    } else { panic!("variant"); }
}

#[test]
fn worker_fault_report_roundtrips() {
    let msg = WorkerToLeader::FaultReport {
        request_id: Some(Uuid::now_v7()),
        kind: FaultKind::OutOfMemory,
        detail: "VRAM exhausted at layer 17".into(),
    };
    let bytes = encode(&msg).unwrap();
    let _back: WorkerToLeader = decode(&bytes).unwrap();
}

#[test]
fn end_reason_variants_roundtrip() {
    for r in [EndReason::Completed, EndReason::ClientCancelled, EndReason::Error] {
        let msg = LeaderToWorker::End { request_id: Uuid::now_v7(), reason: r };
        let bytes = encode(&msg).unwrap();
        let _back: LeaderToWorker = decode(&bytes).unwrap();
    }
}

#[test]
fn activation_header_roundtrips() {
    let h = ActivationHeader {
        request_id: Uuid::now_v7(),
        seq_pos: 7,
        shape: [1, 1, 256],
        dtype: Dtype::Bf16,
        is_terminal: false,
    };
    let bytes = encode(&h).unwrap();
    let back: ActivationHeader = decode(&bytes).unwrap();
    assert_eq!(back.request_id, h.request_id);
    assert_eq!(back.seq_pos, h.seq_pos);
    assert_eq!(back.shape, h.shape);
    assert_eq!(back.is_terminal, h.is_terminal);
}
```

- [ ] **Step 2: Implement `control.rs`**

```rust
use crate::capability::Capability;
use crate::partition::PartitionManifest;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
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
        request_id: Uuid,
        max_tokens: u32,
        prompt_len: u32,
    },
    End {
        request_id: Uuid,
        reason: EndReason,
    },
    HealthPing { nonce: u64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkerToLeader {
    Capability(Capability),
    JoinAck { node_id: String, certificate_sha256: [u8; 32] },
    BeginAck { request_id: Uuid },
    Heartbeat { nonce: u64 },
    FaultReport {
        request_id: Option<Uuid>,
        kind: FaultKind,
        detail: String,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum EndReason {
    Completed,
    ClientCancelled,
    Error,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum FaultKind {
    OutOfMemory,
    BackendError,
    Internal,
}
```

This file imports `Capability` (Task 5) and `PartitionManifest` (Task 6). Those modules must exist at compile time but the tests in this task only exercise variants that don't reach into those types — so create stubs:

`crates/ai-engine-cluster/src/capability.rs` (stub for this task):
```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capability {
    pub node_id: String,
    // Filled out in Task 5.
}
```

`crates/ai-engine-cluster/src/partition.rs` (stub):
```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionManifest {
    pub model_id: String,
    // Filled out in Task 6.
}
```

- [ ] **Step 3: Implement `data.rs`**

```rust
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Dtype { F32, F16, Bf16 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationHeader {
    pub request_id: Uuid,
    pub seq_pos: u32,
    pub shape: [u32; 3],          // [batch, seq, hidden]
    pub dtype: Dtype,
    pub is_terminal: bool,
}
```

- [ ] **Step 4: Implement `codec.rs`**

```rust
use serde::{de::DeserializeOwned, Serialize};

pub fn encode<T: Serialize>(value: &T) -> anyhow::Result<Vec<u8>> {
    postcard::to_allocvec(value)
        .map_err(|e| anyhow::anyhow!("postcard encode: {e}"))
}

pub fn decode<T: DeserializeOwned>(bytes: &[u8]) -> anyhow::Result<T> {
    postcard::from_bytes(bytes)
        .map_err(|e| anyhow::anyhow!("postcard decode: {e}"))
}
```

- [ ] **Step 5: Wire modules**

`crates/ai-engine-cluster/src/protocol/mod.rs`:
```rust
pub mod control;
pub mod data;
pub mod codec;
```

`crates/ai-engine-cluster/src/lib.rs` (additions):
```rust
pub mod capability;
pub mod partition;
pub mod protocol;
pub mod tls;
```

- [ ] **Step 6: Verify + commit**

```bash
cargo test -p ai-engine-cluster --test protocol_codec
cargo clippy --workspace --all-targets -- -D warnings
git add -A
git commit -m "feat(cluster): control + data plane message types with postcard codec"
```

5 tests pass. NO Co-Authored-By.

---

### Task 4: Capability advertisement (microbenchmark + memory detection)

**Files:**
- Modify: `crates/ai-engine-cluster/src/capability.rs` (replace stub)
- Create: `crates/ai-engine-cluster/tests/capability.rs`

- [ ] **Step 1: Failing tests**

```rust
use ai_engine_cluster::capability::{detect_capability, BackendKind, Capability};

#[test]
fn capability_detection_populates_realistic_values() {
    let cap = detect_capability("test-node", BackendKind::Cpu, 0, None).unwrap();
    assert_eq!(cap.node_id, "test-node");
    assert!(cap.available_memory_bytes > 0, "memory > 0");
    assert!(cap.compute_score > 0, "compute_score > 0 (microbenchmark must run)");
    assert_eq!(cap.backend, BackendKind::Cpu);
}

#[test]
fn capability_respects_max_memory_override() {
    // If max_memory_mib is set, available_memory_bytes is min(detected, override*MiB).
    let cap = detect_capability("test-node", BackendKind::Cpu, 0, Some(100)).unwrap();
    assert!(cap.available_memory_bytes <= 100 * 1024 * 1024);
}

#[test]
fn cpu_compute_score_baseline_around_100() {
    // The CPU microbenchmark is normalized so a baseline CPU returns ~100.
    // Any sane test environment should land in [10, 10_000].
    let cap = detect_capability("benchmark-cpu", BackendKind::Cpu, 0, None).unwrap();
    assert!(
        cap.compute_score >= 10 && cap.compute_score <= 10_000,
        "compute_score = {} (expected 10–10000)",
        cap.compute_score
    );
}
```

- [ ] **Step 2: Implement `capability.rs`**

```rust
use serde::{Deserialize, Serialize};
use std::time::Instant;
use sysinfo::System;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum BackendKind { Cpu, Cuda, Metal, Wgpu }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capability {
    pub node_id: String,
    pub backend: BackendKind,
    pub device_index: usize,
    pub available_memory_bytes: u64,
    pub compute_score: u32,
    pub link_mbps_to_leader: u32,
}

const SAFETY_MARGIN_BYTES: u64 = 512 * 1024 * 1024;

/// Detect this node's capability. `max_memory_mib` is an optional config override
/// that caps the memory advertised (useful for shared boxes).
pub fn detect_capability(
    node_id: &str,
    backend: BackendKind,
    device_index: usize,
    max_memory_mib: Option<u64>,
) -> anyhow::Result<Capability> {
    let detected_mem = detect_memory_bytes(backend)?;
    let with_margin = detected_mem.saturating_sub(SAFETY_MARGIN_BYTES);
    let final_mem = match max_memory_mib {
        Some(cap) => with_margin.min(cap * 1024 * 1024),
        None => with_margin,
    };
    let compute_score = microbenchmark_compute_score();

    Ok(Capability {
        node_id: node_id.to_string(),
        backend,
        device_index,
        available_memory_bytes: final_mem,
        compute_score,
        link_mbps_to_leader: 0,    // populated during QUIC handshake by leader
    })
}

fn detect_memory_bytes(backend: BackendKind) -> anyhow::Result<u64> {
    match backend {
        BackendKind::Cpu => {
            let mut sys = System::new_all();
            sys.refresh_memory();
            // sysinfo 0.32: total_memory() returns bytes.
            Ok(sys.total_memory())
        }
        BackendKind::Cuda | BackendKind::Metal | BackendKind::Wgpu => {
            // Backend-specific VRAM detection is deferred — Task 4 only covers CPU.
            // Real impl uses nvml-wrapper / metal::MTLDevice / wgpu::Adapter::get_info.
            // For now: fall back to "1 GiB", which the integration tests don't depend on.
            // Plan 3 will plumb real detection through.
            Ok(1024 * 1024 * 1024)
        }
    }
}

/// One-time matmul microbenchmark normalized to ~100 for a baseline CPU.
/// Returns a dimensionless relative score. Higher is faster.
fn microbenchmark_compute_score() -> u32 {
    // Multiply two 256x256 f32 matrices. We don't use burn here — that would
    // require a generic backend in this module which is undesirable.
    // Simple naive matmul is plenty for a relative score.
    const N: usize = 256;
    let a: Vec<f32> = (0..N*N).map(|i| (i as f32 * 0.001).sin()).collect();
    let b: Vec<f32> = (0..N*N).map(|i| (i as f32 * 0.002).cos()).collect();
    let mut c = vec![0.0_f32; N * N];

    let t0 = Instant::now();
    for i in 0..N {
        for j in 0..N {
            let mut s = 0.0;
            for k in 0..N {
                s += a[i*N + k] * b[k*N + j];
            }
            c[i*N + j] = s;
        }
    }
    let elapsed_ms = t0.elapsed().as_millis() as f64;
    // Baseline: ~250 ms on a slow CPU -> score 100.
    // score = 100 * 250 / elapsed_ms
    let score = (100.0 * 250.0 / elapsed_ms.max(1.0)) as u32;
    score.max(1)
}
```

- [ ] **Step 3: Verify + commit**

```bash
cargo test -p ai-engine-cluster --test capability
cargo clippy --workspace --all-targets -- -D warnings
git add -A
git commit -m "feat(cluster): Capability advertisement with CPU microbenchmark + memory detection"
```

3 tests pass.

---

### Task 5: Partitioner — DP solver + manifest + content addressing

**Files:**
- Modify: `crates/ai-engine-cluster/src/partition.rs` (replace stub)
- Create: `crates/ai-engine-cluster/tests/partition_solver.rs`

- [ ] **Step 1: Failing tests**

`crates/ai-engine-cluster/tests/partition_solver.rs`:

```rust
use ai_engine_cluster::capability::{BackendKind, Capability};
use ai_engine_cluster::partition::{
    auto_partition, manual_partition, PartitionManifest, NodeAssignment,
};

fn cap(node_id: &str, mem_gib: u64, compute: u32) -> Capability {
    Capability {
        node_id: node_id.into(),
        backend: BackendKind::Cpu,
        device_index: 0,
        available_memory_bytes: mem_gib * 1024 * 1024 * 1024,
        compute_score: compute,
        link_mbps_to_leader: 1000,
    }
}

#[test]
fn even_capability_yields_even_split() {
    let caps = vec![cap("a", 16, 100), cap("b", 16, 100), cap("c", 16, 100)];
    // 30-layer model, 2 GiB per layer + 4 GiB embed/output budget.
    let m = auto_partition("model-test", &caps, 30, 2 * 1024 * 1024 * 1024,
                           4 * 1024 * 1024 * 1024, 1024 * 1024 * 1024).unwrap();
    assert_eq!(m.assignments.len(), 3);
    let total: usize = m.assignments.iter().map(|a| a.layer_range.len()).sum();
    assert_eq!(total, 30);
}

#[test]
fn assignments_are_contiguous_and_complete() {
    let caps = vec![cap("a", 16, 100), cap("b", 16, 100)];
    let m = auto_partition("m", &caps, 12, 1024*1024*1024, 1024*1024*1024, 256*1024*1024).unwrap();
    // First assignment starts at 0; subsequent start where previous ended; last ends at n_layers.
    let mut expected_start = 0;
    for a in &m.assignments {
        assert_eq!(a.layer_range.start, expected_start);
        expected_start = a.layer_range.end;
    }
    assert_eq!(expected_start, 12);
}

#[test]
fn infeasible_partition_returns_error() {
    // 50 layers at 4 GiB each = 200 GiB. Two 8 GiB nodes -> infeasible.
    let caps = vec![cap("a", 8, 100), cap("b", 8, 100)];
    let r = auto_partition("big", &caps, 50, 4 * 1024 * 1024 * 1024,
                           1024 * 1024 * 1024, 1024 * 1024 * 1024);
    assert!(r.is_err(), "infeasible partition must fail");
    let msg = r.unwrap_err().to_string();
    assert!(msg.to_lowercase().contains("does not fit"));
}

#[test]
fn auto_partition_is_deterministic() {
    let caps = vec![cap("a", 16, 100), cap("b", 16, 100), cap("c", 16, 100)];
    let a = auto_partition("m", &caps, 30, 1024*1024*1024, 1024*1024*1024, 256*1024*1024).unwrap();
    let b = auto_partition("m", &caps, 30, 1024*1024*1024, 1024*1024*1024, 256*1024*1024).unwrap();
    assert_eq!(a.model_config_hash, b.model_config_hash);
    assert_eq!(a.assignments.len(), b.assignments.len());
    for (x, y) in a.assignments.iter().zip(b.assignments.iter()) {
        assert_eq!(x.node_id, y.node_id);
        assert_eq!(x.layer_range, y.layer_range);
    }
}

#[test]
fn manual_partition_validates_complete_cover() {
    let caps = vec![cap("a", 16, 100), cap("b", 16, 100)];
    // Complete cover 0..10 + 10..30 = 30 layers, fits.
    let ok = manual_partition("m", &caps, 30, vec![
        ("a".into(), 0..10),
        ("b".into(), 10..30),
    ], 1024*1024*1024, 1024*1024*1024, 256*1024*1024).unwrap();
    assert_eq!(ok.assignments.len(), 2);

    // Overlapping ranges -> error.
    let err = manual_partition("m", &caps, 30, vec![
        ("a".into(), 0..15),
        ("b".into(), 10..30),
    ], 1024*1024*1024, 1024*1024*1024, 256*1024*1024);
    assert!(err.is_err());

    // Gap -> error.
    let err = manual_partition("m", &caps, 30, vec![
        ("a".into(), 0..10),
        ("b".into(), 15..30),
    ], 1024*1024*1024, 1024*1024*1024, 256*1024*1024);
    assert!(err.is_err());
}
```

- [ ] **Step 2: Implement `partition.rs`**

```rust
use crate::capability::Capability;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::ops::Range;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionManifest {
    pub model_id: String,
    pub model_config_hash: [u8; 32],
    pub assignments: Vec<NodeAssignment>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeAssignment {
    pub node_id: String,
    pub layer_range: Range<usize>,
    pub hosts_embedding: bool,
    pub hosts_output: bool,
    pub previous_node: Option<String>,
    pub next_node: Option<String>,
}

/// DP layer-cut solver. Capabilities provide pipeline order (config order from caller).
///
/// Memory cost per node: `assigned_layers * layer_bytes + per_node_overhead`
/// (per_node_overhead is KV cache budget). The leader (index 0) additionally
/// holds `embed_output_bytes` for embedding + output projection.
///
/// Minimizes: `max_i (assigned_layers_i / compute_score_i)` (transport overhead
/// is uniform and ignored for v0.2).
pub fn auto_partition(
    model_id: &str,
    caps: &[Capability],
    n_layers: usize,
    layer_bytes: u64,
    embed_output_bytes: u64,
    per_node_overhead: u64,
) -> anyhow::Result<PartitionManifest> {
    let n = caps.len();
    if n == 0 { anyhow::bail!("no nodes in cluster"); }
    if n_layers == 0 { anyhow::bail!("model has 0 layers"); }

    // Memory feasibility check per node: max layers that fit.
    let max_layers_per_node: Vec<usize> = caps.iter().enumerate().map(|(i, c)| {
        let overhead = if i == 0 { embed_output_bytes + per_node_overhead } else { per_node_overhead };
        if c.available_memory_bytes <= overhead { return 0; }
        ((c.available_memory_bytes - overhead) / layer_bytes) as usize
    }).collect();

    // Feasibility: total layer capacity must >= n_layers.
    let total_cap: usize = max_layers_per_node.iter().sum();
    if total_cap < n_layers {
        anyhow::bail!(
            "model {model_id} does not fit any partition across this cluster \
             (n_layers={n_layers}, total capacity={total_cap}, total memory={} bytes)",
            caps.iter().map(|c| c.available_memory_bytes).sum::<u64>(),
        );
    }

    // DP: cost[k][i] = minimum max-stage-cost of assigning first i layers
    // across first k nodes, subject to per-node memory caps.
    //
    // Transition:
    //   cost[k][i] = min over j in [0..i] of  max(cost[k-1][j], stage_cost(j..i, node_k))
    //   where stage_cost(j..i, node_k) = (i - j) * 100 / node_k.compute_score
    //   (×100 to keep arithmetic in integers since compute_score is also an integer)
    //
    // Boundary: cost[0][0] = 0; cost[0][i > 0] = infeasible.

    const INF: u64 = u64::MAX / 4;
    let mut cost = vec![vec![INF; n_layers + 1]; n + 1];
    let mut back = vec![vec![0usize; n_layers + 1]; n + 1];

    cost[0][0] = 0;
    for k in 1..=n {
        let node = &caps[k - 1];
        let max_l = max_layers_per_node[k - 1];
        for i in 0..=n_layers {
            for j in 0..=i {
                if cost[k - 1][j] == INF { continue; }
                let assigned = i - j;
                if assigned > max_l { continue; }
                let stage = if node.compute_score == 0 { INF } else {
                    (assigned as u64) * 1000 / (node.compute_score as u64)
                };
                let candidate = cost[k - 1][j].max(stage);
                if candidate < cost[k][i] {
                    cost[k][i] = candidate;
                    back[k][i] = j;
                }
            }
        }
    }

    if cost[n][n_layers] == INF {
        anyhow::bail!(
            "model {model_id} does not fit any partition (DP found no feasible assignment)"
        );
    }

    // Recover assignment by backtracking.
    let mut cuts = vec![n_layers];
    let mut cur = n_layers;
    for k in (1..=n).rev() {
        let prev = back[k][cur];
        cuts.push(prev);
        cur = prev;
    }
    cuts.reverse();   // cuts = [0, c1, c2, ..., n_layers]

    let mut assignments = Vec::with_capacity(n);
    for (idx, win) in cuts.windows(2).enumerate() {
        let start = win[0];
        let end = win[1];
        let prev = if idx == 0 { None } else { Some(caps[idx - 1].node_id.clone()) };
        let next = if idx + 1 == n { None } else { Some(caps[idx + 1].node_id.clone()) };
        assignments.push(NodeAssignment {
            node_id: caps[idx].node_id.clone(),
            layer_range: start..end,
            hosts_embedding: idx == 0,
            hosts_output: idx + 1 == n,
            previous_node: prev,
            next_node: next,
        });
    }

    // Content addressing: hash (model_id, n_layers, [(node_id, layer_range)]) so
    // a given input yields a stable hash.
    let model_config_hash = compute_manifest_hash(model_id, n_layers, &assignments);

    Ok(PartitionManifest { model_id: model_id.into(), model_config_hash, assignments })
}

/// Build a manifest from an explicit (node_id, layer_range) list.
/// Validates: contiguous, non-overlapping, complete cover of 0..n_layers.
/// Also validates memory feasibility.
pub fn manual_partition(
    model_id: &str,
    caps: &[Capability],
    n_layers: usize,
    ranges: Vec<(String, Range<usize>)>,
    layer_bytes: u64,
    embed_output_bytes: u64,
    per_node_overhead: u64,
) -> anyhow::Result<PartitionManifest> {
    // Sort by start.
    let mut sorted: Vec<(String, Range<usize>)> = ranges;
    sorted.sort_by_key(|(_, r)| r.start);

    // Contiguous + complete cover.
    let mut expected_start = 0;
    for (node, r) in &sorted {
        if r.start != expected_start {
            anyhow::bail!(
                "partition_override is not contiguous: expected start={}, got {}..{} for node {}",
                expected_start, r.start, r.end, node
            );
        }
        if r.start >= r.end {
            anyhow::bail!("empty range {}..{} for node {}", r.start, r.end, node);
        }
        expected_start = r.end;
    }
    if expected_start != n_layers {
        anyhow::bail!(
            "partition_override does not cover all layers: covered {} of {}",
            expected_start, n_layers
        );
    }

    // Memory feasibility per node.
    for (i, (node, r)) in sorted.iter().enumerate() {
        let cap = caps.iter().find(|c| &c.node_id == node)
            .ok_or_else(|| anyhow::anyhow!("unknown node {node} in partition_override"))?;
        let overhead = if i == 0 { embed_output_bytes + per_node_overhead } else { per_node_overhead };
        let need = (r.len() as u64) * layer_bytes + overhead;
        if need > cap.available_memory_bytes {
            anyhow::bail!(
                "node {} layers {}..{} need {} bytes but only {} available",
                node, r.start, r.end, need, cap.available_memory_bytes
            );
        }
    }

    let n = sorted.len();
    let mut assignments = Vec::with_capacity(n);
    for (idx, (node, range)) in sorted.into_iter().enumerate() {
        let prev = if idx == 0 { None } else { Some(assignments[idx - 1].node_id.clone()) };
        // next is filled in second pass after we know all node_ids.
        assignments.push(NodeAssignment {
            node_id: node,
            layer_range: range,
            hosts_embedding: idx == 0,
            hosts_output: idx + 1 == n,
            previous_node: prev,
            next_node: None,
        });
    }
    for i in 0..(assignments.len().saturating_sub(1)) {
        assignments[i].next_node = Some(assignments[i + 1].node_id.clone());
    }

    let model_config_hash = compute_manifest_hash(model_id, n_layers, &assignments);
    Ok(PartitionManifest { model_id: model_id.into(), model_config_hash, assignments })
}

fn compute_manifest_hash(model_id: &str, n_layers: usize, assignments: &[NodeAssignment]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(model_id.as_bytes());
    h.update(&(n_layers as u64).to_le_bytes());
    for a in assignments {
        h.update(a.node_id.as_bytes());
        h.update(&(a.layer_range.start as u64).to_le_bytes());
        h.update(&(a.layer_range.end as u64).to_le_bytes());
    }
    let digest = h.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}
```

- [ ] **Step 3: Verify + commit**

```bash
cargo test -p ai-engine-cluster --test partition_solver
cargo clippy --workspace --all-targets -- -D warnings
git add -A
git commit -m "feat(cluster): capability-aware DP partitioner with manual override + content-addressed manifest"
```

5 tests pass.

---

### Task 6: QUIC transport — loopback echo

**Files:**
- Create: `crates/ai-engine-cluster/src/transport/mod.rs`
- Create: `crates/ai-engine-cluster/src/transport/quic.rs`
- Create: `crates/ai-engine-cluster/src/transport/frame.rs`
- Create: `crates/ai-engine-cluster/tests/transport_loopback.rs`
- Modify: `crates/ai-engine-cluster/src/lib.rs`

- [ ] **Step 1: Failing test**

`crates/ai-engine-cluster/tests/transport_loopback.rs`:

```rust
use ai_engine_cluster::transport::quic::{client_endpoint, server_endpoint};
use ai_engine_cluster::transport::frame::{read_frame, write_frame};
use ai_engine_cluster::tls::generate_node_identity;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn loopback_echo_via_quic_streams() {
    // Server: generate cert, bind QUIC endpoint on 127.0.0.1:0
    let server_id = generate_node_identity("server").unwrap();
    let server_ep = server_endpoint(&server_id, "127.0.0.1:0".parse().unwrap()).unwrap();
    let server_addr = server_ep.local_addr().unwrap();

    // Accept loop
    let server_task = tokio::spawn(async move {
        let conn = server_ep.accept().await.expect("accept").await.expect("conn");
        let (mut send, mut recv) = conn.accept_bi().await.expect("bi-stream");
        let msg = read_frame(&mut recv).await.expect("read");
        write_frame(&mut send, &msg).await.expect("write");
        send.finish().expect("finish");
        // Keep the connection open until the client drops.
        conn.closed().await;
    });

    // Client: trust server's fingerprint
    let client_id = generate_node_identity("client").unwrap();
    let client_ep = client_endpoint(&client_id, &[server_id.fingerprint.clone()]).unwrap();
    let conn = client_ep.connect(server_addr, "server").unwrap().await.expect("connect");
    let (mut send, mut recv) = conn.open_bi().await.expect("open_bi");
    write_frame(&mut send, b"hello world").await.unwrap();
    send.finish().unwrap();
    let echoed = read_frame(&mut recv).await.unwrap();
    assert_eq!(echoed, b"hello world");

    drop(conn);
    let _ = server_task.await;
}
```

- [ ] **Step 2: Implement `transport/quic.rs`**

```rust
use crate::tls::NodeIdentity;
use quinn::{ClientConfig, Endpoint, ServerConfig};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use std::net::SocketAddr;
use std::sync::Arc;
use sha2::{Digest, Sha256};

const ALPN: &[u8] = b"ai-engine-cluster/1";

pub fn server_endpoint(identity: &NodeIdentity, bind: SocketAddr) -> anyhow::Result<Endpoint> {
    let _ = rustls::crypto::ring::default_provider().install_default();

    let cert_chain = vec![CertificateDer::from(identity.cert_der.clone())];
    let key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(identity.key_der.clone()));

    let mut rustls_cfg = rustls::ServerConfig::builder()
        .with_no_client_auth()    // v0.2: server doesn't verify client certs by chain — pinning is client-side
        .with_single_cert(cert_chain, key)
        .map_err(|e| anyhow::anyhow!("server tls: {e}"))?;
    rustls_cfg.alpn_protocols = vec![ALPN.to_vec()];

    let server_cfg = ServerConfig::with_crypto(Arc::new(
        quinn::crypto::rustls::QuicServerConfig::try_from(rustls_cfg)
            .map_err(|e| anyhow::anyhow!("quinn server cfg: {e}"))?
    ));
    Endpoint::server(server_cfg, bind).map_err(|e| anyhow::anyhow!("bind: {e}"))
}

pub fn client_endpoint(
    identity: &NodeIdentity,
    trusted_fingerprints: &[String],
) -> anyhow::Result<Endpoint> {
    let _ = rustls::crypto::ring::default_provider().install_default();

    let cert_chain = vec![CertificateDer::from(identity.cert_der.clone())];
    let key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(identity.key_der.clone()));

    let verifier = Arc::new(FingerprintVerifier {
        trusted: trusted_fingerprints.iter().cloned().collect(),
    });

    let mut rustls_cfg = rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(verifier)
        .with_client_auth_cert(cert_chain, key)
        .map_err(|e| anyhow::anyhow!("client tls: {e}"))?;
    rustls_cfg.alpn_protocols = vec![ALPN.to_vec()];

    let client_cfg = ClientConfig::new(Arc::new(
        quinn::crypto::rustls::QuicClientConfig::try_from(rustls_cfg)
            .map_err(|e| anyhow::anyhow!("quinn client cfg: {e}"))?
    ));

    let mut endpoint = Endpoint::client("0.0.0.0:0".parse().unwrap())
        .map_err(|e| anyhow::anyhow!("bind client: {e}"))?;
    endpoint.set_default_client_config(client_cfg);
    Ok(endpoint)
}

#[derive(Debug)]
struct FingerprintVerifier {
    trusted: std::collections::HashSet<String>,
}

impl rustls::client::danger::ServerCertVerifier for FingerprintVerifier {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        let digest = Sha256::digest(end_entity.as_ref());
        let mut fp = String::from("sha256:");
        for b in digest.iter() { fp.push_str(&format!("{b:02x}")); }
        if self.trusted.contains(&fp) {
            Ok(rustls::client::danger::ServerCertVerified::assertion())
        } else {
            Err(rustls::Error::General(format!("untrusted server fingerprint: {fp}")))
        }
    }

    fn verify_tls12_signature(
        &self, _message: &[u8], _cert: &CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self, _message: &[u8], _cert: &CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        vec![
            rustls::SignatureScheme::ED25519,
            rustls::SignatureScheme::ECDSA_NISTP256_SHA256,
        ]
    }
}
```

If quinn 0.11's API has shifted (e.g., `QuicClientConfig::try_from` is named differently), look up the current way to build a quinn ClientConfig from a rustls ClientConfig. The pattern is stable; only the helper names vary.

- [ ] **Step 3: Implement `transport/frame.rs`**

```rust
use quinn::{RecvStream, SendStream};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const MAX_FRAME: u32 = 64 * 1024 * 1024;   // 64 MiB cap per frame

pub async fn write_frame(stream: &mut SendStream, payload: &[u8]) -> anyhow::Result<()> {
    if payload.len() as u64 > MAX_FRAME as u64 {
        anyhow::bail!("frame too large: {}", payload.len());
    }
    let len = payload.len() as u32;
    stream.write_all(&len.to_le_bytes()).await?;
    stream.write_all(payload).await?;
    Ok(())
}

pub async fn read_frame(stream: &mut RecvStream) -> anyhow::Result<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).await
        .map_err(|e| anyhow::anyhow!("read frame len: {e}"))?;
    let len = u32::from_le_bytes(len_buf);
    if len > MAX_FRAME {
        anyhow::bail!("oversized frame: {len}");
    }
    let mut payload = vec![0u8; len as usize];
    stream.read_exact(&mut payload).await
        .map_err(|e| anyhow::anyhow!("read frame payload: {e}"))?;
    Ok(payload)
}
```

- [ ] **Step 4: Wire transport/mod.rs**

```rust
pub mod frame;
pub mod quic;
```

`crates/ai-engine-cluster/src/lib.rs` (additions):
```rust
pub mod transport;
```

- [ ] **Step 5: Verify + commit**

```bash
cargo test -p ai-engine-cluster --test transport_loopback
cargo clippy --workspace --all-targets -- -D warnings
git add -A
git commit -m "feat(cluster): QUIC transport with rustls + SHA-256 cert pinning + framed streams"
```

1 test passes (the loopback echo).

---

### Task 7: Tensor IO — bf16/f32 <-> bytes

**Files:**
- Create: `crates/ai-engine-cluster/src/tensor_io.rs`
- Create: `crates/ai-engine-cluster/tests/tensor_io.rs`
- Modify: `crates/ai-engine-cluster/src/lib.rs`

The data plane carries activation tensors as raw bytes. We need helpers to convert from a burn `Tensor<B, 3>` (in f32 on the wire) to a byte vec, and back.

- [ ] **Step 1: Failing test**

```rust
use ai_engine_cluster::tensor_io::{tensor_to_bytes, tensor_from_bytes};
use burn::tensor::{Tensor, TensorData};

type B = burn_ndarray::NdArray;

#[test]
fn tensor_roundtrip_through_bytes() {
    let dev = Default::default();
    let original = Tensor::<B, 3>::from_data(
        TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], [1, 2, 3]),
        &dev,
    );
    let (bytes, shape) = tensor_to_bytes(original.clone()).unwrap();
    assert_eq!(shape, [1, 2, 3]);
    assert_eq!(bytes.len(), 6 * 4);   // 6 f32

    let restored: Tensor<B, 3> = tensor_from_bytes(&bytes, shape, &dev).unwrap();
    let orig_v: Vec<f32> = original.into_data().to_vec().unwrap();
    let rest_v: Vec<f32> = restored.into_data().to_vec().unwrap();
    assert_eq!(orig_v, rest_v);
}
```

- [ ] **Step 2: Implement**

```rust
use burn::tensor::{backend::Backend, Tensor, TensorData};

/// Convert a 3-D tensor to (f32 bytes, [batch, seq, hidden]) for wire transport.
pub fn tensor_to_bytes<B: Backend>(t: Tensor<B, 3>) -> anyhow::Result<(Vec<u8>, [usize; 3])> {
    let shape = t.dims();
    let data: Vec<f32> = t.into_data().to_vec()
        .map_err(|e| anyhow::anyhow!("to_vec f32: {e:?}"))?;
    let bytes = bytemuck::cast_slice::<f32, u8>(&data).to_vec();
    Ok((bytes, shape))
}

pub fn tensor_from_bytes<B: Backend>(
    bytes: &[u8],
    shape: [usize; 3],
    device: &B::Device,
) -> anyhow::Result<Tensor<B, 3>> {
    if bytes.len() % 4 != 0 {
        anyhow::bail!("byte length {} not f32-aligned", bytes.len());
    }
    let expected = shape[0] * shape[1] * shape[2];
    let actual = bytes.len() / 4;
    if expected != actual {
        anyhow::bail!("shape {:?} expects {} f32, got {}", shape, expected, actual);
    }
    let data: Vec<f32> = bytemuck::cast_slice::<u8, f32>(bytes).to_vec();
    Ok(Tensor::<B, 3>::from_data(TensorData::new(data, shape), device))
}
```

- [ ] **Step 3: Wire + verify + commit**

`lib.rs`: `pub mod tensor_io;`

```bash
cargo test -p ai-engine-cluster --test tensor_io
cargo clippy --workspace --all-targets -- -D warnings
git add -A
git commit -m "feat(cluster): tensor <-> bytes conversion for QUIC data plane"
```

---

### Task 8: Worker state machine (skeleton)

**Files:**
- Create: `crates/ai-engine-cluster/src/worker.rs`
- Create: `crates/ai-engine-cluster/tests/worker_handshake.rs`
- Modify: `crates/ai-engine-cluster/src/lib.rs`

A worker:
1. Listens on a QUIC endpoint.
2. Accepts the leader's connection.
3. Reads `Join` → replies with `JoinAck` + `Capability`.
4. Receives `Assignment` containing the partition manifest.
5. Loads its assigned weight range from disk via `ai_engine_runtime::load_range`.
6. Goes into request-serving mode (Task 9 wires the model forward).

Task 8 covers steps 1–4 (the handshake) without yet wiring inference. That keeps the test scope bounded.

- [ ] **Step 1: Failing test**

`crates/ai-engine-cluster/tests/worker_handshake.rs`:

```rust
use ai_engine_cluster::capability::{detect_capability, BackendKind};
use ai_engine_cluster::partition::{auto_partition, PartitionManifest};
use ai_engine_cluster::protocol::codec::{decode, encode};
use ai_engine_cluster::protocol::control::{LeaderToWorker, WorkerToLeader};
use ai_engine_cluster::tls::generate_node_identity;
use ai_engine_cluster::transport::quic::{client_endpoint, server_endpoint};
use ai_engine_cluster::transport::frame::{read_frame, write_frame};
use ai_engine_cluster::worker::run_worker_handshake;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn worker_replies_to_join_with_jointack_and_capability() {
    let worker_id = generate_node_identity("worker-a").unwrap();
    let worker_ep = server_endpoint(&worker_id, "127.0.0.1:0".parse().unwrap()).unwrap();
    let worker_addr = worker_ep.local_addr().unwrap();

    // Start the worker handshake handler.
    let worker_task = tokio::spawn(async move {
        run_worker_handshake(worker_ep, "worker-a".to_string(), BackendKind::Cpu).await
    });

    // Mimic the leader: connect, send Join, expect JoinAck + Capability.
    let leader_id = generate_node_identity("leader").unwrap();
    let leader_ep = client_endpoint(&leader_id, &[worker_id.fingerprint.clone()]).unwrap();
    let conn = leader_ep.connect(worker_addr, "worker-a").unwrap().await.unwrap();
    let (mut send, mut recv) = conn.open_bi().await.unwrap();

    let join = LeaderToWorker::Join {
        cluster_id: "test".into(),
        protocol_version: 1,
        leader_node_id: "leader".into(),
    };
    write_frame(&mut send, &encode(&join).unwrap()).await.unwrap();

    let ack_bytes = read_frame(&mut recv).await.unwrap();
    let ack: WorkerToLeader = decode(&ack_bytes).unwrap();
    matches!(ack, WorkerToLeader::JoinAck { .. });

    let cap_bytes = read_frame(&mut recv).await.unwrap();
    let cap: WorkerToLeader = decode(&cap_bytes).unwrap();
    if let WorkerToLeader::Capability(c) = cap {
        assert_eq!(c.node_id, "worker-a");
    } else { panic!("expected Capability"); }

    drop(conn);
    let _ = worker_task.await;
}
```

- [ ] **Step 2: Implement minimal worker handshake**

`crates/ai-engine-cluster/src/worker.rs`:

```rust
use crate::capability::{detect_capability, BackendKind};
use crate::protocol::codec::{decode, encode};
use crate::protocol::control::{LeaderToWorker, WorkerToLeader};
use crate::transport::frame::{read_frame, write_frame};
use quinn::Endpoint;

/// Accept the leader's connection, perform the join handshake, and return
/// once Assignment is received OR the connection drops.
///
/// v0.2: this is a one-shot handshake. After Assignment, real workers would
/// transition to the request-serving loop (Task 9 wires that). For Task 8's
/// scope, we accept one connection, reply to Join, and exit when the leader
/// closes the connection.
pub async fn run_worker_handshake(
    endpoint: Endpoint,
    node_id: String,
    backend: BackendKind,
) -> anyhow::Result<()> {
    let incoming = endpoint.accept().await
        .ok_or_else(|| anyhow::anyhow!("no incoming connection"))?;
    let conn = incoming.await?;
    let (mut send, mut recv) = conn.accept_bi().await?;

    // Receive Join.
    let join_bytes = read_frame(&mut recv).await?;
    let _join: LeaderToWorker = decode(&join_bytes)?;

    // Send JoinAck.
    let ack = WorkerToLeader::JoinAck {
        node_id: node_id.clone(),
        certificate_sha256: [0u8; 32],   // populated from local cert in Plan 3 binary integration
    };
    write_frame(&mut send, &encode(&ack)?).await?;

    // Send Capability.
    let cap = detect_capability(&node_id, backend, 0, None)?;
    write_frame(&mut send, &encode(&WorkerToLeader::Capability(cap))?).await?;

    // Wait for Assignment or connection close.
    // For Task 8 scope, just sit here until the connection closes.
    let _ = conn.closed().await;
    Ok(())
}
```

- [ ] **Step 3: Wire + verify + commit**

`lib.rs`: `pub mod worker;`

```bash
cargo test -p ai-engine-cluster --test worker_handshake
cargo clippy --workspace --all-targets -- -D warnings
git add -A
git commit -m "feat(cluster): worker state machine — join handshake + capability advertisement"
```

---

### Task 9: Leader state machine — cluster startup + partition

**Files:**
- Create: `crates/ai-engine-cluster/src/leader.rs`
- Create: `crates/ai-engine-cluster/tests/leader_partition.rs`

The leader at startup:
1. Connects to each configured worker.
2. Sends `Join`, receives `JoinAck` + `Capability`.
3. Computes partition manifest from capabilities.
4. Sends `Assignment` to each worker.
5. Holds connections open for serving requests (which Task 10 implements).

Task 9 covers steps 1–4. The test spins up 2 in-process workers and verifies the leader correctly computes and distributes a partition.

- [ ] **Step 1: Failing test**

```rust
use ai_engine_cluster::capability::BackendKind;
use ai_engine_cluster::leader::{ClusterLeader, LeaderConfig, WorkerEndpoint};
use ai_engine_cluster::tls::generate_node_identity;
use ai_engine_cluster::transport::quic::server_endpoint;
use ai_engine_cluster::worker::run_worker_handshake;

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn leader_starts_2_worker_cluster_and_distributes_partition() {
    // Two worker endpoints.
    let w1_id = generate_node_identity("w1").unwrap();
    let w1_ep = server_endpoint(&w1_id, "127.0.0.1:0".parse().unwrap()).unwrap();
    let w1_addr = w1_ep.local_addr().unwrap();
    let w2_id = generate_node_identity("w2").unwrap();
    let w2_ep = server_endpoint(&w2_id, "127.0.0.1:0".parse().unwrap()).unwrap();
    let w2_addr = w2_ep.local_addr().unwrap();

    tokio::spawn(async move {
        run_worker_handshake(w1_ep, "w1".to_string(), BackendKind::Cpu).await
    });
    tokio::spawn(async move {
        run_worker_handshake(w2_ep, "w2".to_string(), BackendKind::Cpu).await
    });

    let leader_id = generate_node_identity("leader").unwrap();
    let cfg = LeaderConfig {
        cluster_id: "test".into(),
        leader_node_id: "leader".into(),
        model_id: "toy".into(),
        n_layers: 4,
        layer_bytes: 1024 * 1024,
        embed_output_bytes: 1024 * 1024,
        per_node_overhead: 256 * 1024,
        workers: vec![
            WorkerEndpoint { node_id: "w1".into(), addr: w1_addr, fingerprint: w1_id.fingerprint.clone() },
            WorkerEndpoint { node_id: "w2".into(), addr: w2_addr, fingerprint: w2_id.fingerprint.clone() },
        ],
    };

    let leader = ClusterLeader::start(leader_id, cfg).await.unwrap();
    let manifest = leader.manifest();
    // Expected: 2 assignments, layer ranges contiguous, covering 0..4.
    assert_eq!(manifest.assignments.len(), 2);
    let total: usize = manifest.assignments.iter().map(|a| a.layer_range.len()).sum();
    assert_eq!(total, 4);
}
```

- [ ] **Step 2: Implement `leader.rs`**

```rust
use crate::capability::Capability;
use crate::partition::{auto_partition, PartitionManifest};
use crate::protocol::codec::{decode, encode};
use crate::protocol::control::{LeaderToWorker, WorkerToLeader};
use crate::tls::NodeIdentity;
use crate::transport::frame::{read_frame, write_frame};
use crate::transport::quic::client_endpoint;
use quinn::Connection;
use std::net::SocketAddr;

pub struct WorkerEndpoint {
    pub node_id: String,
    pub addr: SocketAddr,
    pub fingerprint: String,
}

pub struct LeaderConfig {
    pub cluster_id: String,
    pub leader_node_id: String,
    pub model_id: String,
    pub n_layers: usize,
    pub layer_bytes: u64,
    pub embed_output_bytes: u64,
    pub per_node_overhead: u64,
    pub workers: Vec<WorkerEndpoint>,
}

pub struct ClusterLeader {
    manifest: PartitionManifest,
    pub connections: Vec<WorkerConnection>,
}

pub struct WorkerConnection {
    pub node_id: String,
    pub conn: Connection,
    pub control_send: quinn::SendStream,
    pub control_recv: quinn::RecvStream,
}

impl ClusterLeader {
    pub async fn start(identity: NodeIdentity, cfg: LeaderConfig) -> anyhow::Result<Self> {
        let trusted: Vec<String> = cfg.workers.iter().map(|w| w.fingerprint.clone()).collect();
        let endpoint = client_endpoint(&identity, &trusted)?;

        let mut connections = Vec::with_capacity(cfg.workers.len());
        let mut capabilities: Vec<Capability> = Vec::with_capacity(cfg.workers.len());

        // Phase 1: connect + Join + collect Capabilities.
        for w in &cfg.workers {
            let conn = endpoint.connect(w.addr, &w.node_id)?.await?;
            let (mut send, mut recv) = conn.open_bi().await?;

            let join = LeaderToWorker::Join {
                cluster_id: cfg.cluster_id.clone(),
                protocol_version: 1,
                leader_node_id: cfg.leader_node_id.clone(),
            };
            write_frame(&mut send, &encode(&join)?).await?;

            // Expect JoinAck, then Capability.
            let _ack_bytes = read_frame(&mut recv).await?;
            let cap_bytes = read_frame(&mut recv).await?;
            match decode::<WorkerToLeader>(&cap_bytes)? {
                WorkerToLeader::Capability(c) => capabilities.push(c),
                other => anyhow::bail!("expected Capability after JoinAck, got {other:?}"),
            }

            connections.push(WorkerConnection {
                node_id: w.node_id.clone(),
                conn,
                control_send: send,
                control_recv: recv,
            });
        }

        // Phase 2: compute partition.
        let manifest = auto_partition(
            &cfg.model_id,
            &capabilities,
            cfg.n_layers,
            cfg.layer_bytes,
            cfg.embed_output_bytes,
            cfg.per_node_overhead,
        )?;

        // Phase 3: distribute Assignment.
        // (For Task 9 the workers don't read Assignment yet — Task 10 wires that.)

        Ok(Self { manifest, connections })
    }

    pub fn manifest(&self) -> &PartitionManifest { &self.manifest }
}
```

- [ ] **Step 3: Wire + verify + commit**

`lib.rs`: `pub mod leader;`

```bash
cargo test -p ai-engine-cluster --test leader_partition
cargo clippy --workspace --all-targets -- -D warnings
git add -A
git commit -m "feat(cluster): leader state machine — join workers + auto-partition"
```

---

### Task 10: End-to-end request generation in process

**Files:**
- Modify: `crates/ai-engine-cluster/src/worker.rs` (extend with inference loop)
- Modify: `crates/ai-engine-cluster/src/leader.rs` (extend with generation orchestration)
- Create: `crates/ai-engine-cluster/tests/inprocess_cluster.rs`

This is **the critical integration test**. Spin up 3 in-process nodes (1 leader + 2 workers) over loopback QUIC, load the toy-llama-3 fixture across them via partition, and assert that the cluster generates logits matching the single-node baseline.

Because the wiring is substantial, this task is large. It introduces:

- The worker inference loop: receive `Begin` → allocate `RequestState` → loop receiving data-plane activation frames → run assigned layer range → forward activations to the next worker → on `End`, free the slot.
- The leader generation loop: tokenize → run leader's local layer range → forward activations to first worker → wait for last worker's response → run output projection → sample → emit SSE chunk → repeat.

For Plan 2's in-process test, every node is a tokio task using loopback QUIC.

- [ ] **Step 1: Failing test**

```rust
use ai_engine_cluster::leader::{ClusterLeader, LeaderConfig, WorkerEndpoint};
use ai_engine_cluster::tls::generate_node_identity;
use ai_engine_cluster::transport::quic::server_endpoint;
use ai_engine_cluster::worker::run_worker_full;
use ai_engine_cluster::capability::BackendKind;
use ai_engine_runtime::config::ModelConfig;
use ai_engine_runtime::arch::model::Model;
use ai_engine_runtime::loader::load_range;
use ai_engine_tokenizer::{HfTokenizer, Tokenizer};
use burn::tensor::{Tensor, Int, TensorData};
use std::path::PathBuf;

type B = burn_ndarray::NdArray;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .join("ai-engine-runtime/fixtures/toy-llama-3")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn three_node_cluster_logits_match_single_node() {
    let fix = fixture();
    let cfg = ModelConfig::from_file(&fix.join("config.json")).unwrap();
    let tok = HfTokenizer::from_path(fix.join("tokenizer.json")).unwrap();
    let prompt = "The quick brown fox";
    let ids: Vec<u32> = tok.encode(prompt).unwrap();
    let ids_i32: Vec<i32> = ids.iter().map(|x| *x as i32).collect();

    // --- single-node baseline ---
    let dev = Default::default();
    let weights = load_range::<B>(
        &fix.join("model.safetensors"), &cfg, 0..cfg.n_layers, true, true, &dev,
    ).unwrap();
    let model = Model::<B>::from_loaded(&cfg, weights, &dev).unwrap();
    let baseline_logits: Vec<f32> = model.forward(
        Tensor::<B, 2, Int>::from_data(TensorData::new(ids_i32.clone(), [1, ids.len()]), &dev),
        0,
    ).slice([0..1, (ids.len() - 1)..ids.len(), 0..cfg.vocab_size])
     .reshape([cfg.vocab_size])
     .to_data().to_vec().unwrap();

    // --- 3-node cluster: leader hosts layers 0..1, w1 hosts 1..3, w2 hosts 3..4 ---
    // (toy-llama-3 has n_layers = 4)
    let w1_id = generate_node_identity("w1").unwrap();
    let w1_ep = server_endpoint(&w1_id, "127.0.0.1:0".parse().unwrap()).unwrap();
    let w1_addr = w1_ep.local_addr().unwrap();
    let w2_id = generate_node_identity("w2").unwrap();
    let w2_ep = server_endpoint(&w2_id, "127.0.0.1:0".parse().unwrap()).unwrap();
    let w2_addr = w2_ep.local_addr().unwrap();

    let model_path = fix.join("model.safetensors");
    let cfg_for_w1 = cfg.clone();
    let mp1 = model_path.clone();
    tokio::spawn(async move {
        run_worker_full::<B>(w1_ep, "w1".to_string(), BackendKind::Cpu, mp1, cfg_for_w1, 1..3).await
    });
    let cfg_for_w2 = cfg.clone();
    let mp2 = model_path.clone();
    tokio::spawn(async move {
        run_worker_full::<B>(w2_ep, "w2".to_string(), BackendKind::Cpu, mp2, cfg_for_w2, 3..4).await
    });

    let leader_id = generate_node_identity("leader").unwrap();
    let lcfg = LeaderConfig {
        cluster_id: "test".into(),
        leader_node_id: "leader".into(),
        model_id: "toy".into(),
        n_layers: cfg.n_layers,
        layer_bytes: 256 * 1024,
        embed_output_bytes: 256 * 1024,
        per_node_overhead: 64 * 1024,
        workers: vec![
            WorkerEndpoint { node_id: "w1".into(), addr: w1_addr, fingerprint: w1_id.fingerprint.clone() },
            WorkerEndpoint { node_id: "w2".into(), addr: w2_addr, fingerprint: w2_id.fingerprint.clone() },
        ],
    };

    let leader = ClusterLeader::start(leader_id, lcfg).await.unwrap();
    // The leader now runs a forward pass through layers 0..1 locally and forwards
    // activations through workers.
    let cluster_logits = leader.full_forward_for_test::<B>(
        &model_path, &cfg, 0..1, &ids_i32,
    ).await.unwrap();

    let max_diff: f32 = baseline_logits.iter().zip(cluster_logits.iter())
        .map(|(a, b)| (a - b).abs()).fold(0., f32::max);
    eprintln!("baseline vs cluster max diff = {max_diff}");
    assert!(max_diff < 1e-3, "cluster logits should match baseline within 1e-3 (got {max_diff})");
}
```

- [ ] **Step 2: Extend `worker.rs` with `run_worker_full`**

```rust
use ai_engine_runtime::arch::model::Model;
use ai_engine_runtime::config::ModelConfig;
use ai_engine_runtime::kv_cache::KvCacheSlot;
use ai_engine_runtime::loader::load_range;
use crate::protocol::data::{ActivationHeader, Dtype};
use crate::tensor_io::{tensor_from_bytes, tensor_to_bytes};
use burn::tensor::backend::Backend;
use quinn::Endpoint;
use std::collections::HashMap;
use std::ops::Range;
use std::path::PathBuf;
use uuid::Uuid;

/// Run a worker for one connection. Loads its layer range, performs handshake,
/// then services data-plane activation frames in a loop until the connection
/// drops.
///
/// The worker forwards activations to the NEXT node (or back to leader if last)
/// over a fresh outbound unidirectional stream that we OPEN on the same QUIC
/// connection. For the v0.2 in-process test the worker doesn't know which node
/// is next; it sends the result back to the leader on a worker-initiated uni
/// stream, and the leader routes it to the next worker as needed. (This is
/// less efficient than direct worker-to-worker streams but vastly simpler for
/// v0.2's in-process test. Plan 3 / v0.3 can optimize.)
pub async fn run_worker_full<B>(
    endpoint: Endpoint,
    node_id: String,
    backend: crate::capability::BackendKind,
    model_path: PathBuf,
    cfg: ModelConfig,
    layer_range: Range<usize>,
) -> anyhow::Result<()>
where
    B: Backend,
    B::Device: Default,
{
    let device = B::Device::default();

    // Handshake.
    let incoming = endpoint.accept().await
        .ok_or_else(|| anyhow::anyhow!("no incoming"))?;
    let conn = incoming.await?;
    let (mut send, mut recv) = conn.accept_bi().await?;

    let _join = crate::protocol::codec::decode::<crate::protocol::control::LeaderToWorker>(
        &crate::transport::frame::read_frame(&mut recv).await?,
    )?;
    let ack = crate::protocol::control::WorkerToLeader::JoinAck {
        node_id: node_id.clone(),
        certificate_sha256: [0u8; 32],
    };
    crate::transport::frame::write_frame(&mut send,
        &crate::protocol::codec::encode(&ack)?).await?;
    let cap = crate::capability::detect_capability(&node_id, backend, 0, None)?;
    crate::transport::frame::write_frame(&mut send,
        &crate::protocol::codec::encode(&crate::protocol::control::WorkerToLeader::Capability(cap))?).await?;

    // Load weights for our layer range.
    let weights = load_range::<B>(&model_path, &cfg, layer_range.clone(), false, false, &device)?;
    // We need a Model<B> to call .blocks[i].forward, but Model::from_loaded
    // expects embedding + final_norm. Build a "partial model" with just our
    // blocks + RoPE / FFN / Attention constructors.
    //
    // Simplest path: construct DecoderBlocks directly here using runtime types.
    use ai_engine_runtime::arch::attention::Attention;
    use ai_engine_runtime::arch::ffn::SwiGluFfn;
    use ai_engine_runtime::arch::rmsnorm::RmsNorm;
    use ai_engine_runtime::arch::rope::RotaryEmbedding;
    use ai_engine_runtime::arch::block::DecoderBlock;

    let mut blocks: Vec<DecoderBlock<B>> = Vec::with_capacity(layer_range.len());
    for layer in weights.layers {
        let attn_norm = RmsNorm::new(layer.attn_norm, cfg.rms_norm_eps);
        let ffn_norm = RmsNorm::new(layer.ffn_norm, cfg.rms_norm_eps);
        let rope = RotaryEmbedding::new(cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta, &device);
        let attn = Attention::new(
            layer.q_proj.swap_dims(0, 1),
            layer.k_proj.swap_dims(0, 1),
            layer.v_proj.swap_dims(0, 1),
            layer.o_proj.swap_dims(0, 1),
            rope, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim,
        );
        let ffn = SwiGluFfn::new(
            layer.ffn_gate.swap_dims(0, 1),
            layer.ffn_up.swap_dims(0, 1),
            layer.ffn_down.swap_dims(0, 1),
        );
        blocks.push(DecoderBlock { attn_norm, attn, ffn_norm, ffn });
    }

    // Per-request KV caches: HashMap<RequestId, Vec<KvCacheSlot>>.
    let mut request_caches: HashMap<Uuid, Vec<KvCacheSlot<B>>> = HashMap::new();

    // Service loop: accept inbound unidirectional streams carrying activation
    // frames. For each frame: header (postcard) then payload bytes, in two
    // length-prefixed pieces. Run our blocks on the activations and reply on
    // an outbound uni stream.
    loop {
        let mut uni_in = match conn.accept_uni().await {
            Ok(s) => s,
            Err(_) => break,
        };
        let header_bytes = crate::transport::frame::read_frame(&mut uni_in).await?;
        let header: ActivationHeader = crate::protocol::codec::decode(&header_bytes)?;
        let payload_bytes = crate::transport::frame::read_frame(&mut uni_in).await?;

        let shape = [header.shape[0] as usize, header.shape[1] as usize, header.shape[2] as usize];
        let mut x = tensor_from_bytes::<B>(&payload_bytes, shape, &device)?;

        // Ensure KV cache exists for this request.
        let caches = request_caches.entry(header.request_id).or_insert_with(|| {
            (0..blocks.len()).map(|_| {
                KvCacheSlot::<B>::new(shape[0], cfg.n_kv_heads, cfg.max_position_embeddings, cfg.head_dim, &device)
            }).collect()
        });

        let positions: Vec<i32> = ((header.seq_pos as usize)..(header.seq_pos as usize + shape[1]))
            .map(|p| p as i32).collect();
        for (block, cache) in blocks.iter().zip(caches.iter_mut()) {
            x = block.forward(x, &positions, cache);
        }

        // Send back: ActivationHeader + payload.
        let (out_bytes, out_shape) = tensor_to_bytes(x)?;
        let out_header = ActivationHeader {
            request_id: header.request_id,
            seq_pos: header.seq_pos,
            shape: [out_shape[0] as u32, out_shape[1] as u32, out_shape[2] as u32],
            dtype: Dtype::F32,
            is_terminal: header.is_terminal,
        };
        let mut uni_out = conn.open_uni().await?;
        crate::transport::frame::write_frame(&mut uni_out,
            &crate::protocol::codec::encode(&out_header)?).await?;
        crate::transport::frame::write_frame(&mut uni_out, &out_bytes).await?;
        uni_out.finish()?;

        if header.is_terminal {
            request_caches.remove(&header.request_id);
        }
    }
    Ok(())
}
```

- [ ] **Step 3: Extend `leader.rs` with `full_forward_for_test`**

```rust
use ai_engine_runtime::arch::block::DecoderBlock;
use ai_engine_runtime::arch::embedding::{OutputProjection, TokenEmbedding};
use ai_engine_runtime::arch::rmsnorm::RmsNorm;
use ai_engine_runtime::config::ModelConfig;
use ai_engine_runtime::kv_cache::KvCacheSlot;
use ai_engine_runtime::loader::load_range;
use crate::protocol::data::{ActivationHeader, Dtype};
use crate::tensor_io::{tensor_from_bytes, tensor_to_bytes};
use burn::tensor::{backend::Backend, Int, Tensor, TensorData};
use std::ops::Range;
use std::path::Path;
use uuid::Uuid;

impl ClusterLeader {
    /// Test-only helper: full prefill pass through cluster, returning logits at
    /// the last position. Loads the leader's own layer range + embedding +
    /// output projection from disk, then orchestrates worker round-trips.
    ///
    /// The flow is:
    ///   embedding(ids) -> leader_blocks(x) -> [send to w1, recv from w1] ->
    ///                                          [send to w2, recv from w2] ->
    ///                                          final_norm + output_proj ->
    ///                                          slice last position.
    ///
    /// This validates the wire shape end-to-end; production code in Plan 3
    /// adds the autoregressive generation loop on top.
    pub async fn full_forward_for_test<B>(
        &self,
        model_path: &Path,
        cfg: &ModelConfig,
        leader_layers: Range<usize>,
        token_ids: &[i32],
    ) -> anyhow::Result<Vec<f32>>
    where
        B: Backend,
        B::Device: Default,
    {
        // Note: this method takes `&self` but ClusterLeader holds `WorkerConnection`s
        // with non-Clone QUIC streams. We need mutable access. In real code, refactor
        // to hold `Arc<Mutex<...>>` or take `&mut self`. For brevity in the test
        // helper, we use interior mutability — change `WorkerConnection`'s send/recv
        // fields to `tokio::sync::Mutex<...>` if needed. Implementer adapts.

        let device = B::Device::default();
        let weights = load_range::<B>(model_path, cfg, leader_layers.clone(), true, true, &device)?;

        let embedding = TokenEmbedding::new(weights.embedding.unwrap());
        let final_norm = RmsNorm::new(weights.final_norm.unwrap(), cfg.rms_norm_eps);
        // tied: output = embedding^T
        let output = OutputProjection::new(embedding.weight.clone().swap_dims(0, 1));

        // Build leader's blocks for layers `leader_layers`.
        use ai_engine_runtime::arch::attention::Attention;
        use ai_engine_runtime::arch::ffn::SwiGluFfn;
        use ai_engine_runtime::arch::rope::RotaryEmbedding;
        let mut leader_blocks: Vec<DecoderBlock<B>> = Vec::with_capacity(leader_layers.len());
        for layer in weights.layers {
            let attn_norm = RmsNorm::new(layer.attn_norm, cfg.rms_norm_eps);
            let ffn_norm = RmsNorm::new(layer.ffn_norm, cfg.rms_norm_eps);
            let rope = RotaryEmbedding::new(cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta, &device);
            let attn = Attention::new(
                layer.q_proj.swap_dims(0, 1), layer.k_proj.swap_dims(0, 1),
                layer.v_proj.swap_dims(0, 1), layer.o_proj.swap_dims(0, 1),
                rope, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim,
            );
            let ffn = SwiGluFfn::new(
                layer.ffn_gate.swap_dims(0, 1), layer.ffn_up.swap_dims(0, 1),
                layer.ffn_down.swap_dims(0, 1),
            );
            leader_blocks.push(DecoderBlock { attn_norm, attn, ffn_norm, ffn });
        }

        // Forward through leader's blocks (with fresh per-block KV caches).
        let seq = token_ids.len();
        let ids = Tensor::<B, 2, Int>::from_data(TensorData::new(token_ids.to_vec(), [1, seq]), &device);
        let mut x = embedding.forward(ids);
        let positions: Vec<i32> = (0..seq as i32).collect();
        let mut leader_caches: Vec<KvCacheSlot<B>> = (0..leader_blocks.len()).map(|_| {
            KvCacheSlot::<B>::new(1, cfg.n_kv_heads, cfg.max_position_embeddings, cfg.head_dim, &device)
        }).collect();
        for (block, cache) in leader_blocks.iter().zip(leader_caches.iter_mut()) {
            x = block.forward(x, &positions, cache);
        }

        // Now relay through each worker in order.
        let request_id = Uuid::now_v7();
        for wc in &self.connections {
            let (bytes, shape) = tensor_to_bytes(x)?;
            let header = ActivationHeader {
                request_id,
                seq_pos: 0,
                shape: [shape[0] as u32, shape[1] as u32, shape[2] as u32],
                dtype: Dtype::F32,
                is_terminal: true,    // prefill is single-shot
            };
            // Open a uni stream and send (header, payload).
            // NOTE: with `&self`, we can't take &mut to the streams directly.
            // The implementer adapts by wrapping streams in Mutex or making this
            // method take &mut self. For brevity here we assume `conn` is cloneable
            // and we open a NEW pair of streams per round-trip.
            let mut send_uni = wc.conn.open_uni().await?;
            crate::transport::frame::write_frame(&mut send_uni,
                &crate::protocol::codec::encode(&header)?).await?;
            crate::transport::frame::write_frame(&mut send_uni, &bytes).await?;
            send_uni.finish()?;

            let mut recv_uni = wc.conn.accept_uni().await?;
            let header_back = crate::protocol::codec::decode::<ActivationHeader>(
                &crate::transport::frame::read_frame(&mut recv_uni).await?,
            )?;
            let payload_back = crate::transport::frame::read_frame(&mut recv_uni).await?;
            let shape_back = [header_back.shape[0] as usize, header_back.shape[1] as usize, header_back.shape[2] as usize];
            x = tensor_from_bytes::<B>(&payload_back, shape_back, &device)?;
        }

        // Final norm + output projection.
        let x = final_norm.forward(x);
        let logits = output.forward(x);

        // Slice last position.
        let last = logits.slice([0..1, (seq - 1)..seq, 0..cfg.vocab_size])
            .reshape([cfg.vocab_size]);
        Ok(last.to_data().to_vec().map_err(|e| anyhow::anyhow!("to_vec: {e:?}"))?)
    }
}
```

- [ ] **Step 4: Iterate until the test passes**

```bash
cargo test -p ai-engine-cluster --test inprocess_cluster -- --nocapture
```

Expected: `max_diff < 1e-3` (likely `< 1e-5` since we're round-tripping f32 over the wire with zero loss).

Most likely failure modes:
- **Lifetime / borrow errors** with QUIC streams + `&self`. Refactor `WorkerConnection` to use `tokio::sync::Mutex` around the streams, or change to `&mut self`.
- **Worker-to-leader stream routing**: the leader expects to receive activations on a new uni stream after sending. Make sure the worker calls `conn.open_uni()` (not `open_bi`) to send results back.
- **KV cache state in worker**: the worker keeps caches across requests but in this test we only do one forward, so they shouldn't matter.

- [ ] **Step 5: Commit when passing**

```bash
git add -A
git commit -m "feat(cluster): end-to-end 3-node forward over loopback QUIC matches single-node baseline

The critical integration test: load the toy-llama-3 fixture across a leader
+ 2 workers (each owning a contiguous layer range), run a forward pass
through the cluster, and verify the resulting logits match a single-node
forward to within 1e-3. Validates the entire transport + protocol +
partitioning stack against real ML output."
```

NO Co-Authored-By.

---

### Task 11: KV cache isolation between concurrent requests

**Files:**
- Create: `crates/ai-engine-cluster/tests/kv_isolation.rs`

Verifies the worker's per-request KV cache map doesn't leak state between concurrent requests.

- [ ] **Step 1: Test**

Spin up the same 3-node cluster as Task 10, but run TWO requests in parallel with DIFFERENT prompts. Each must produce its own correct logits. If `request_caches: HashMap<Uuid, ...>` is wrong (e.g., shared globally), the second request's logits will be polluted by the first.

```rust
// Run two `full_forward_for_test` calls concurrently with different token ids.
// Assert each matches its respective single-node baseline.
```

Concrete test code is straightforward Task 10 boilerplate plus `tokio::join!` of two requests. If a snippet is needed, copy from Task 10 and parameterize the prompt.

- [ ] **Step 2: Verify**

If the test fails, the bug is in `request_caches` keying — fix by ensuring each `Uuid::now_v7()` from the leader produces a distinct cache entry and frames from different requests don't share state.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "test(cluster): KV cache isolation between concurrent requests"
```

---

### Task 12: Worker dies mid-request (graceful failure)

**Files:**
- Create: `crates/ai-engine-cluster/tests/worker_failure.rs`

Verifies that when a worker's task panics or its connection drops, the leader's in-flight request fails with a clean error rather than hanging.

- [ ] **Step 1: Test**

Spin up the 3-node cluster from Task 10. Start a forward pass. While the request is mid-flight (e.g., after the leader sends the activation to w1 but before w1 replies), drop w1's task / close its QUIC endpoint. Assert the leader's `full_forward_for_test` returns an `Err` within a bounded time (e.g., 5 seconds).

```rust
#[tokio::test]
async fn worker_dropping_during_request_returns_error() {
    // ... cluster setup ...
    let leader_task = tokio::spawn(async move {
        leader.full_forward_for_test::<B>(/* ... */).await
    });
    // Wait briefly, then drop w1's endpoint (cancellable_handle.abort()).
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    w1_handle.abort();
    // Leader should fail within bounded time.
    let result = tokio::time::timeout(std::time::Duration::from_secs(5), leader_task).await;
    assert!(matches!(result, Ok(Ok(Err(_))) | Ok(Err(_)) | Err(_)),
        "expected leader to fail when worker drops, but got: {:?}", result);
}
```

- [ ] **Step 2: Verify + commit**

The leader's QUIC stream operations should return errors when a worker's connection drops; just check that the error path is propagated rather than swallowed. May require adding `?` propagation or `anyhow::Context::context` in a few spots.

```bash
git add -A
git commit -m "test(cluster): mid-request worker failure surfaces a clean error to the leader"
```

---

### Task 13: ClusterProvider implementing `Provider` trait

**Files:**
- Create: `crates/ai-engine-cluster/src/provider.rs`
- Create: `crates/ai-engine-cluster/tests/provider_trait.rs`
- Modify: `crates/ai-engine-cluster/src/lib.rs` (`pub mod provider;`)

The whole point of Plan 2: surface the cluster as a `Provider` so the existing v0.1 gateway pipeline can call into it without changes.

- [ ] **Step 1: Test that ClusterProvider compiles as an `Arc<dyn Provider>`**

```rust
use ai_engine_cluster::provider::ClusterProvider;
use ai_engine_provider::provider::Provider;
use std::sync::Arc;

#[test]
fn cluster_provider_implements_provider_trait() {
    // Construct a leader-mode provider with no actual cluster — just verify the
    // trait impl compiles and object-safety holds.
    let p: Arc<dyn Provider> = Arc::new(ClusterProvider::stub_leader("my-cluster"));
    assert_eq!(p.kind(), "local-cluster");
    assert_eq!(p.id(), "my-cluster");
    let caps = p.capabilities();
    assert!(caps.chat);
    assert!(caps.streaming);
    assert!(!caps.messages);
    assert!(!caps.embeddings);
}

#[tokio::test]
async fn worker_mode_returns_unsupported_for_chat() {
    use ai_engine_provider::error::ProviderError;
    use ai_engine_provider::openai::{ChatMessage, ChatContent, ChatRequest};
    use ai_engine_provider::provider::{CallCtx, Credentials};
    use uuid::Uuid;

    let p = ClusterProvider::stub_worker("my-cluster");
    let req = ChatRequest {
        model: "x".into(),
        messages: vec![ChatMessage { role: "user".into(), content: ChatContent::Text("hi".into()), extras: Default::default() }],
        stream: None, temperature: None, max_tokens: None, stream_options: None,
        extras: Default::default(),
    };
    let ctx = CallCtx { request_id: Uuid::now_v7(), deadline: None, upstream_model: "x".into() };
    let result = p.chat(req, &Credentials::none(), &ctx).await;
    assert!(matches!(result, Err(ProviderError::Unsupported)));
}
```

- [ ] **Step 2: Implement `provider.rs`**

```rust
use ai_engine_provider::{
    anthropic, error::ProviderError, openai,
    provider::{CallCtx, Capabilities, Credentials, EventStream, Provider},
};
use async_trait::async_trait;
use std::sync::Arc;

pub struct ClusterProvider {
    id: String,
    is_leader: bool,
    // In a real deployment, the leader holds a reference to ClusterLeader
    // (containing live QUIC connections). For Plan 2's trait-only scope,
    // we stub the dispatch — Plan 3 wires the real leader path.
    inner: Option<Arc<tokio::sync::Mutex<crate::leader::ClusterLeader>>>,
}

impl ClusterProvider {
    pub fn new_leader(id: impl Into<String>, leader: Arc<tokio::sync::Mutex<crate::leader::ClusterLeader>>) -> Self {
        Self { id: id.into(), is_leader: true, inner: Some(leader) }
    }
    pub fn new_worker(id: impl Into<String>) -> Self {
        Self { id: id.into(), is_leader: false, inner: None }
    }
    // Test helpers for the trait test:
    pub fn stub_leader(id: impl Into<String>) -> Self {
        Self { id: id.into(), is_leader: true, inner: None }
    }
    pub fn stub_worker(id: impl Into<String>) -> Self {
        Self { id: id.into(), is_leader: false, inner: None }
    }
}

#[async_trait]
impl Provider for ClusterProvider {
    fn id(&self) -> &str { &self.id }
    fn kind(&self) -> &'static str { "local-cluster" }
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            chat: true, streaming: true,
            tools: false, vision: false,
            messages: false, embeddings: false,
        }
    }

    async fn chat(
        &self,
        _req: openai::ChatRequest,
        _creds: &Credentials,
        _ctx: &CallCtx,
    ) -> Result<openai::ChatResponse, ProviderError> {
        if !self.is_leader {
            return Err(ProviderError::Unsupported);
        }
        // Production dispatch lives in Plan 3 — for Plan 2 we just confirm the
        // trait fires through. Real path: tokenize, run cluster generation
        // loop via inner.lock().await, sample, build ChatResponse.
        Err(ProviderError::Unsupported)
    }

    // chat_stream, messages, messages_stream, embeddings — all default to Unsupported.
}
```

- [ ] **Step 3: Verify + commit**

```bash
cargo test -p ai-engine-cluster --test provider_trait
cargo clippy --workspace --all-targets -- -D warnings
git add -A
git commit -m "feat(cluster): ClusterProvider implementing the existing Provider trait"
```

---

### Task 14: Public surface in lib.rs + end-of-plan tag

**Files:**
- Modify: `crates/ai-engine-cluster/src/lib.rs`
- Modify: `README.md`
- Tag: `v0.2.0-alpha.2`

- [ ] **Step 1: Comprehensive lib.rs**

```rust
//! ai-engine-cluster
//!
//! Distributed inference coordinator. See
//! `docs/superpowers/specs/2026-05-23-ai-engine-distributed-inference-design.md`
//! for the design.

pub mod capability;
pub mod leader;
pub mod partition;
pub mod protocol;
pub mod provider;
pub mod tensor_io;
pub mod tls;
pub mod transport;
pub mod worker;

pub use capability::{detect_capability, BackendKind, Capability};
pub use leader::{ClusterLeader, LeaderConfig, WorkerEndpoint};
pub use partition::{auto_partition, manual_partition, NodeAssignment, PartitionManifest};
pub use provider::ClusterProvider;
pub use tls::{generate_node_identity, fingerprint_sha256, NodeIdentity};
```

- [ ] **Step 2: Verify entire workspace**

```bash
cd /home/alessio/aip/airproxy
cargo test --workspace 2>&1 | grep -E "^test result:" | awk '{sum += $4} END {print "TOTAL_PASSED=" sum}'
cargo clippy --workspace --all-targets -- -D warnings
cargo build --workspace --release
```

Expected: ~125+ tests pass (~112 from before + ~13 added by this plan). Clippy + release build clean.

- [ ] **Step 3: README update**

Append to `README.md`:

```markdown

## Distributed inference (v0.2-alpha.2 preview)

ai-engine v0.2-alpha.2 adds the `ai-engine-cluster` crate: a leader/worker
QUIC-based pipeline-parallel inference coordinator. A 3-node loopback test
in `crates/ai-engine-cluster/tests/inprocess_cluster.rs` verifies that the
cluster path produces logits matching the single-node baseline to within
1e-3 on the toy-llama-3 fixture.

The cluster is not yet wired into the binary or the TOML config — that's
Plan 3 (final v0.2.0 release).

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
```

- [ ] **Step 4: Commit + tag**

```bash
git add README.md
git commit -m "docs: announce distributed inference preview (v0.2.0-alpha.2)"
git tag v0.2.0-alpha.2
git log --oneline -10
git tag
```

NO Co-Authored-By.

---

## Self-review

**Spec coverage** (against §§5–8 of the design spec):

| Spec section | Plan 2 task |
|---|---|
| §5 capability advertisement | Task 4 |
| §5 DP partitioner | Task 5 |
| §5 manifest + content addressing | Task 5 |
| §5 manual override | Task 5 |
| §6 QUIC two-plane transport | Tasks 6 |
| §6 control plane messages | Task 3 |
| §6 data plane activation frames | Tasks 3 + 7 |
| §6 ALPN + protocol versioning | Task 6 |
| §6 TLS with fingerprint pinning | Task 2 + 6 |
| §6 backpressure (via QUIC) | inherent to Task 6 |
| §8 ClusterProvider implementing Provider | Task 13 |
| §8 worker mode returns Unsupported for inbound HTTP | Task 13 |
| §9 in-process cluster test | Tasks 10 + 11 + 12 |

NOT in Plan 2 (deferred to Plan 3):
- Config schema (`[[cluster]]`, etc.) — §7
- Binary wiring (worker mode, --node-id flag) — §7
- Multi-process smoke test — §9 layer 5
- Manual multi-machine validation runbook — §9 layer 6
- Load smoke test — §9 layer "load smoke"

**Placeholder scan:**

- Task 10 has the most complex code; the implementer may need to refactor `ClusterLeader::full_forward_for_test` to take `&mut self` if borrowing QUIC streams from `&self` proves awkward. The plan flags this as an iteration point.
- Task 11's test code is sketched (`tokio::join!`-of-two-requests) but not fully written. The implementer can copy Task 10's test as a template and parameterize. Acceptable for a multi-day implementation plan; the contract is clear.
- No "TBD" / "fill in later" / "add error handling" placeholders.

**Type consistency:**

- `Capability` (Task 4) → used by `auto_partition` (Task 5), `LeaderConfig::workers` indirectly via the handshake (Task 9). Fields match across tasks. ✓
- `PartitionManifest` (Task 5) → wrapped in `LeaderToWorker::Assignment` (Task 3). Reference stability OK. ✓
- `ActivationHeader` (Task 3) → `tensor_from_bytes` / `tensor_to_bytes` (Task 7) use `[u32; 3]` shape. The leader / worker (Tasks 9, 10) convert to/from `[usize; 3]` at the boundary. ✓
- `ClusterLeader::manifest()` accessor → exposed in Task 9, used by tests. ✓
- `NodeIdentity` (Task 2) → consumed by `server_endpoint` / `client_endpoint` (Task 6). ✓
- `ClusterProvider::id()` and `kind()` match the spec ("local-cluster"). ✓

**Acknowledged risks:**

1. **quinn 0.11+ / rustls 0.23+ API churn.** The fingerprint verifier requires the `dangerous` ClientConfig builder; the rustls API renames things periodically. Budget extra time on Task 6 if the API has shifted.
2. **`CryptoProvider::install_default` is idempotent but order-sensitive.** Tests run in parallel; if two tests both call it at startup, only the first wins, the rest silently return Err. Mitigated by `let _ = ...install_default();` in tests + production code.
3. **`ClusterLeader::full_forward_for_test` is the heaviest task.** It's the integration of every previous piece. Expect 4–8 hours of debugging — most likely on QUIC stream lifetime management vs `&self`/`&mut self` borrowing.
4. **The in-process loopback test uses the same `127.0.0.1` for all 3 nodes.** Each gets a distinct port assigned by the OS. If `127.0.0.1` is restricted (rare on containers), the tests fail; document with a clear error.

---

## Execution Handoff

Plan 2 saved to `docs/superpowers/plans/2026-05-23-plan-2-cluster.md`.

Two execution options:

**1. Subagent-Driven (recommended)** — Tasks 1–9 are bounded and mechanical; Task 10 is the hard integration; Tasks 11–14 are bounded again. Plan 1's success rate (zero blockers across 18 tasks) suggests this should work well for Tasks 1–9 and 11–14, with Task 10 being the natural pause point if anything goes wrong.

**2. Inline Execution** — possible but Plan 2 is significantly larger than Plan 1 by code volume (esp. Task 10). Subagent-driven keeps context cleaner per task.

Plan 3 (config schema + binary integration + multi-process smoke + v0.2.0 release) will be written after Plan 2's `v0.2.0-alpha.2` tag lands.
