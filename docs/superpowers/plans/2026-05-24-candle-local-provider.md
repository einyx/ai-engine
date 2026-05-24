# CandleProvider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `CandleProvider` — a single-node, GPU-capable, native-quantized inference provider — as a new provider `kind = "candle-local"` alongside the existing burn-based distributed cluster.

**Architecture:** A new feature-gated crate `ai-engine-candle` wraps `candle_transformers::models::quantized_llama::ModelWeights` (native Q4_0/Q4K/Q6K matmul on CUDA/Metal/CPU). It implements the existing `ai_engine_provider::Provider` trait. A replica pool of N independently-loaded `ModelWeights` serves N concurrent requests. The burn cluster is untouched.

**Tech Stack:** Rust, candle-core 0.10.2, candle-transformers 0.10.2, tokio, the existing `ai-engine-provider` / `ai-engine-tokenizer` / `ai-engine-runtime` (for `load_tokenizer_from_gguf` and `sample::sample`) crates.

**Design spec:** `docs/superpowers/specs/2026-05-24-candle-local-provider-design.md`

**Resolved API facts (from candle-transformers 0.10.2 source):**
- `candle_core::quantized::gguf_file::Content::read(reader: &mut R) -> Result<Content>`
- `ModelWeights::from_gguf(ct: Content, reader: &mut R, device: &Device) -> Result<ModelWeights>`
- `ModelWeights::forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor>` — returns last-position logits, shape `[batch, vocab]`.
- KV cache auto-resets when `index_pos == 0`. So each generation starts the prompt at `index_pos=0`; replicas are reusable across sequential requests with no explicit reset.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `crates/ai-engine-candle/Cargo.toml` | crate deps + cpu/cuda/metal features | Create |
| `crates/ai-engine-candle/src/lib.rs` | crate root, re-exports | Create |
| `crates/ai-engine-candle/src/device.rs` | device spec → `candle_core::Device` | Create |
| `crates/ai-engine-candle/src/model.rs` | single-replica wrapper: load + generate | Create |
| `crates/ai-engine-candle/src/pool.rs` | replica pool + acquire | Create |
| `crates/ai-engine-candle/src/provider.rs` | `CandleProvider` impl of `Provider` | Create |
| `Cargo.toml` (workspace) | add member + workspace deps | Modify |
| `crates/ai-engine-config/src/lib.rs` | `candle-local` kind, device, pool_size fields | Modify |
| `crates/ai-engine-config/src/validate.rs` | validation rules for candle-local | Modify |
| `crates/ai-engine/Cargo.toml` | `backend-candle` feature → dep on ai-engine-candle | Modify |
| `crates/ai-engine/src/app.rs` | feature-gated match arm for candle-local | Modify |
| `crates/ai-engine/tests/candle_smoke.rs` | env-gated real-model integration test | Create |

---

## Task 1: Scaffold the `ai-engine-candle` crate

**Files:**
- Create: `crates/ai-engine-candle/Cargo.toml`
- Create: `crates/ai-engine-candle/src/lib.rs`
- Modify: `Cargo.toml` (workspace root — add member + workspace deps)

- [ ] **Step 1.1: Add candle to workspace deps**

In the root `Cargo.toml`, under `[workspace.dependencies]`, add:

```toml
candle-core = "0.10.2"
candle-transformers = "0.10.2"
```

Add `"crates/ai-engine-candle"` to `[workspace] members`.

- [ ] **Step 1.2: Create the crate manifest**

Create `crates/ai-engine-candle/Cargo.toml`:

```toml
[package]
name = "ai-engine-candle"
version.workspace = true
edition.workspace = true
license.workspace = true

[features]
default = ["cpu"]
cpu = []
cuda = ["candle-core/cuda", "candle-transformers/cuda"]
metal = ["candle-core/metal", "candle-transformers/metal"]

[dependencies]
candle-core = { workspace = true }
candle-transformers = { workspace = true }
ai-engine-provider = { path = "../ai-engine-provider" }
ai-engine-tokenizer = { path = "../ai-engine-tokenizer" }
ai-engine-runtime = { path = "../ai-engine-runtime" }
anyhow.workspace = true
tokio = { workspace = true, features = ["sync", "rt"] }
tracing.workspace = true
uuid = { workspace = true }
async-trait = { workspace = true }
futures = { workspace = true }
```

(If any of `async-trait`, `futures`, `uuid` are not in `[workspace.dependencies]`, check how `ai-engine-cluster/Cargo.toml` declares them and mirror that — the cluster provider also implements the async `Provider` trait so the same deps apply.)

- [ ] **Step 1.3: Create a stub lib.rs**

Create `crates/ai-engine-candle/src/lib.rs`:

```rust
//! candle-backed native-quantized local GPU inference provider.
//!
//! Wraps `candle_transformers::models::quantized_llama` to run GGUF Q4/Q6
//! Llama-3 models with native quantized matmul on CUDA/Metal/CPU. Implements
//! the `ai_engine_provider::Provider` trait as `kind = "candle-local"`.

pub mod device;
pub mod model;
pub mod pool;
pub mod provider;

pub use provider::CandleProvider;
```

(The `mod` lines will not compile until the modules exist; later tasks add them. To keep this task's commit compiling, create empty placeholder files for each module with a single doc comment, then fill them in subsequent tasks. Create `device.rs`, `model.rs`, `pool.rs`, `provider.rs` each containing just `//! placeholder` for now.)

- [ ] **Step 1.4: Verify the workspace builds with candle added**

Run: `cargo build -p ai-engine-candle 2>&1 | tail -15`
Expected: clean build. **Risk:** candle 0.10.2 pulls transitive deps (`half`, `gemm`, etc.) that may version-conflict with burn 0.21's tree. Cargo allows multiple versions; if a hard conflict appears (a `links` collision or a duplicate-symbol error), report it — it may require pinning a shared dep. If the build is clean, proceed.

Also verify the rest of the workspace still builds (candle is additive):
Run: `cargo build --workspace 2>&1 | tail -5`
Expected: clean.

- [ ] **Step 1.5: Commit**

```bash
git add Cargo.toml crates/ai-engine-candle/
git commit -m "feat(candle): scaffold ai-engine-candle crate with cpu/cuda/metal features"
```

---

## Task 2: Device resolution

**Files:**
- Create (replace placeholder): `crates/ai-engine-candle/src/device.rs`
- Test: inline `#[cfg(test)]` in `device.rs`

- [ ] **Step 2.1: Write the failing test**

Replace `crates/ai-engine-candle/src/device.rs` placeholder with the function + tests below, but first write ONLY the test module to see it fail. Add at the bottom of the file:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_cpu_spec() {
        let d = resolve_device("cpu").unwrap();
        assert!(d.is_cpu());
    }

    #[test]
    fn resolve_auto_does_not_error() {
        // On a CI box with no GPU, "auto" must fall back to CPU without erroring.
        let d = resolve_device("auto").unwrap();
        // Can't assert which device without knowing hardware; just assert Ok.
        let _ = d;
    }

    #[test]
    fn resolve_unknown_spec_errors() {
        assert!(resolve_device("banana").is_err());
    }
}
```

- [ ] **Step 2.2: Run test to verify it fails**

Run: `cargo test -p ai-engine-candle device 2>&1 | tail -15`
Expected: FAIL — `resolve_device` not found (compile error).

- [ ] **Step 2.3: Implement `resolve_device`**

Put this at the top of `crates/ai-engine-candle/src/device.rs` (above the test module):

```rust
//! Device spec parsing and auto-detection for the candle backend.

use candle_core::Device;

/// Resolve a device spec string into a candle `Device`.
///
/// - `"auto"`  : CUDA(0) if available and the `cuda` feature is on, else Metal
///   if the `metal` feature is on, else CPU.
/// - `"cpu"`   : CPU.
/// - `"cuda:N"`: CUDA device N (requires `cuda` feature).
/// - `"metal"` : Metal device 0 (requires `metal` feature).
pub fn resolve_device(spec: &str) -> anyhow::Result<Device> {
    match spec {
        "auto" => {
            #[cfg(feature = "cuda")]
            {
                if let Ok(d) = Device::new_cuda(0) {
                    tracing::info!("candle device: cuda:0");
                    return Ok(d);
                }
            }
            #[cfg(feature = "metal")]
            {
                if let Ok(d) = Device::new_metal(0) {
                    tracing::info!("candle device: metal:0");
                    return Ok(d);
                }
            }
            tracing::info!("candle device: cpu (auto fallback)");
            Ok(Device::Cpu)
        }
        "cpu" => Ok(Device::Cpu),
        "metal" => {
            #[cfg(feature = "metal")]
            {
                Ok(Device::new_metal(0)?)
            }
            #[cfg(not(feature = "metal"))]
            {
                anyhow::bail!("device 'metal' requested but ai-engine-candle was built without the 'metal' feature")
            }
        }
        other if other.starts_with("cuda:") => {
            let idx: usize = other
                .strip_prefix("cuda:")
                .unwrap()
                .parse()
                .map_err(|_| anyhow::anyhow!("invalid cuda device index in '{other}'"))?;
            #[cfg(feature = "cuda")]
            {
                Ok(Device::new_cuda(idx)?)
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = idx;
                anyhow::bail!("device '{other}' requested but ai-engine-candle was built without the 'cuda' feature")
            }
        }
        other => anyhow::bail!("unknown device spec '{other}' (expected auto|cpu|metal|cuda:N)"),
    }
}
```

- [ ] **Step 2.4: Run test to verify it passes**

Run: `cargo test -p ai-engine-candle device 2>&1 | tail -15`
Expected: PASS (3 tests). On a CPU-only build, `resolve_cpu_spec` and `resolve_auto_does_not_error` pass; `resolve_unknown_spec_errors` passes.

- [ ] **Step 2.5: Commit**

```bash
git add crates/ai-engine-candle/src/device.rs
git commit -m "feat(candle): device spec resolution with auto-detect + feature gating"
```

---

## Task 3: Config schema for `candle-local`

**Files:**
- Modify: `crates/ai-engine-config/src/lib.rs`
- Modify: `crates/ai-engine-config/src/validate.rs`
- Test: `crates/ai-engine-config/src/lib.rs` (inline tests) or the existing config test file

**Pre-step:** Read `crates/ai-engine-config/src/lib.rs` to find the existing model/cluster entry struct (the one holding `id`, `kind`, `weights_path`, etc.). The candle fields attach to that struct. Read `crates/ai-engine-config/src/validate.rs` for the existing validation pattern.

- [ ] **Step 3.1: Write the failing test**

Add to the config crate's tests (inline `#[cfg(test)]` in `lib.rs` or wherever config parse tests live):

```rust
#[test]
fn parse_candle_local_model_with_defaults() {
    let toml = r#"
        [[model]]
        id = "llama-gpu"
        kind = "candle-local"
        weights_path = "/models/llama.gguf"
    "#;
    let cfg: Config = toml::from_str(toml).unwrap();
    let m = &cfg.model[0];
    assert_eq!(m.kind, "candle-local");
    assert_eq!(m.device.as_deref(), None);          // default applied at use-site
    assert_eq!(m.pool_size, None);                  // default applied at use-site
}

#[test]
fn parse_candle_local_model_explicit() {
    let toml = r#"
        [[model]]
        id = "llama-gpu"
        kind = "candle-local"
        weights_path = "/models/llama.gguf"
        device = "cuda:0"
        pool_size = 4
    "#;
    let cfg: Config = toml::from_str(toml).unwrap();
    let m = &cfg.model[0];
    assert_eq!(m.device.as_deref(), Some("cuda:0"));
    assert_eq!(m.pool_size, Some(4));
}
```

(Adjust `Config`, `cfg.model`, field names to match the actual config types you found in the pre-step. If models live under `[[cluster.model]]` or a different table, mirror that exact shape.)

- [ ] **Step 3.2: Run test to verify it fails**

Run: `cargo test -p ai-engine-config candle_local 2>&1 | tail -15`
Expected: FAIL — unknown fields `device` / `pool_size`, or `kind` value not accepted.

- [ ] **Step 3.3: Add the fields**

In the model entry struct in `crates/ai-engine-config/src/lib.rs`, add:

```rust
    /// Candle device spec (only for kind = "candle-local"). auto|cpu|metal|cuda:N.
    #[serde(default)]
    pub device: Option<String>,
    /// Number of model replicas for concurrency (only for kind = "candle-local").
    #[serde(default)]
    pub pool_size: Option<usize>,
```

(If the struct uses `#[serde(deny_unknown_fields)]`, these additions are required to parse. If model entries are an enum keyed on `kind`, add a `CandleLocal` variant instead — match the existing pattern exactly.)

- [ ] **Step 3.4: Run test to verify it passes**

Run: `cargo test -p ai-engine-config candle_local 2>&1 | tail -15`
Expected: PASS (2 tests).

- [ ] **Step 3.5: Write the failing validation test**

Add:

```rust
#[test]
fn candle_local_rejects_zero_pool_size() {
    let toml = r#"
        [[model]]
        id = "x"
        kind = "candle-local"
        weights_path = "/models/llama.gguf"
        pool_size = 0
    "#;
    let cfg: Config = toml::from_str(toml).unwrap();
    assert!(cfg.validate().is_err(), "pool_size=0 must fail validation");
}

#[test]
fn candle_local_rejects_non_gguf_weights() {
    let toml = r#"
        [[model]]
        id = "x"
        kind = "candle-local"
        weights_path = "/models/llama.safetensors"
    "#;
    let cfg: Config = toml::from_str(toml).unwrap();
    assert!(cfg.validate().is_err(), "candle-local requires a .gguf weights_path");
}
```

(Match the actual validate entrypoint — it may be `cfg.validate()` or a free function `validate(&cfg)`. Check the pre-step findings.)

- [ ] **Step 3.6: Run to verify it fails**

Run: `cargo test -p ai-engine-config candle_local_rejects 2>&1 | tail -15`
Expected: FAIL — validation currently passes (no rule yet).

- [ ] **Step 3.7: Add validation rules**

In `crates/ai-engine-config/src/validate.rs`, in the per-model validation loop, add:

```rust
        if m.kind == "candle-local" {
            if let Some(ps) = m.pool_size {
                if ps == 0 {
                    anyhow::bail!("model '{}': pool_size must be >= 1", m.id);
                }
            }
            if !m.weights_path.ends_with(".gguf") {
                anyhow::bail!(
                    "model '{}': candle-local requires a .gguf weights_path, got '{}'",
                    m.id,
                    m.weights_path
                );
            }
        }
```

(Adjust field access to the real struct. If `weights_path` is `Option<String>`, require it to be `Some` and end in `.gguf`.)

- [ ] **Step 3.8: Run to verify it passes**

Run: `cargo test -p ai-engine-config candle_local 2>&1 | tail -15`
Expected: PASS (all 4 candle tests).

- [ ] **Step 3.9: Commit**

```bash
git add crates/ai-engine-config/
git commit -m "feat(config): candle-local provider kind with device + pool_size + validation"
```

---

## Task 4: Single-replica model wrapper (`CandleModel`)

**Files:**
- Create (replace placeholder): `crates/ai-engine-candle/src/model.rs`

This wraps one `ModelWeights` + tokenizer + device + the generation loop. Note: candle's `forward` is `&mut self`, and the KV cache auto-resets at `index_pos=0`, so each `generate` call starts a clean sequence.

- [ ] **Step 4.1: Write the failing test**

The full real-model load needs a GGUF file and is exercised in the env-gated integration test (Task 8). Here, test the parts that don't need a model: the generation params struct and the stop logic. Add at the bottom of `model.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gen_params_defaults() {
        let p = GenParams::default();
        assert_eq!(p.max_tokens, 256);
        assert_eq!(p.temperature, 0.0);
    }

    #[test]
    fn should_stop_on_eos() {
        let p = GenParams { max_tokens: 100, temperature: 0.0 };
        // produced 5 tokens, last is eos -> stop
        assert!(should_stop(5, 42, 42, &p));
        // produced 5 tokens, last is not eos, under budget -> continue
        assert!(!should_stop(5, 7, 42, &p));
        // hit max_tokens -> stop
        assert!(should_stop(100, 7, 42, &p));
    }
}
```

- [ ] **Step 4.2: Run to verify it fails**

Run: `cargo test -p ai-engine-candle model 2>&1 | tail -15`
Expected: FAIL — `GenParams` / `should_stop` undefined.

- [ ] **Step 4.3: Implement `model.rs`**

Replace the placeholder with:

```rust
//! Single-replica candle model wrapper: GGUF load + autoregressive generation.

use anyhow::Context;
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_llama::ModelWeights;
use std::path::Path;
use std::sync::Arc;

use ai_engine_runtime::sample::{self, SampleParams};
use ai_engine_tokenizer::{HfTokenizer, Tokenizer};

/// Generation parameters for one chat completion.
#[derive(Debug, Clone)]
pub struct GenParams {
    pub max_tokens: usize,
    pub temperature: f32,
}

impl Default for GenParams {
    fn default() -> Self {
        Self { max_tokens: 256, temperature: 0.0 }
    }
}

/// Stop generation when we hit the EOS token or the max_tokens budget.
pub fn should_stop(produced: usize, last_token: u32, eos: u32, params: &GenParams) -> bool {
    produced >= params.max_tokens || last_token == eos
}

/// One model replica: holds candle weights (with internal KV cache), tokenizer,
/// device, and the EOS token id. `forward` mutates the KV cache; the cache
/// auto-resets when called with `index_pos == 0`, so each `generate` starts a
/// clean sequence.
pub struct CandleModel {
    weights: ModelWeights,
    tokenizer: Arc<HfTokenizer>,
    device: Device,
    eos_token_id: u32,
}

impl CandleModel {
    /// Load a GGUF Llama-3 checkpoint into a candle `ModelWeights` on `device`.
    /// The tokenizer is shared (`Arc`) across replicas.
    pub fn load(
        gguf_path: &Path,
        device: Device,
        tokenizer: Arc<HfTokenizer>,
    ) -> anyhow::Result<Self> {
        let mut file = std::fs::File::open(gguf_path)
            .with_context(|| format!("open {}", gguf_path.display()))?;
        let content = gguf_file::Content::read(&mut file)
            .with_context(|| format!("read gguf {}", gguf_path.display()))?;
        // EOS token id from GGUF metadata (Llama-3: 128009).
        let eos_token_id = content
            .metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.to_u32().ok())
            .context("gguf missing tokenizer.ggml.eos_token_id")?;
        let weights = ModelWeights::from_gguf(content, &mut file, &device)
            .map_err(|e| anyhow::anyhow!("ModelWeights::from_gguf: {e}"))?;
        Ok(Self { weights, tokenizer, device, eos_token_id })
    }

    /// Run an autoregressive generation from `prompt`. Calls `on_token` for each
    /// generated token id (for streaming). Returns the full generated token ids.
    /// `prompt_tokens_out` is set to the prompt token count (for usage).
    pub fn generate(
        &mut self,
        prompt: &str,
        params: &GenParams,
        mut on_token: impl FnMut(u32),
        prompt_tokens_out: &mut usize,
    ) -> anyhow::Result<Vec<u32>> {
        let prompt_ids = self.tokenizer.encode(prompt)?;
        *prompt_tokens_out = prompt_ids.len();
        if prompt_ids.is_empty() {
            anyhow::bail!("empty prompt after tokenization");
        }

        let sample_params = SampleParams { temperature: params.temperature };

        // Prefill: feed the whole prompt at index_pos = 0 (resets KV cache).
        let input = Tensor::new(prompt_ids.as_slice(), &self.device)?
            .reshape((1, prompt_ids.len()))?;
        let logits = self
            .weights
            .forward(&input, 0)
            .map_err(|e| anyhow::anyhow!("forward(prefill): {e}"))?;
        let logits_v: Vec<f32> = logits.squeeze(0)?.to_vec1()?;
        let mut next = sample::sample(&logits_v, &sample_params);

        let mut produced = 0usize;
        let mut out = Vec::new();
        let mut index_pos = prompt_ids.len();

        loop {
            if should_stop(produced, next, self.eos_token_id, params) {
                // include the EOS-or-budget token? No: stop before emitting EOS.
                if next != self.eos_token_id && produced < params.max_tokens {
                    // unreachable given should_stop, but keep the loop total.
                }
                break;
            }
            on_token(next);
            out.push(next);
            produced += 1;

            let input = Tensor::new(&[next], &self.device)?.reshape((1, 1))?;
            let logits = self
                .weights
                .forward(&input, index_pos)
                .map_err(|e| anyhow::anyhow!("forward(decode): {e}"))?;
            let logits_v: Vec<f32> = logits.squeeze(0)?.to_vec1()?;
            next = sample::sample(&logits_v, &sample_params);
            index_pos += 1;
        }

        Ok(out)
    }

    /// Decode token ids back to text.
    pub fn decode(&self, ids: &[u32]) -> anyhow::Result<String> {
        self.tokenizer.decode(ids)
    }
}
```

**Note for the implementer:** verify against the real `ai_engine_runtime::sample` API — this plan assumes `sample::sample(&[f32], &SampleParams) -> u32` and a `SampleParams { temperature: f32 }`. Read `crates/ai-engine-runtime/src/sample.rs` and adjust the call + struct to the actual signature (the field set may differ; e.g. it may take `temperature: Option<f32>` or include top-p). Also verify `HfTokenizer::encode -> anyhow::Result<Vec<u32>>` and `decode(&[u32]) -> anyhow::Result<String>` against `crates/ai-engine-tokenizer/src/lib.rs` (the `Tokenizer` trait). Keep the generation logic identical; only fix signatures.

- [ ] **Step 4.4: Run to verify it passes**

Run: `cargo test -p ai-engine-candle model 2>&1 | tail -15`
Expected: PASS (the 2 unit tests; the model-loading path is covered in Task 8).

- [ ] **Step 4.5: Commit**

```bash
git add crates/ai-engine-candle/src/model.rs
git commit -m "feat(candle): CandleModel wrapper with GGUF load + autoregressive generate"
```

---

## Task 5: Replica pool

**Files:**
- Create (replace placeholder): `crates/ai-engine-candle/src/pool.rs`

- [ ] **Step 5.1: Write the failing test**

The pool's acquire logic can be tested without a real model by making the pooled item generic in a test, but to keep it simple and real, test pool construction-count logic via a small helper. Add to `pool.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_robin_index_wraps() {
        // next_index cycles 0,1,0,1 for n=2
        let counter = std::sync::atomic::AtomicUsize::new(0);
        assert_eq!(next_index(&counter, 2), 0);
        assert_eq!(next_index(&counter, 2), 1);
        assert_eq!(next_index(&counter, 2), 0);
        assert_eq!(next_index(&counter, 2), 1);
    }
}
```

- [ ] **Step 5.2: Run to verify it fails**

Run: `cargo test -p ai-engine-candle pool 2>&1 | tail -15`
Expected: FAIL — `next_index` undefined.

- [ ] **Step 5.3: Implement `pool.rs`**

```rust
//! Replica pool: N independently-loaded `CandleModel`s for concurrent requests.

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::{Mutex, MutexGuard};

use ai_engine_tokenizer::HfTokenizer;
use candle_core::Device;

use crate::model::CandleModel;

/// Round-robin replica index. Pure function for testability.
pub(crate) fn next_index(counter: &AtomicUsize, n: usize) -> usize {
    counter.fetch_add(1, Ordering::Relaxed) % n
}

/// A pool of `n` model replicas. Each replica is an independently-loaded
/// `CandleModel` (weights NOT shared — candle-transformers bundles
/// weights+KV-cache with no sharing API). The tokenizer IS shared via `Arc`.
pub struct ReplicaPool {
    replicas: Vec<Mutex<CandleModel>>,
    counter: AtomicUsize,
}

impl ReplicaPool {
    /// Load the GGUF `n` times into `n` replicas on `device`.
    pub fn new(
        gguf_path: &Path,
        device: Device,
        tokenizer: Arc<HfTokenizer>,
        n: usize,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(n >= 1, "pool_size must be >= 1");
        let mut replicas = Vec::with_capacity(n);
        for i in 0..n {
            tracing::info!("candle: loading replica {}/{}", i + 1, n);
            let m = CandleModel::load(gguf_path, device.clone(), tokenizer.clone())?;
            replicas.push(Mutex::new(m));
        }
        Ok(Self { replicas, counter: AtomicUsize::new(0) })
    }

    /// Acquire a free replica. Tries each replica's `try_lock`; if all are busy,
    /// awaits the lock on a round-robin-chosen replica.
    pub async fn acquire(&self) -> MutexGuard<'_, CandleModel> {
        for r in &self.replicas {
            if let Ok(guard) = r.try_lock() {
                return guard;
            }
        }
        let idx = next_index(&self.counter, self.replicas.len());
        self.replicas[idx].lock().await
    }
}
```

- [ ] **Step 5.4: Run to verify it passes**

Run: `cargo test -p ai-engine-candle pool 2>&1 | tail -15`
Expected: PASS.

- [ ] **Step 5.5: Commit**

```bash
git add crates/ai-engine-candle/src/pool.rs
git commit -m "feat(candle): replica pool with try-lock acquire + round-robin fallback"
```

---

## Task 6: `CandleProvider` (the Provider impl)

**Files:**
- Create (replace placeholder): `crates/ai-engine-candle/src/provider.rs`

**Pre-step:** Read `crates/ai-engine-cluster/src/provider.rs` for: (1) the exact `Provider` trait method signatures (`chat`, `chat_stream`, `kind`, `id`, `capabilities`), (2) the `render_prompt` function (lines ~303-321) — duplicate it here, (3) how `ChatResponse` / streaming chunks are constructed, (4) the `Capabilities` struct fields. Mirror those exactly.

- [ ] **Step 6.1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use ai_engine_provider::openai::{ChatContent, ChatMessage, ChatRequest};

    #[test]
    fn render_prompt_concatenates_roles() {
        let req = ChatRequest {
            model: "x".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: ChatContent::Text("Hello".into()),
                extras: Default::default(),
            }],
            stream: None, temperature: None, max_tokens: None,
            stream_options: None, extras: Default::default(),
        };
        assert_eq!(render_prompt(&req), "user: Hello\n");
    }
}
```

(Match `ChatRequest`'s actual field set from the cluster provider's tests — copy the construction from `crates/ai-engine-cluster/tests/provider_trait.rs` which builds a `ChatRequest`.)

- [ ] **Step 6.2: Run to verify it fails**

Run: `cargo test -p ai-engine-candle provider 2>&1 | tail -15`
Expected: FAIL — `render_prompt` / `CandleProvider` undefined.

- [ ] **Step 6.3: Implement `provider.rs`**

```rust
//! `CandleProvider`: implements `ai_engine_provider::Provider` as
//! `kind = "candle-local"`, backed by a candle replica pool.

use std::path::Path;
use std::sync::Arc;

use ai_engine_provider::error::ProviderError;
use ai_engine_provider::openai::{self, ChatRequest};
use ai_engine_provider::provider::{CallCtx, Capabilities, Credentials, Provider};
use ai_engine_tokenizer::HfTokenizer;
use async_trait::async_trait;

use crate::device::resolve_device;
use crate::model::GenParams;
use crate::pool::ReplicaPool;

pub struct CandleProvider {
    id: String,
    pool: ReplicaPool,
}

impl CandleProvider {
    /// Build a provider: resolve device, load tokenizer from the GGUF, build the
    /// replica pool.
    pub fn new(
        id: impl Into<String>,
        gguf_path: &Path,
        device_spec: &str,
        pool_size: usize,
    ) -> anyhow::Result<Self> {
        let device = resolve_device(device_spec)?;
        let tokenizer = Arc::new(
            ai_engine_runtime::load_tokenizer_from_gguf(gguf_path)
                .map_err(|e| anyhow::anyhow!("load tokenizer from gguf: {e}"))?,
        );
        let pool = ReplicaPool::new(gguf_path, device, tokenizer, pool_size)?;
        Ok(Self { id: id.into(), pool })
    }
}

/// Render chat messages into a prompt string. Mirrors the cluster provider's
/// `render_prompt` for behavioral consistency across providers.
pub(crate) fn render_prompt(req: &ChatRequest) -> String {
    let mut out = String::new();
    for m in &req.messages {
        let text = match &m.content {
            openai::ChatContent::Text(s) => s.clone(),
            openai::ChatContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| p.get("text").and_then(|t| t.as_str()).map(String::from))
                .collect::<Vec<_>>()
                .join("\n"),
        };
        out.push_str(&m.role);
        out.push_str(": ");
        out.push_str(&text);
        out.push('\n');
    }
    out
}

#[async_trait]
impl Provider for CandleProvider {
    fn kind(&self) -> &str { "candle-local" }
    fn id(&self) -> &str { &self.id }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            chat: true,
            streaming: true,
            messages: false,
            embeddings: false,
            tools: false,
            vision: false,
        }
    }

    async fn chat(
        &self,
        req: ChatRequest,
        _creds: &Credentials,
        _ctx: &CallCtx,
    ) -> Result<openai::ChatResponse, ProviderError> {
        let prompt = render_prompt(&req);
        let params = GenParams {
            max_tokens: req.max_tokens.unwrap_or(256) as usize,
            temperature: req.temperature.unwrap_or(0.0),
        };

        // Generation is blocking CPU/GPU work; run it on a blocking thread so we
        // don't stall the async runtime. The replica guard is held for the call.
        let mut guard = self.pool.acquire().await;
        let mut prompt_tokens = 0usize;
        let ids = tokio::task::block_in_place(|| {
            guard.generate(&prompt, &params, |_t| {}, &mut prompt_tokens)
        })
        .map_err(|e| ProviderError::Upstream(format!("candle generate: {e}")))?;
        let text = guard
            .decode(&ids)
            .map_err(|e| ProviderError::Upstream(format!("candle decode: {e}")))?;
        let completion_tokens = ids.len();
        drop(guard);

        Ok(build_chat_response(&req, text, prompt_tokens, completion_tokens))
    }

    async fn chat_stream(
        &self,
        req: ChatRequest,
        _creds: &Credentials,
        _ctx: &CallCtx,
    ) -> Result<
        std::pin::Pin<Box<dyn futures::Stream<Item = Result<openai::ChatStreamChunk, ProviderError>> + Send>>,
        ProviderError,
    > {
        // Minimal streaming: generate fully on a blocking thread, pushing decoded
        // token deltas into a channel, surface the channel as a Stream. Token-level
        // SSE without holding the runtime.
        let prompt = render_prompt(&req);
        let params = GenParams {
            max_tokens: req.max_tokens.unwrap_or(256) as usize,
            temperature: req.temperature.unwrap_or(0.0),
        };
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Result<openai::ChatStreamChunk, ProviderError>>();
        // NOTE: acquiring a replica + running generate must happen on a task. The
        // implementer should spawn a task that acquires the pool guard, runs
        // `generate` with an `on_token` closure that decodes the single token and
        // sends a ChatStreamChunk delta through `tx`, then sends the final
        // finish chunk. Use the cluster provider's chat_stream as the reference
        // for the exact ChatStreamChunk shape and the finish_reason chunk.
        let model_id = req.model.clone();
        let provider = /* see implementer note below */ ();
        let _ = (&tx, &params, &prompt, &model_id, provider);
        Ok(Box::pin(tokio_stream::wrappers::UnboundedReceiverStream::new(rx)))
    }
}
```

**Implementer note (chat_stream):** the streaming impl above is a skeleton. Reference `crates/ai-engine-cluster/src/provider.rs::chat_stream` for the exact `ChatStreamChunk` construction, the per-token delta chunk shape, and the terminal finish chunk. The pool guard cannot be moved across threads if it borrows `&self`; the clean approach is to make `CandleProvider` hold `Arc<ReplicaPool>` so a spawned task can `acquire()` independently. Adjust the `new` constructor to wrap the pool in `Arc` if needed. Decode each token individually for the delta (single-token decode may merge with the previous via the tokenizer's byte-level BPE — use the same incremental-decode approach the cluster streaming uses, or decode the cumulative ids and diff). Keep `chat` (non-streaming) as the primary correctness path; `chat_stream` correctness is validated by the streaming smoke if one exists, else by manual check.

**Also implement** `build_chat_response(req, text, prompt_tokens, completion_tokens) -> openai::ChatResponse` mirroring how the cluster provider builds its response (one choice, role "assistant", `ChatContent::Text`, usage populated). Copy that construction from the cluster provider.

- [ ] **Step 6.4: Run to verify it passes**

Run: `cargo test -p ai-engine-candle provider 2>&1 | tail -15`
Expected: PASS (`render_prompt_concatenates_roles`). The full chat path is exercised in Task 8.

- [ ] **Step 6.5: Verify the crate builds + clippy**

Run:
```
cargo build -p ai-engine-candle 2>&1 | tail -5
cargo clippy -p ai-engine-candle --all-targets -- -D warnings 2>&1 | tail -5
```
Expected: clean. Fix any clippy issues (the skeleton `chat_stream` must actually compile — complete it per the implementer note rather than leaving dead bindings).

- [ ] **Step 6.6: Commit**

```bash
git add crates/ai-engine-candle/src/provider.rs
git commit -m "feat(candle): CandleProvider implementing Provider (chat + chat_stream)"
```

---

## Task 7: Wire into the gateway (`app.rs`)

**Files:**
- Modify: `crates/ai-engine/Cargo.toml`
- Modify: `crates/ai-engine/src/app.rs`

**Pre-step:** Read `crates/ai-engine/src/app.rs` to find where providers are constructed from config (the `match` on `kind` / the provider-building function). Read `crates/ai-engine/Cargo.toml` for the existing feature/dep layout.

- [ ] **Step 7.1: Add the feature + optional dep**

In `crates/ai-engine/Cargo.toml`:

```toml
[features]
# ... existing ...
backend-candle = ["dep:ai-engine-candle"]

[dependencies]
# ... existing ...
ai-engine-candle = { path = "../ai-engine-candle", optional = true }
```

- [ ] **Step 7.2: Add the match arm (feature-gated)**

In `crates/ai-engine/src/app.rs`, in the provider-construction match on `kind`, add:

```rust
            "candle-local" => {
                #[cfg(feature = "backend-candle")]
                {
                    let gguf = std::path::Path::new(&model.weights_path);
                    let device = model.device.as_deref().unwrap_or("auto");
                    let pool_size = model.pool_size.unwrap_or(2);
                    let provider = ai_engine_candle::CandleProvider::new(
                        &model.id, gguf, device, pool_size,
                    )?;
                    Arc::new(provider) as Arc<dyn Provider>
                }
                #[cfg(not(feature = "backend-candle"))]
                {
                    anyhow::bail!(
                        "model '{}' uses kind=candle-local but this binary was built without the 'backend-candle' feature; rebuild with --features backend-candle",
                        model.id
                    );
                }
            }
```

(Adjust `model.weights_path` / `model.device` / `model.pool_size` / `model.id` to the real field accessors. Match the surrounding arms' style for wrapping in `Arc<dyn Provider>`.)

- [ ] **Step 7.3: Verify both build configurations**

Run:
```
cargo build -p ai-engine 2>&1 | tail -5
cargo build -p ai-engine --features backend-candle 2>&1 | tail -5
```
Expected: both clean. The first (no candle feature) must compile the `bail!` arm; the second must compile the real arm.

- [ ] **Step 7.4: Workspace test + clippy**

Run:
```
cargo test --workspace 2>&1 | grep "test result" | tail -5
cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -3
```
Expected: existing tests still pass; clippy clean. (Workspace build here is without `backend-candle` by default — that's fine; the candle crate's own unit tests run via `cargo test -p ai-engine-candle`.)

- [ ] **Step 7.5: Commit**

```bash
git add crates/ai-engine/Cargo.toml crates/ai-engine/src/app.rs
git commit -m "feat(candle): wire candle-local provider into gateway behind backend-candle feature"
```

---

## Task 8: Real-model integration smoke + README + tag

**Files:**
- Create: `crates/ai-engine/tests/candle_smoke.rs`
- Modify: `README.md`

- [ ] **Step 8.1: Write the env-gated integration test**

Create `crates/ai-engine/tests/candle_smoke.rs`:

```rust
//! Env-gated real-model smoke for the candle-local provider. Requires:
//!   AI_ENGINE_REAL_GGUF=/path/to/Llama-3.2-1B-Instruct-Q4_0.gguf
//! Run: cargo test -p ai-engine --test candle_smoke --features backend-candle -- --ignored --nocapture

#![cfg(feature = "backend-candle")]

use ai_engine_candle::CandleProvider;
use ai_engine_provider::openai::{ChatContent, ChatMessage, ChatRequest};
use ai_engine_provider::provider::{CallCtx, Credentials, Provider};
use uuid::Uuid;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn candle_local_real_model_chat_is_coherent() {
    let gguf = match std::env::var("AI_ENGINE_REAL_GGUF") {
        Ok(p) => std::path::PathBuf::from(p),
        Err(_) => {
            eprintln!("SKIP: set AI_ENGINE_REAL_GGUF to run the candle smoke");
            return;
        }
    };
    if !gguf.exists() {
        eprintln!("SKIP: AI_ENGINE_REAL_GGUF does not exist: {}", gguf.display());
        return;
    }

    // pool_size=1 keeps the test light; device "auto" uses GPU if the cuda/metal
    // feature is compiled, else CPU.
    let provider = CandleProvider::new("llama-gpu", &gguf, "auto", 1)
        .expect("build CandleProvider");

    let req = ChatRequest {
        model: "llama-gpu".into(),
        messages: vec![ChatMessage {
            role: "user".into(),
            content: ChatContent::Text("Hello, who are you?".into()),
            extras: Default::default(),
        }],
        stream: None,
        temperature: Some(0.0),
        max_tokens: Some(20),
        stream_options: None,
        extras: Default::default(),
    };
    let ctx = CallCtx {
        request_id: Uuid::now_v7(),
        deadline: None,
        upstream_model: "llama-gpu".into(),
    };

    let t0 = std::time::Instant::now();
    let resp = provider.chat(req, &Credentials::none(), &ctx).await.expect("chat ok");
    let dt = t0.elapsed();

    let choice = &resp.choices[0];
    let text = match &choice.message.content {
        ChatContent::Text(s) => s.clone(),
        ChatContent::Parts(_) => panic!("expected Text content"),
    };
    eprintln!("CANDLE CHAT RESPONSE: {text:?}");
    let usage = resp.usage.as_ref().expect("usage populated");
    let toks = usage.completion_tokens.max(1);
    eprintln!("tok/s: {:.2}", toks as f64 / dt.as_secs_f64());

    assert!(usage.completion_tokens >= 1, "must produce at least one token");
    assert!(!text.trim().is_empty(), "decoded text must be non-empty");
}
```

(Adjust `ChatRequest` / `CallCtx` / `Usage` field names to the real types — copy the exact construction from `crates/ai-engine-cluster/tests/provider_trait.rs`, which builds these same structs.)

- [ ] **Step 8.2: Verify it compiles and the skip path works**

Run:
```
cargo test -p ai-engine --test candle_smoke --features backend-candle --no-run 2>&1 | tail -10
cargo test -p ai-engine --test candle_smoke --features backend-candle -- --ignored --nocapture 2>&1 | tail -10
```
Expected: compiles; skip-path prints SKIP and passes (env var unset).

- [ ] **Step 8.3: Run against the real model (CPU)**

```bash
AI_ENGINE_REAL_GGUF=/tmp/ai-engine-validation/model.gguf \
  cargo test -p ai-engine --test candle_smoke --features backend-candle -- --ignored --nocapture 2>&1 | tail -20
```
Expected: coherent output (e.g. `"You: I am a..."` style — matches what the burn path produced). Note the CPU tok/s.

- [ ] **Step 8.4: Run against the real model (GPU) — if CUDA available**

```bash
AI_ENGINE_REAL_GGUF=/tmp/ai-engine-validation/model.gguf \
  cargo test -p ai-engine --test candle_smoke --features backend-candle,ai-engine-candle/cuda -- --ignored --nocapture 2>&1 | tail -20
```
Expected: coherent output; tok/s should be dramatically higher than CPU (target 50-150 tok/s for the 1B model on the RTX 4070, vs candle-via-burn's earlier 6 tok/s — this path uses native quantized matmul so weights stay packed). Record the number.

If the `cuda` feature fails to build (nvcc/toolchain issue), report it; the CPU path is the correctness gate and CUDA is the perf bonus.

- [ ] **Step 8.5: README + final verification**

Run final checks:
```
cargo test --workspace 2>&1 | grep "test result" | awk '{p += $4; ig += $8} END {print "PASSED=" p " IGNORED=" ig}'
cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -3
cargo build --workspace --release 2>&1 | tail -3
```

Append a v0.4.0-alpha.1 section to `README.md` Release history:

```markdown
### v0.4.0-alpha.1 — native-quantized local GPU inference (candle)

ai-engine v0.4.0-alpha.1 adds `kind = "candle-local"`: a single-node,
GPU-capable provider backed by candle-transformers' quantized Llama-3
(`quantized_llama`). Unlike the burn cluster (which dequantizes Q4 weights to
f32), candle runs native Q4_0/Q4K/Q6K matmul on CUDA/Metal/CPU — weights stay
packed, so a 1B model fits comfortably in VRAM and decodes far faster.

\`\`\`toml
[[model]]
id = "llama-3.2-1b-gpu"
kind = "candle-local"
weights_path = "/srv/models/Llama-3.2-1B-Instruct-Q4_0.gguf"
device = "auto"     # auto | cpu | cuda:N | metal
pool_size = 2       # concurrent-request replicas
\`\`\`

Build with GPU support (requires CUDA toolkit for cuda):
\`\`\`bash
cargo build --release --features backend-candle,ai-engine-candle/cuda
\`\`\`

Measured (RTX 4070, Llama-3.2-1B-Instruct-Q4_0): <fill in the GPU tok/s from
Step 8.4>, vs 0.338 tok/s on the burn CPU path.

Scope: Llama-3 family only, single-node only. The burn distributed cluster
(`kind = "local-cluster"`) is unchanged and remains the multi-node path.
```

(Fill the measured tok/s from Step 8.4. If CUDA wasn't available, state the CPU number and note GPU is untested on this box.)

- [ ] **Step 8.6: Commit + tag**

```bash
git add crates/ai-engine/tests/candle_smoke.rs README.md
git commit -m "feat(candle): real-model smoke test + v0.4.0-alpha.1 release notes"
git tag v0.4.0-alpha.1
git log --oneline -10
```

---

## Out of Scope (deferred)

- Non-Llama architectures (Qwen, Mistral) via other candle-transformers models.
- Distributed candle inference (sharding across nodes) — the burn cluster owns this.
- Weight sharing across replicas (candle-transformers has no API; each replica loads independently).
- Continuous batching / paged attention — pool-of-N is the concurrency model.
