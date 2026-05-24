# Plan 9 — v0.3.0-alpha.5: GGUF binary wiring

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `ai-engine` binary load `.gguf` checkpoints transparently. Today the binary always calls `load_range` (safetensors). Plan 9 introduces a `load_weights` dispatch function that switches between `load_range` and `load_gguf` based on the file extension of `weights_path`. Operators set `weights_path = "/srv/models/llama-3.gguf"` in the TOML and everything else works.

**Architecture:** A new `load_weights` entry point in `ai-engine-runtime::loader` that inspects the path's extension (`.gguf` → `load_gguf`, otherwise → `load_range`). All callers (`Model::from_loaded` users, the cluster `build_leader_model` and `run_worker_full`, the binary's `build_app_state` + `worker_main`) shift from calling `load_range` directly to calling `load_weights`. Existing safetensors fixtures + tests are unaffected because `.safetensors` and bare directory paths still route to `load_range`.

**Tech Stack:** No new deps.

**Scope rule:** Plan 9 wires the EXISTING GGUF reader into the binary path. NO new ggml_types beyond Q4_0 (Plan 10 candidate). NO GGUF-embedded tokenizer extraction — `tokenizer_path` in TOML still required. NO GGUF-embedded ModelConfig extraction — `config_path` still required. These polish items wait.

**Baseline:** Branch `main` at `v0.3.0-alpha.4`. 205 passing + 5 ignored. Clippy clean.

---

### Task 1: `load_weights` dispatch function

**Files:**
- Modify: `crates/ai-engine-runtime/src/loader.rs`
- Create: `crates/ai-engine-runtime/tests/load_weights_dispatch.rs`

- [ ] **Step 1: Failing test**

`crates/ai-engine-runtime/tests/load_weights_dispatch.rs`:

```rust
use ai_engine_runtime::arch::linear::LinearWeight;
use ai_engine_runtime::config::ModelConfig;
use ai_engine_runtime::loader::load_weights;
use std::path::PathBuf;

type B = burn_ndarray::NdArray;

fn safetensors_fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/toy-llama-3")
}

fn gguf_fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/toy-llama-3-gguf")
}

#[test]
fn dispatches_safetensors_path_to_load_range() {
    let cfg = ModelConfig::from_file(&safetensors_fixture().join("config.json")).unwrap();
    let dev = Default::default();
    let weights = load_weights::<B>(
        &safetensors_fixture().join("model.safetensors"), &cfg,
        0..cfg.n_layers, true, true, &dev,
    ).unwrap();
    // Toy safetensors has dense (bf16) Linears, so expect Dense after load.
    for layer in &weights.layers {
        assert!(matches!(layer.q_proj, LinearWeight::Dense(_)));
    }
}

#[test]
fn dispatches_gguf_path_to_load_gguf() {
    let cfg = ModelConfig::from_file(&gguf_fixture().join("config.json")).unwrap();
    let dev = Default::default();
    let weights = load_weights::<B>(
        &gguf_fixture().join("model.gguf"), &cfg,
        0..cfg.n_layers, true, true, &dev,
    ).unwrap();
    // Toy GGUF fixture has Q4_0 Linears.
    for layer in &weights.layers {
        assert!(matches!(layer.q_proj, LinearWeight::Q4Gguf(_)));
    }
}

#[test]
fn unknown_extension_errors_clearly() {
    let cfg = ModelConfig::from_file(&safetensors_fixture().join("config.json")).unwrap();
    let dev = Default::default();
    let err = load_weights::<B>(
        std::path::Path::new("/nonexistent/model.bin"), &cfg,
        0..1, true, true, &dev,
    ).unwrap_err().to_string();
    assert!(
        err.to_lowercase().contains("unsupported") || err.to_lowercase().contains(".bin"),
        "got error: {err}"
    );
}
```

- [ ] **Step 2: Confirm fails**

```bash
cd /home/alessio/aip/airproxy
cargo test -p ai-engine-runtime --test load_weights_dispatch 2>&1 | tail -10
# Expected: compile error — load_weights doesn't exist.
```

- [ ] **Step 3: Implement**

Append to `crates/ai-engine-runtime/src/loader.rs`:

```rust
/// Dispatch loader. Picks `load_gguf` or `load_range` based on file extension.
/// `.gguf` → load_gguf. `.safetensors` (or no extension) → load_range.
/// Other extensions error with a clear message.
pub fn load_weights<B: Backend>(
    path: &std::path::Path,
    cfg: &crate::config::ModelConfig,
    layer_range: std::ops::Range<usize>,
    hosts_embedding: bool,
    hosts_output: bool,
    device: &B::Device,
) -> anyhow::Result<LoadedWeights<B>> {
    match path.extension().and_then(|s| s.to_str()) {
        Some("gguf") => load_gguf::<B>(path, cfg, layer_range, hosts_embedding, hosts_output, device),
        Some("safetensors") | None => load_range::<B>(path, cfg, layer_range, hosts_embedding, hosts_output, device),
        Some(other) => anyhow::bail!("unsupported weights file extension `.{other}` (use `.safetensors` or `.gguf`)"),
    }
}
```

Re-export from `lib.rs`:

```rust
pub use loader::{load_gguf, load_range, load_weights, LoadedWeights};
```

- [ ] **Step 4: Verify**

```bash
cargo test -p ai-engine-runtime --test load_weights_dispatch
cargo clippy --workspace --all-targets -- -D warnings
```

3 new tests pass.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(runtime): load_weights dispatches .gguf vs .safetensors by extension"
```

NO Co-Authored-By footer (global preference).

---

### Task 2: Migrate runtime + cluster callers to `load_weights`

**Files:**
- Modify: `crates/ai-engine-runtime/src/arch/model.rs` (any direct `load_range` calls)
- Modify: `crates/ai-engine-cluster/src/leader.rs` (`build_leader_model`)
- Modify: `crates/ai-engine-cluster/src/worker.rs` (`run_worker_full`)

`Model::from_loaded` accepts a `LoadedWeights` directly — it doesn't call the loader itself. Callers of `Model::from_loaded` that previously called `load_range` are: tests + `build_leader_model` + `run_worker_full`.

For tests: leave them on `load_range` (they're testing the safetensors path specifically; some load Q8/Q4 fixtures which are still .safetensors-based). No changes.

For `build_leader_model` (in `leader.rs`): currently calls `load_range::<B>(model_path, cfg, ...)`. Change to `load_weights::<B>(model_path, cfg, ...)`.

For `run_worker_full` (in `worker.rs`): same change.

- [ ] **Step 1: Update `build_leader_model`**

Find the `load_range::<B>(...)` call inside `build_leader_model` in `crates/ai-engine-cluster/src/leader.rs`. Replace `load_range` with `load_weights`. Import `load_weights` instead of (or alongside) `load_range`.

- [ ] **Step 2: Update `run_worker_full`**

Find the `load_range::<B>(&model_path, &cfg, layer_range.clone(), false, false, &device)?` call in `crates/ai-engine-cluster/src/worker.rs`. Replace `load_range` with `load_weights`.

- [ ] **Step 3: Verify all existing tests still pass**

```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

This is a pure-substitution change — `load_weights` for a `.safetensors` path resolves to `load_range` and produces byte-identical output. ALL 205 existing tests must still pass.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(cluster): leader/worker call load_weights instead of load_range"
```

NO Co-Authored-By.

---

### Task 3: End-to-end GGUF via cluster path

**Files:**
- Create: `crates/ai-engine-cluster/tests/gguf_cluster.rs`

Prove a 3-node cluster works end-to-end loading the GGUF fixture, via the same code path the binary uses.

- [ ] **Step 1: Test**

`crates/ai-engine-cluster/tests/gguf_cluster.rs`:

```rust
//! 3-node cluster loading the toy-llama-3-gguf fixture via load_weights dispatch.
//! Just verifies the full cluster forward runs without diverging — same shape as
//! the existing q8_cluster / q4_cluster tests, but pointing at the GGUF fixture.

use std::path::PathBuf;

type B = burn_ndarray::NdArray;

fn gguf_fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .join("ai-engine-runtime/fixtures/toy-llama-3-gguf")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn gguf_cluster_generation_runs_end_to_end() {
    use ai_engine_cluster::{
        capability::BackendKind, leader::{ClusterLeader, LeaderConfig, WorkerEndpoint},
        tls::generate_node_identity, transport::quic::server_endpoint,
        worker::run_worker_full,
    };

    let fix = gguf_fixture();
    let cfg = ai_engine_runtime::config::ModelConfig::from_file(&fix.join("config.json")).unwrap();
    let tok = ai_engine_tokenizer::HfTokenizer::from_path(fix.join("tokenizer.json")).unwrap();
    let prompt = "The quick brown fox";
    let ids: Vec<u32> = ai_engine_tokenizer::Tokenizer::encode(&tok, prompt).unwrap();
    let ids_i32: Vec<i32> = ids.iter().map(|x| *x as i32).collect();

    let w1_id = generate_node_identity("w1").unwrap();
    let w1_ep = server_endpoint(&w1_id, "127.0.0.1:0".parse().unwrap()).unwrap();
    let w1_addr = w1_ep.local_addr().unwrap();
    let w2_id = generate_node_identity("w2").unwrap();
    let w2_ep = server_endpoint(&w2_id, "127.0.0.1:0".parse().unwrap()).unwrap();
    let w2_addr = w2_ep.local_addr().unwrap();

    // GGUF path: the fixture's model.gguf is the weights file.
    let model_path = fix.join("model.gguf");
    let cfg_w1 = cfg.clone();
    let mp1 = model_path.clone();
    tokio::spawn(async move {
        run_worker_full::<B>(w1_ep, "w1".into(), BackendKind::Cpu, mp1, cfg_w1).await
    });
    let cfg_w2 = cfg.clone();
    let mp2 = model_path.clone();
    tokio::spawn(async move {
        run_worker_full::<B>(w2_ep, "w2".into(), BackendKind::Cpu, mp2, cfg_w2).await
    });

    let leader_id = generate_node_identity("leader").unwrap();
    let lcfg = LeaderConfig {
        cluster_id: "gguf-test".into(),
        leader_node_id: "leader".into(),
        model_id: "toy-gguf".into(),
        n_layers: cfg.n_layers,
        layer_bytes: 256 * 1024,
        embed_output_bytes: 256 * 1024,
        per_node_overhead: 64 * 1024,
        workers: vec![
            WorkerEndpoint { node_id: "w1".into(), addr: w1_addr, fingerprint: w1_id.fingerprint.clone() },
            WorkerEndpoint { node_id: "w2".into(), addr: w2_addr, fingerprint: w2_id.fingerprint.clone() },
        ],
        partition_override: Some(vec![("w1".into(), 0..2), ("w2".into(), 2..4)]),
    };
    let leader = ClusterLeader::start(&leader_id, lcfg).await.unwrap();

    let tokens = leader.generate::<B>(
        &model_path, &cfg, 0..0, &ids_i32, 3,
        ai_engine_runtime::sample::SamplingConfig {
            temperature: 0.0, top_p: None, top_k: None, seed: 0,
        },
    ).await.unwrap();

    assert_eq!(tokens.len(), 3, "expected 3 generated tokens");
}
```

- [ ] **Step 2: Run + commit**

```bash
cargo test -p ai-engine-cluster --test gguf_cluster -- --nocapture
cargo clippy --workspace --all-targets -- -D warnings
git add -A
git commit -m "test(cluster): GGUF fixture loads and generates 3 tokens end-to-end"
```

NO Co-Authored-By.

---

### Task 4: Multi-process GGUF smoke

**Files:**
- Create: `crates/ai-engine/tests/multiproc_smoke_gguf.rs`

Same shape as the existing `multiproc_smoke.rs` but with `weights_path = "...model.gguf"` instead of `.safetensors`.

- [ ] **Step 1: Test**

Copy `multiproc_smoke.rs` and modify just the `weights_path` in `write_config` to point at the GGUF fixture:

`crates/ai-engine/tests/multiproc_smoke_gguf.rs`:

```rust
use std::io::{BufRead, BufReader};
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

const FIXTURE_PATH: &str = "../ai-engine-runtime/fixtures/toy-llama-3-gguf";

fn fixture_abspath() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(FIXTURE_PATH)
        .canonicalize()
        .expect("fixture canonicalize")
}

fn free_port() -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let p = l.local_addr().unwrap().port();
    drop(l);
    p
}

fn release_binary() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/release/ai-engine")
        .canonicalize()
        .expect("release binary — run `cargo build --release` first")
}

fn write_config(
    dir: &std::path::Path,
    fix: &std::path::Path,
    leader_http_port: u16,
    leader_quic_port: u16,
    w1_quic_port: u16,
    w2_quic_port: u16,
    leader_fp: &str,
    w1_fp: &str,
    w2_fp: &str,
) -> PathBuf {
    let toml = format!(
        r#"
[server]
bind = "127.0.0.1:{leader_http_port}"
log_format = "pretty"
log_level = "warn"

[auth]
mode = "passthrough"

[[cluster]]
id = "smoke-gguf"
leader = "leader"
quic_bind = "127.0.0.1:0"

[cluster.model]
id = "toy-llama-gguf"
config_path = "{fix}/config.json"
weights_path = "{fix}/model.gguf"
tokenizer_path = "{fix}/tokenizer.json"

[[cluster.partition_override]]
node = "worker-1"
layers = "0..2"

[[cluster.partition_override]]
node = "worker-2"
layers = "2..4"

[[cluster.node]]
id = "leader"
addr = "127.0.0.1:{leader_quic_port}"
cert_fingerprint = "{leader_fp}"
backend = "cpu"

[[cluster.node]]
id = "worker-1"
addr = "127.0.0.1:{w1_quic_port}"
cert_fingerprint = "{w1_fp}"
backend = "cpu"

[[cluster.node]]
id = "worker-2"
addr = "127.0.0.1:{w2_quic_port}"
cert_fingerprint = "{w2_fp}"
backend = "cpu"

[[provider]]
id = "smoke-gguf-cluster"
kind = "local-cluster"
cluster = "smoke-gguf"

[[route]]
match = {{ model = "toy-llama-gguf" }}
provider = "smoke-gguf-cluster"

[pipeline."/v1/chat/completions"]
stages = ["auth", "model_route", "forward", "log"]
"#,
        fix = fix.display(),
    );
    let path = dir.join("ai-engine.toml");
    std::fs::write(&path, toml).unwrap();
    path
}

fn read_fingerprint_from_stderr(child: &mut Child) -> anyhow::Result<String> {
    let stderr = child.stderr.take().unwrap();
    let reader = BufReader::new(stderr);
    for line in reader.lines() {
        let line = line?;
        eprintln!("[child] {line}");
        if let Some(idx) = line.find("fingerprint: sha256:") {
            let rest = &line[idx + "fingerprint: ".len()..];
            let fp: String = rest.split_whitespace().next().unwrap_or("").to_string();
            return Ok(fp);
        }
    }
    anyhow::bail!("fingerprint line not seen on stderr")
}

#[test]
#[ignore = "multi-process GGUF smoke; requires release build; run with --ignored"]
fn three_process_cluster_serves_chat_from_gguf() {
    let bin = release_binary();
    let fix = fixture_abspath();
    let leader_http_port = free_port();
    let leader_quic_port = free_port();
    let w1_quic_port = free_port();
    let w2_quic_port = free_port();

    let workdir = tempfile::tempdir().unwrap();
    let leader_home = workdir.path().join("leader-home");
    let w1_home = workdir.path().join("w1-home");
    let w2_home = workdir.path().join("w2-home");
    std::fs::create_dir_all(&leader_home).unwrap();
    std::fs::create_dir_all(&w1_home).unwrap();
    std::fs::create_dir_all(&w2_home).unwrap();

    let placeholder = "sha256:0000000000000000000000000000000000000000000000000000000000000000";
    let cfg_path = write_config(
        workdir.path(), &fix,
        leader_http_port, leader_quic_port, w1_quic_port, w2_quic_port,
        placeholder, placeholder, placeholder,
    );

    let mut w1 = Command::new(&bin)
        .arg("--config").arg(&cfg_path).arg("--node-id").arg("worker-1")
        .env("HOME", &w1_home).stderr(Stdio::piped()).stdout(Stdio::null())
        .spawn().unwrap();
    let mut w2 = Command::new(&bin)
        .arg("--config").arg(&cfg_path).arg("--node-id").arg("worker-2")
        .env("HOME", &w2_home).stderr(Stdio::piped()).stdout(Stdio::null())
        .spawn().unwrap();
    let w1_fp = read_fingerprint_from_stderr(&mut w1).expect("w1 fingerprint");
    let w2_fp = read_fingerprint_from_stderr(&mut w2).expect("w2 fingerprint");

    let leader_id = ai_engine_cluster::tls::load_or_generate_node_identity(
        "leader", &leader_home.join(".ai-engine"),
    ).unwrap();

    let final_cfg = write_config(
        workdir.path(), &fix,
        leader_http_port, leader_quic_port, w1_quic_port, w2_quic_port,
        &leader_id.fingerprint, &w1_fp, &w2_fp,
    );

    let mut leader = Command::new(&bin)
        .arg("--config").arg(&final_cfg).arg("--node-id").arg("leader")
        .env("HOME", &leader_home).stderr(Stdio::piped()).stdout(Stdio::null())
        .spawn().unwrap();

    let leader_url = format!("http://127.0.0.1:{leader_http_port}");
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(2)).build().unwrap();

    let mut ready = false;
    for _ in 0..100 {
        std::thread::sleep(Duration::from_millis(200));
        if let Ok(r) = client.get(format!("{leader_url}/healthz")).send() {
            if r.status().as_u16() == 200 { ready = true; break; }
        }
    }
    assert!(ready, "leader didn't become ready within 20s");

    let response = client.post(format!("{leader_url}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "toy-llama-gguf",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 3
        }))
        .send().expect("POST chat completion");
    assert_eq!(response.status().as_u16(), 200);
    let body: serde_json::Value = response.json().expect("JSON body");
    let usage = &body["usage"];
    assert_eq!(usage["completion_tokens"], 3, "expected 3 completion tokens");

    let _ = leader.kill(); let _ = w1.kill(); let _ = w2.kill();
    let _ = leader.wait(); let _ = w1.wait(); let _ = w2.wait();
}
```

- [ ] **Step 2: Run + commit**

```bash
cargo build --workspace --release
cargo test -p ai-engine --test multiproc_smoke_gguf -- --ignored --nocapture
git add -A
git commit -m "test(smoke): 3-process cluster serves chat from GGUF checkpoint"
```

NO Co-Authored-By.

---

### Task 5: README + tag v0.3.0-alpha.5

**Files:**
- Modify: `README.md`
- Tag: `v0.3.0-alpha.5`

- [ ] **Step 1: Final verification**

```bash
cd /home/alessio/aip/airproxy
cargo test --workspace 2>&1 | grep "test result" | awk '{p += $4; ig += $8} END {print "PASSED=" p " IGNORED=" ig}'
cargo clippy --workspace --all-targets -- -D warnings
cargo build --workspace --release
cargo test -p ai-engine --test multiproc_smoke -- --ignored --nocapture 2>&1 | tail -5
cargo test -p ai-engine --test streaming_smoke -- --ignored --nocapture 2>&1 | tail -5
cargo test -p ai-engine --test multiproc_smoke_mdns -- --ignored --nocapture 2>&1 | tail -5
cargo test -p ai-engine --test multiproc_smoke_gguf -- --ignored --nocapture 2>&1 | tail -5
```

- [ ] **Step 2: README**

Append to release history:

```markdown
### v0.3.0-alpha.5 — GGUF binary wiring

ai-engine v0.3.0-alpha.5 loads `.gguf` checkpoints through the binary
path. Just point `weights_path` at a GGUF file:

\`\`\`toml
[cluster.model]
id = "llama-3-70b"
config_path = "/srv/models/llama-3-70b/config.json"
weights_path = "/srv/models/llama-3-70b/model.gguf"     # <-- .gguf, not .safetensors
tokenizer_path = "/srv/models/llama-3-70b/tokenizer.json"
\`\`\`

The new `load_weights` function dispatches on file extension; everything
else (workers, leader, partitioning, generation) is unchanged.

Known limitations (still deferred):
- `config_path` + `tokenizer_path` still required even when the GGUF
  embeds them. Pulling these from GGUF metadata is a future cleanup.
- Only Q4_0 + F32 + F16 + BF16 GGUF tensor types decoded.
```

- [ ] **Step 3: Commit + tag**

```bash
git add README.md
git commit -m "docs: announce v0.3.0-alpha.5 GGUF binary wiring release"
git tag v0.3.0-alpha.5
git log --oneline -5
git tag
```

NO Co-Authored-By.

---

## Self-review

- **Spec coverage**: extension-based dispatch (Task 1) → all callers migrated (Task 2) → in-process cluster gate (Task 3) → cross-process smoke (Task 4) → release (Task 5).
- **Placeholder scan**: clean.
- **Type consistency**: `load_weights` signature mirrors `load_range` and `load_gguf` exactly — identical type parameters and return type, so caller migration is mechanical.

## Execution

Plan 9 saved to `docs/superpowers/plans/2026-05-24-plan-9-gguf-binary-wiring.md`. **Subagent-driven** is overkill for 5 tasks of this size; **inline** is the better fit.
