//! Env-gated 3-process smoke test against a real Llama-3 GGUF (Plan 11, Task 3).
//!
//! Requires:
//!   - `cargo build --release` to be run first
//!   - `AI_ENGINE_REAL_GGUF=/path/to/model.gguf` env var pointing at a real GGUF
//!
//! Run with:
//!   AI_ENGINE_REAL_GGUF=/tmp/ai-engine-validation/model.gguf \
//!     cargo test -p ai-engine --test real_model_smoke -- --ignored --nocapture
//!
//! Without the env var the test prints SKIP and returns 0.

use std::io::{BufRead, BufReader};
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use ai_engine_provider::openai::{ChatContent, ChatResponse};

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
        .expect("release binary missing — run `cargo build --release` first")
}

fn read_fingerprint_from_stderr(child: &mut Child, prefix: &str) -> anyhow::Result<String> {
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow::anyhow!("child stderr already taken"))?;
    let reader = BufReader::new(stderr);
    for line in reader.lines() {
        let line = line?;
        eprintln!("[{prefix}] {line}");
        if let Some(idx) = line.find("fingerprint: sha256:") {
            let rest = &line[idx + "fingerprint: ".len()..];
            let fp = rest
                .split_whitespace()
                .next()
                .ok_or_else(|| anyhow::anyhow!("malformed fingerprint line"))?
                .to_string();
            return Ok(fp);
        }
    }
    anyhow::bail!("fingerprint line not seen on stderr for `{prefix}`")
}

fn pump_stderr(mut child_stderr: std::process::ChildStderr, prefix: String) {
    std::thread::spawn(move || {
        let reader = BufReader::new(&mut child_stderr);
        for line in reader.lines().map_while(Result::ok) {
            eprintln!("[{prefix}] {line}");
        }
    });
}

struct KillOnDrop(Child);

impl Drop for KillOnDrop {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
#[ignore = "opt-in real-model smoke; requires release build + AI_ENGINE_REAL_GGUF env var; run with --ignored"]
fn three_process_cluster_with_real_llama3_gguf() {
    // --- env-var gate ---
    let real_gguf = match std::env::var("AI_ENGINE_REAL_GGUF").ok() {
        Some(p) => {
            let pb = PathBuf::from(&p);
            if !pb.exists() {
                eprintln!(
                    "[SKIP] AI_ENGINE_REAL_GGUF={p} does not exist — skipping real-model smoke"
                );
                return;
            }
            pb
        }
        None => {
            eprintln!(
                "[SKIP] AI_ENGINE_REAL_GGUF not set — skipping real-model smoke (set to path of a Llama-3 GGUF to enable)"
            );
            return;
        }
    };

    let start = Instant::now();

    // Read n_layers from the GGUF so we don't hard-code a layer count.
    let cfg = ai_engine_runtime::config::ModelConfig::from_gguf_file(&real_gguf)
        .expect("read ModelConfig from GGUF");
    let nl = cfg.n_layers;
    let mid = nl / 2;
    eprintln!("[test] GGUF n_layers={nl}, partition: 0..{mid} / {mid}..{nl}");

    let bin = release_binary();
    let leader_http_port = free_port();
    let leader_quic_port = free_port();
    let w1_quic_port = free_port();
    let w2_quic_port = free_port();

    // Use /tmp so we don't pollute the repo.
    let workdir = tempfile::Builder::new()
        .prefix("real-model-smoke-")
        .tempdir_in("/tmp")
        .unwrap();
    let leader_home = workdir.path().join("leader-home");
    let w1_home = workdir.path().join("w1-home");
    let w2_home = workdir.path().join("w2-home");
    std::fs::create_dir_all(&leader_home).unwrap();
    std::fs::create_dir_all(&w1_home).unwrap();
    std::fs::create_dir_all(&w2_home).unwrap();

    // Helper: write the cluster TOML.
    let write_config = |leader_fp: &str, w1_fp: &str, w2_fp: &str| -> PathBuf {
        let toml = format!(
            r#"
[server]
bind = "127.0.0.1:{leader_http_port}"
log_format = "pretty"
log_level = "warn"

[auth]
mode = "passthrough"

[[cluster]]
id = "real-llama3-smoke"
leader = "leader"
quic_bind = "127.0.0.1:0"

[cluster.model]
id = "real-llama3"
weights_path = "{weights}"

[[cluster.partition_override]]
node = "worker-1"
layers = "0..{mid}"

[[cluster.partition_override]]
node = "worker-2"
layers = "{mid}..{nl}"

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
id = "real-llama3-smoke-cluster"
kind = "local-cluster"
cluster = "real-llama3-smoke"

[[route]]
match = {{ model = "real-llama3" }}
provider = "real-llama3-smoke-cluster"

[pipeline."/v1/chat/completions"]
stages = ["auth", "model_route", "forward", "log"]
"#,
            weights = real_gguf.display(),
        );
        let path = workdir.path().join("ai-engine.toml");
        std::fs::write(&path, toml).unwrap();
        path
    };

    let placeholder =
        "sha256:0000000000000000000000000000000000000000000000000000000000000000";
    let cfg_path = write_config(placeholder, placeholder, placeholder);

    // Spawn workers first so we can capture their fingerprints.
    let mut w1 = Command::new(&bin)
        .arg("--config")
        .arg(&cfg_path)
        .arg("--node-id")
        .arg("worker-1")
        .env("HOME", &w1_home)
        .stderr(Stdio::piped())
        .stdout(Stdio::null())
        .spawn()
        .expect("spawn worker-1");
    let mut w2 = Command::new(&bin)
        .arg("--config")
        .arg(&cfg_path)
        .arg("--node-id")
        .arg("worker-2")
        .env("HOME", &w2_home)
        .stderr(Stdio::piped())
        .stdout(Stdio::null())
        .spawn()
        .expect("spawn worker-2");

    let w1_fp = read_fingerprint_from_stderr(&mut w1, "w1").expect("w1 fingerprint");
    let w2_fp = read_fingerprint_from_stderr(&mut w2, "w2").expect("w2 fingerprint");
    eprintln!("[test] w1 fp = {w1_fp}");
    eprintln!("[test] w2 fp = {w2_fp}");

    if let Some(s) = w1.stderr.take() {
        pump_stderr(s, "w1".into());
    }
    if let Some(s) = w2.stderr.take() {
        pump_stderr(s, "w2".into());
    }

    let _w1_guard = KillOnDrop(w1);
    let _w2_guard = KillOnDrop(w2);

    // Generate / load leader identity so we know its fingerprint.
    let leader_id = ai_engine_cluster::tls::load_or_generate_node_identity(
        "leader",
        &leader_home.join(".ai-engine"),
    )
    .expect("leader identity");
    eprintln!("[test] leader fp = {}", leader_id.fingerprint);

    let final_cfg = write_config(&leader_id.fingerprint, &w1_fp, &w2_fp);

    let mut leader = Command::new(&bin)
        .arg("--config")
        .arg(&final_cfg)
        .arg("--node-id")
        .arg("leader")
        .env("HOME", &leader_home)
        .stderr(Stdio::piped())
        .stdout(Stdio::null())
        .spawn()
        .expect("spawn leader");
    if let Some(s) = leader.stderr.take() {
        pump_stderr(s, "leader".into());
    }
    let _leader_guard = KillOnDrop(leader);

    let leader_url = format!("http://127.0.0.1:{leader_http_port}");
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(300))
        .build()
        .unwrap();

    // Wait for leader /healthz — generous timeout for real-model weight loading.
    let mut ready = false;
    for _ in 0..600 {
        std::thread::sleep(Duration::from_millis(500));
        if let Ok(r) = client.get(format!("{leader_url}/healthz")).send() {
            if r.status().as_u16() == 200 {
                ready = true;
                break;
            }
        }
    }
    assert!(ready, "leader didn't become ready within 5 minutes");
    eprintln!(
        "[test] leader ready after {:.2}s",
        start.elapsed().as_secs_f64()
    );

    let response = client
        .post(format!("{leader_url}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "real-llama3",
            "messages": [{"role": "user", "content": "Hello, who are you?"}],
            "temperature": 0.0,
            "max_tokens": 40
        }))
        .send()
        .expect("POST chat completion");
    let status = response.status();
    let body_text = response.text().expect("response body text");
    eprintln!("[test] response status = {status}");
    eprintln!("[test] response body = {body_text}");
    assert_eq!(status.as_u16(), 200, "non-200 response: {body_text}");

    let body: serde_json::Value =
        serde_json::from_str(&body_text).expect("response is valid JSON");
    let completion_tokens = body["usage"]["completion_tokens"].as_u64().unwrap_or(0);
    assert!(
        completion_tokens >= 1,
        "expected at least 1 completion token, got body: {body_text}"
    );

    // Deserialize into the typed ChatResponse to inspect message content.
    let typed: ChatResponse =
        serde_json::from_str(&body_text).expect("deserialize ChatResponse");
    let choice = typed.choices.first().expect("at least one choice");
    match &choice.message.content {
        ChatContent::Text(s) => {
            assert!(!s.trim().is_empty(), "response text is empty");
            eprintln!("CHAT RESPONSE: {s}");
        }
        other => panic!("expected ChatContent::Text, got {other:?}"),
    }

    eprintln!(
        "[test] success after {:.2}s",
        start.elapsed().as_secs_f64()
    );
}
