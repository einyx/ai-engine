//! Multi-process smoke test for GGUF minimal-config wiring (v0.3.0-alpha.6).
//!
//! Same shape as `multiproc_smoke_gguf.rs`, but the cluster's `[cluster.model]`
//! block omits `config_path` and `tokenizer_path` entirely — only `weights_path`
//! is set. Proves that `build_app_state` and `worker_main` derive both
//! ModelConfig and the tokenizer from the GGUF metadata end-to-end through
//! 3 OS processes.
//!
//! Marked `#[ignore]` — run with
//! `cargo test -p ai-engine --test multiproc_smoke_gguf_only -- --ignored --nocapture`
//! after `cargo build --release`.

use std::io::{BufRead, BufReader};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

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
        .expect("release binary missing — run `cargo build --release` first")
}

#[allow(clippy::too_many_arguments)]
fn write_config(
    dir: &Path,
    fix: &Path,
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
id = "smoke-gguf-only"
leader = "leader"
quic_bind = "127.0.0.1:0"

[cluster.model]
id = "toy-llama-gguf-only"
weights_path = "{fix}/model.gguf"

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
id = "smoke-gguf-only-cluster"
kind = "local-cluster"
cluster = "smoke-gguf-only"

[[route]]
match = {{ model = "toy-llama-gguf-only" }}
provider = "smoke-gguf-only-cluster"

[pipeline."/v1/chat/completions"]
stages = ["auth", "model_route", "forward", "log"]
"#,
        fix = fix.display(),
    );
    let path = dir.join("ai-engine.toml");
    std::fs::write(&path, toml).unwrap();
    path
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
#[ignore = "multi-process GGUF-only smoke; requires release build; run with --ignored"]
fn three_process_cluster_with_minimal_gguf_only_config() {
    let start = Instant::now();

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

    let placeholder =
        "sha256:0000000000000000000000000000000000000000000000000000000000000000";
    let cfg_path = write_config(
        workdir.path(),
        &fix,
        leader_http_port,
        leader_quic_port,
        w1_quic_port,
        w2_quic_port,
        placeholder,
        placeholder,
        placeholder,
    );

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

    let leader_id = ai_engine_cluster::tls::load_or_generate_node_identity(
        "leader",
        &leader_home.join(".ai-engine"),
    )
    .expect("leader identity");
    eprintln!("[test] leader fp = {}", leader_id.fingerprint);

    let final_cfg = write_config(
        workdir.path(),
        &fix,
        leader_http_port,
        leader_quic_port,
        w1_quic_port,
        w2_quic_port,
        &leader_id.fingerprint,
        &w1_fp,
        &w2_fp,
    );

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
        .timeout(Duration::from_secs(30))
        .build()
        .unwrap();

    let mut ready = false;
    for _ in 0..100 {
        std::thread::sleep(Duration::from_millis(200));
        if let Ok(r) = client.get(format!("{leader_url}/healthz")).send() {
            if r.status().as_u16() == 200 {
                ready = true;
                break;
            }
        }
    }
    assert!(ready, "leader didn't become ready within 20s");
    eprintln!(
        "[test] leader ready after {:.2}s",
        start.elapsed().as_secs_f64()
    );

    let response = client
        .post(format!("{leader_url}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "toy-llama-gguf-only",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 3
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
    assert_eq!(
        completion_tokens, 3,
        "expected 3 completion tokens, got body: {body_text}"
    );

    eprintln!(
        "[test] success after {:.2}s",
        start.elapsed().as_secs_f64()
    );
}
