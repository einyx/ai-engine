//! Multi-process mDNS-discovery smoke test for v0.3.0-alpha.4.
//!
//! Spawns 3 `ai-engine` processes (1 leader + 2 workers) on localhost. The
//! workers announce themselves via mDNS; the leader's config has
//! `[[cluster.discover]]` set so it browses for them instead of using the
//! static `cert_fingerprint` fields. The static `[[cluster.node]]` entries
//! still appear in the config (the schema validator requires them) with
//! placeholder zero fingerprints — at runtime the leader ignores those.
//!
//! Marked `#[ignore]` because mDNS depends on LAN multicast which is not
//! universally available on CI runners. Run with:
//!   cargo test -p ai-engine --test multiproc_smoke_mdns -- --ignored --nocapture
//! after `cargo build --workspace --release`.

use std::io::{BufRead, BufReader};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

fn fixture_abspath() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../ai-engine-runtime/fixtures/toy-llama-3")
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
) -> PathBuf {
    // Placeholder zero fingerprints satisfy the schema validator. They are
    // unused at runtime because [[cluster.discover]] is set and the leader
    // builds WorkerEndpoints from mDNS announcements instead.
    let placeholder =
        "sha256:0000000000000000000000000000000000000000000000000000000000000000";
    let toml = format!(
        r#"
[server]
bind = "127.0.0.1:{leader_http_port}"
log_format = "pretty"
log_level = "warn"

[auth]
mode = "passthrough"

[[cluster]]
id = "smoke-mdns"
leader = "leader"
quic_bind = "127.0.0.1:{leader_quic_port}"

[cluster.model]
id = "toy-llama"
config_path = "{fix_cfg}"
weights_path = "{fix_wts}"
tokenizer_path = "{fix_tok}"

[cluster.discover]
expected_workers = 2
timeout_secs = 15

[[cluster.partition_override]]
node = "worker-1"
layers = "0..2"

[[cluster.partition_override]]
node = "worker-2"
layers = "2..4"

[[cluster.node]]
id = "leader"
addr = "127.0.0.1:{leader_quic_port}"
cert_fingerprint = "{placeholder}"
backend = "cpu"

[[cluster.node]]
id = "worker-1"
addr = "127.0.0.1:{w1_quic_port}"
cert_fingerprint = "{placeholder}"
backend = "cpu"

[[cluster.node]]
id = "worker-2"
addr = "127.0.0.1:{w2_quic_port}"
cert_fingerprint = "{placeholder}"
backend = "cpu"

[[provider]]
id = "smoke-cluster"
kind = "local-cluster"
cluster = "smoke-mdns"

[[route]]
match = {{ model = "toy-llama" }}
provider = "smoke-cluster"

[pipeline."/v1/chat/completions"]
stages = ["auth", "model_route", "forward", "log"]
"#,
        fix_cfg = fix.join("config.json").display(),
        fix_wts = fix.join("model.safetensors").display(),
        fix_tok = fix.join("tokenizer.json").display(),
    );
    let path = dir.join("ai-engine.toml");
    std::fs::write(&path, toml).unwrap();
    path
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
#[ignore = "multi-process mDNS smoke; requires release build + LAN multicast; run with --ignored"]
fn three_process_cluster_with_mdns_discovery_serves_chat() {
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

    let cfg_path = write_config(
        workdir.path(),
        &fix,
        leader_http_port,
        leader_quic_port,
        w1_quic_port,
        w2_quic_port,
    );

    // Spawn workers — they announce on mDNS as part of their startup.
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
    if let Some(s) = w1.stderr.take() {
        pump_stderr(s, "w1".into());
    }
    if let Some(s) = w2.stderr.take() {
        pump_stderr(s, "w2".into());
    }
    let _w1_guard = KillOnDrop(w1);
    let _w2_guard = KillOnDrop(w2);

    // Give the workers time to bind their QUIC listeners and announce on mDNS.
    std::thread::sleep(Duration::from_secs(2));

    // Spawn leader — it discovers workers via mDNS, ignoring the placeholder
    // fingerprints in [[cluster.node]] because [[cluster.discover]] is set.
    let mut leader = Command::new(&bin)
        .arg("--config")
        .arg(&cfg_path)
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
    for _ in 0..150 {
        std::thread::sleep(Duration::from_millis(200));
        if let Ok(r) = client.get(format!("{leader_url}/healthz")).send() {
            if r.status().as_u16() == 200 {
                ready = true;
                break;
            }
        }
    }
    assert!(ready, "leader didn't become ready within 30s");
    eprintln!(
        "[test] leader ready after {:.2}s",
        start.elapsed().as_secs_f64()
    );

    let response = client
        .post(format!("{leader_url}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "toy-llama",
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
        "expected exactly 3 completion tokens, got body: {body_text}"
    );

    eprintln!(
        "[test] success after {:.2}s",
        start.elapsed().as_secs_f64()
    );
}
