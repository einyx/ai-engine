//! Multi-process any-node ingress smoke test for the leaderless p2p mesh
//! (Phase B). Spawns a 3-node leaderless cluster as separate OS processes,
//! each forming a full QUIC mesh and exposing its own HTTP server. A chat
//! completion is POSTed to a node that is NOT the embedding host, proving any
//! node can ingest a request and coordinate inference over the whole mesh.
//!
//! Marked `#[ignore]` — run with
//! `cargo test -p ai-engine --test multiproc_smoke_anynode -- --ignored --nocapture`
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

struct NodeSpec {
    id: &'static str,
    http_port: u16,
    quic_port: u16,
    layers: &'static str,
    fingerprint: String,
    home: PathBuf,
}

/// Write the per-node config: every node shares the same `[[cluster]]`,
/// `[[cluster.node]]`, partition_override and provider/route wiring, but each
/// gets its own `[server] bind` (HTTP) so the processes don't collide.
fn write_config(path: &Path, fix: &Path, nodes: &[NodeSpec], my_http_port: u16) {
    let mut node_blocks = String::new();
    for n in nodes {
        node_blocks.push_str(&format!(
            r#"
[[cluster.node]]
id = "{id}"
addr = "127.0.0.1:{quic}"
cert_fingerprint = "{fp}"
backend = "cpu"
"#,
            id = n.id,
            quic = n.quic_port,
            fp = n.fingerprint,
        ));
    }
    let mut partition_blocks = String::new();
    for n in nodes {
        partition_blocks.push_str(&format!(
            r#"
[[cluster.partition_override]]
node = "{id}"
layers = "{layers}"
"#,
            id = n.id,
            layers = n.layers,
        ));
    }

    // quic_bind is a legacy/star field still required by the Cluster struct;
    // the leaderless path binds each node to its own `addr` instead, so the
    // value here is unused but must parse.
    let toml = format!(
        r#"
[server]
bind = "127.0.0.1:{my_http_port}"
log_format = "pretty"
log_level = "info"

[auth]
mode = "passthrough"

[[cluster]]
id = "smoke-anynode"
leader = "{leader}"
quic_bind = "127.0.0.1:{leader_quic}"
leaderless = true

[cluster.model]
id = "toy-llama-gguf"
config_path = "{fix_cfg}"
weights_path = "{fix_wts}"
tokenizer_path = "{fix_tok}"
{node_blocks}{partition_blocks}
[[provider]]
id = "smoke-anynode-cluster"
kind = "local-cluster"
cluster = "smoke-anynode"

[[route]]
match = {{ model = "toy-llama-gguf" }}
provider = "smoke-anynode-cluster"

[pipeline."/v1/chat/completions"]
stages = ["auth", "model_route", "forward", "log"]
"#,
        leader = nodes[0].id,
        leader_quic = nodes[0].quic_port,
        fix_cfg = fix.join("config.json").display(),
        fix_wts = fix.join("model.gguf").display(),
        fix_tok = fix.join("tokenizer.json").display(),
    );
    std::fs::write(path, toml).unwrap();
}

fn pump_stderr(child_stderr: std::process::ChildStderr, prefix: String) {
    std::thread::spawn(move || {
        let reader = BufReader::new(child_stderr);
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
#[ignore = "multi-process leaderless smoke; requires release build; run with --ignored"]
fn any_node_ingress_serves_chat() {
    let start = Instant::now();

    let bin = release_binary();
    let fix = fixture_abspath();
    let workdir = tempfile::tempdir().unwrap();

    // 3-node pipeline over the 4-layer toy model: head (embedding, 0..2),
    // middle (2..3), tail (output, 3..4). Pre-generate each node's identity in
    // its own HOME so we know all fingerprints before any process boots — this
    // lets every node form the mesh on first try (no fingerprint discovery
    // round-trip needed).
    let mut nodes: Vec<NodeSpec> = ["node-a", "node-b", "node-c"]
        .iter()
        .zip(["0..2", "2..3", "3..4"])
        .map(|(&id, layers)| {
            let home = workdir.path().join(format!("{id}-home"));
            std::fs::create_dir_all(&home).unwrap();
            let identity = ai_engine_cluster::tls::load_or_generate_node_identity(
                id,
                &home.join(".ai-engine"),
            )
            .expect("node identity");
            eprintln!("[test] {id} fp = {}", identity.fingerprint);
            NodeSpec {
                id,
                http_port: free_port(),
                quic_port: free_port(),
                layers,
                fingerprint: identity.fingerprint,
                home,
            }
        })
        .collect();
    nodes.sort_by_key(|n| n.id);

    // Spawn every node. Each gets its own config (its own HTTP bind) but the
    // same cluster topology.
    let mut guards: Vec<KillOnDrop> = Vec::new();
    for n in &nodes {
        let cfg_path = workdir.path().join(format!("{}.toml", n.id));
        write_config(&cfg_path, &fix, &nodes, n.http_port);
        let mut child = Command::new(&bin)
            .arg("--config")
            .arg(&cfg_path)
            .arg("--node-id")
            .arg(n.id)
            .env("HOME", &n.home)
            .stderr(Stdio::piped())
            .stdout(Stdio::null())
            .spawn()
            .unwrap_or_else(|e| panic!("spawn {}: {e}", n.id));
        if let Some(s) = child.stderr.take() {
            pump_stderr(s, n.id.to_string());
        }
        guards.push(KillOnDrop(child));
    }

    // Ingest from `node-b` (the middle node) — NOT the embedding host (node-a).
    let ingress = &nodes[1];
    assert_ne!(
        ingress.layers, "0..2",
        "ingress node must not be the embedding host"
    );
    let ingress_url = format!("http://127.0.0.1:{}", ingress.http_port);
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .unwrap();

    let mut ready = false;
    for _ in 0..150 {
        std::thread::sleep(Duration::from_millis(200));
        if let Ok(r) = client.get(format!("{ingress_url}/healthz")).send() {
            if r.status().as_u16() == 200 {
                ready = true;
                break;
            }
        }
    }
    assert!(ready, "ingress node didn't become ready within 30s");
    eprintln!(
        "[test] ingress ({}) ready after {:.2}s",
        ingress.id,
        start.elapsed().as_secs_f64()
    );

    let response = client
        .post(format!("{ingress_url}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "toy-llama-gguf",
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
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    assert!(
        !content.is_empty(),
        "expected non-empty choices[0].message.content, got body: {body_text}"
    );

    eprintln!("[test] success after {:.2}s", start.elapsed().as_secs_f64());
}
