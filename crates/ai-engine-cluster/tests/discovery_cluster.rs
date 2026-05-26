//! 3-node cluster started entirely via mDNS — no fingerprints in any test config.
//!
//! Each "worker" is a tokio task that:
//!   1. Generates its identity (cert + fingerprint).
//!   2. Announces itself via mDNS with the fingerprint as a TXT record.
//!   3. Starts its QUIC server endpoint at a random port.
//!   4. Runs `run_worker_full` (waits for Assignment, etc.).
//!
//! The leader:
//!   1. Calls `discover_workers` to find both workers.
//!   2. Builds a `LeaderConfig` from the discovered endpoints.
//!   3. Calls `ClusterLeader::start(...)` as usual — the rest is unchanged.

use std::net::IpAddr;
use std::path::PathBuf;
use std::time::Duration;

type B = burn_ndarray::NdArray;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("ai-engine-runtime/fixtures/toy-llama-3")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn mdns_cluster_generates_chat_completion() {
    let cluster_id = "mdns-test-cluster";

    let fix = fixture();
    let cfg = ai_engine_runtime::config::ModelConfig::from_file(&fix.join("config.json")).unwrap();
    let tok = ai_engine_tokenizer::HfTokenizer::from_path(fix.join("tokenizer.json")).unwrap();

    // --- Worker 1 ---
    let w1_id = ai_engine_cluster::tls::generate_node_identity("w1").unwrap();
    let w1_ep = ai_engine_cluster::transport::quic::server_endpoint(
        &w1_id,
        "127.0.0.1:0".parse().unwrap(),
    )
    .unwrap();
    let w1_port = w1_ep.local_addr().unwrap().port();
    let w1_txt = ai_engine_cluster::discovery::TxtRecords {
        cluster_id: cluster_id.into(),
        node_id: "w1".into(),
        role: "worker".into(),
        protocol_version: 1,
        fingerprint: w1_id.fingerprint.clone(),
        backend: "cpu".into(),
    };
    let _w1_ann = ai_engine_cluster::discovery::Announcer::register(
        IpAddr::V4("127.0.0.1".parse().unwrap()),
        w1_port,
        "w1.local.",
        w1_txt,
    )
    .unwrap();
    let model_path = fix.join("model.safetensors");
    let cfg_w1 = cfg.clone();
    let mp1 = model_path.clone();
    tokio::spawn(async move {
        ai_engine_cluster::worker::run_worker_full::<B>(
            w1_ep,
            "w1".into(),
            ai_engine_cluster::capability::BackendKind::Cpu,
            mp1,
            cfg_w1,
        )
        .await
    });

    // --- Worker 2 ---
    let w2_id = ai_engine_cluster::tls::generate_node_identity("w2").unwrap();
    let w2_ep = ai_engine_cluster::transport::quic::server_endpoint(
        &w2_id,
        "127.0.0.1:0".parse().unwrap(),
    )
    .unwrap();
    let w2_port = w2_ep.local_addr().unwrap().port();
    let w2_txt = ai_engine_cluster::discovery::TxtRecords {
        cluster_id: cluster_id.into(),
        node_id: "w2".into(),
        role: "worker".into(),
        protocol_version: 1,
        fingerprint: w2_id.fingerprint.clone(),
        backend: "cpu".into(),
    };
    let _w2_ann = ai_engine_cluster::discovery::Announcer::register(
        IpAddr::V4("127.0.0.1".parse().unwrap()),
        w2_port,
        "w2.local.",
        w2_txt,
    )
    .unwrap();
    let cfg_w2 = cfg.clone();
    let mp2 = model_path.clone();
    tokio::spawn(async move {
        ai_engine_cluster::worker::run_worker_full::<B>(
            w2_ep,
            "w2".into(),
            ai_engine_cluster::capability::BackendKind::Cpu,
            mp2,
            cfg_w2,
        )
        .await
    });

    // Let mDNS propagate.
    tokio::time::sleep(Duration::from_millis(500)).await;

    // --- Leader: discover, then start ---
    let discovered = ai_engine_cluster::discovery::discover_workers(
        cluster_id,
        2,
        Duration::from_secs(10),
    )
    .await
    .unwrap();
    assert_eq!(
        discovered.len(),
        2,
        "expected 2 discovered workers, got {}",
        discovered.len()
    );

    let leader_id = ai_engine_cluster::tls::generate_node_identity("leader").unwrap();
    let lcfg = ai_engine_cluster::leader::LeaderConfig::from_discovered(
        cluster_id,
        "leader",
        "toy-mdns",
        cfg.n_layers,
        discovered,
        Some(vec![("w1".into(), 0..2), ("w2".into(), 2..4)]),
    );
    let leader = ai_engine_cluster::leader::ClusterLeader::start(&leader_id, lcfg)
        .await
        .unwrap();

    let prompt = "The quick brown fox";
    let ids: Vec<u32> = ai_engine_tokenizer::Tokenizer::encode(&tok, prompt).unwrap();
    let ids_i32: Vec<i32> = ids.iter().map(|x| *x as i32).collect();
    let tokens = leader
        .generate::<B>(
            &model_path,
            &cfg,
            0..0,
            &ids_i32,
            3,
            ai_engine_runtime::sample::SamplingConfig {
                temperature: 0.0,
                top_p: None,
                top_k: None,
                seed: 0,
            },
        )
        .await
        .unwrap();

    assert_eq!(tokens.len(), 3, "expected 3 generated tokens");
}
