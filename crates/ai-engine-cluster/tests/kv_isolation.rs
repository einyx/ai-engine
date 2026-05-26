//! KV-cache isolation across requests with distinct `request_id`s.
//!
//! Per-request KV state on each worker lives in
//! `HashMap<Uuid, Vec<KvCacheSlot<B>>>`. If two requests ever shared the same
//! key (or the worker reused state across keys), the second request's first
//! generated token would diverge from its single-node baseline.
//!
//! Each call to `ClusterLeader::generate` builds a fresh `RequestSession`
//! with a new `Uuid::now_v7()`, so we run three back-to-back generations on
//! the *same* leader and the *same* worker tasks, each with a different
//! prompt, and check each one against its respective single-node greedy
//! first token. If the worker's per-request cache map is buggy, at least one
//! of the comparisons will fail.
//!
//! Sequential rather than concurrent: this test exercises the contract that
//! matters here — distinct `request_id`s do not leak state across requests
//! on the same connection. Concurrent coverage lives in
//! `tests/concurrent_requests.rs`.

use ai_engine_cluster::capability::BackendKind;
use ai_engine_cluster::leader::{ClusterLeader, LeaderConfig, WorkerEndpoint};
use ai_engine_cluster::tls::generate_node_identity;
use ai_engine_cluster::transport::quic::server_endpoint;
use ai_engine_cluster::worker::run_worker_full;
use ai_engine_runtime::arch::model::Model;
use ai_engine_runtime::config::ModelConfig;
use ai_engine_runtime::loader::load_range;
use ai_engine_runtime::sample::{sample, SamplingConfig};
use ai_engine_tokenizer::{HfTokenizer, Tokenizer};
use burn::tensor::{Int, Tensor, TensorData};
use std::path::{Path, PathBuf};

type B = burn_ndarray::NdArray;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("ai-engine-runtime/fixtures/toy-llama-3")
}

fn single_node_first_token(model_path: &Path, cfg: &ModelConfig, ids_i32: &[i32]) -> u32 {
    let dev = Default::default();
    let weights = load_range::<B>(model_path, cfg, 0..cfg.n_layers, true, true, &dev).unwrap();
    let model = Model::<B>::from_loaded(cfg, weights, &dev).unwrap();
    let seq = ids_i32.len();
    let logits: Vec<f32> = model
        .forward(
            Tensor::<B, 2, Int>::from_data(TensorData::new(ids_i32.to_vec(), [1, seq]), &dev),
            0,
        )
        .slice([0..1, (seq - 1)..seq, 0..cfg.vocab_size])
        .reshape([cfg.vocab_size])
        .to_data()
        .to_vec()
        .unwrap();
    sample(
        &logits,
        &SamplingConfig {
            temperature: 0.0,
            top_p: None,
            top_k: None,
            seed: 0,
        },
    )
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn distinct_request_ids_do_not_leak_kv_state() {
    let fix = fixture();
    let cfg = ModelConfig::from_file(&fix.join("config.json")).unwrap();
    let tok = HfTokenizer::from_path(fix.join("tokenizer.json")).unwrap();
    let model_path = fix.join("model.safetensors");

    let prompts = [
        "The quick brown fox",
        "Hello world from the cluster",
        "Distributed inference across nodes",
    ];

    // Compute single-node baselines (first greedy token) for each prompt.
    let baselines: Vec<(Vec<i32>, u32)> = prompts
        .iter()
        .map(|p| {
            let ids: Vec<u32> = tok.encode(p).unwrap();
            let ids_i32: Vec<i32> = ids.iter().map(|x| *x as i32).collect();
            let first = single_node_first_token(&model_path, &cfg, &ids_i32);
            (ids_i32, first)
        })
        .collect();

    // 3-node cluster identical to Task 10.
    let w1_id = generate_node_identity("w1").unwrap();
    let w1_ep = server_endpoint(&w1_id, "127.0.0.1:0".parse().unwrap()).unwrap();
    let w1_addr = w1_ep.local_addr().unwrap();
    let w2_id = generate_node_identity("w2").unwrap();
    let w2_ep = server_endpoint(&w2_id, "127.0.0.1:0".parse().unwrap()).unwrap();
    let w2_addr = w2_ep.local_addr().unwrap();

    let cfg_for_w1 = cfg.clone();
    let mp1 = model_path.clone();
    let _w1_task = tokio::spawn(async move {
        run_worker_full::<B>(
            w1_ep,
            "w1".to_string(),
            BackendKind::Cpu,
            mp1,
            cfg_for_w1,
        )
        .await
    });
    let cfg_for_w2 = cfg.clone();
    let mp2 = model_path.clone();
    let _w2_task = tokio::spawn(async move {
        run_worker_full::<B>(
            w2_ep,
            "w2".to_string(),
            BackendKind::Cpu,
            mp2,
            cfg_for_w2,
        )
        .await
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
            WorkerEndpoint {
                node_id: "w1".into(),
                addr: w1_addr,
                fingerprint: w1_id.fingerprint.clone(),
            },
            WorkerEndpoint {
                node_id: "w2".into(),
                addr: w2_addr,
                fingerprint: w2_id.fingerprint.clone(),
            },
        ],
        // Leader hosts no layers (0..0); workers cover all 4.
        partition_override: Some(vec![("w1".into(), 0..2), ("w2".into(), 2..4)]),
    };

    let leader = ClusterLeader::start(&leader_id, lcfg).await.unwrap();

    // Run all three forwards through the *same* leader against the *same*
    // worker tasks. Each call gets a distinct request_id via Uuid::now_v7()
    // inside RequestSession::new.
    for (i, (ids_i32, baseline_token)) in baselines.iter().enumerate() {
        let cluster_tokens = leader
            .generate::<B>(
                &model_path,
                &cfg,
                0..0,
                ids_i32,
                1,
                SamplingConfig {
                    temperature: 0.0,
                    top_p: None,
                    top_k: None,
                    seed: 0,
                },
            )
            .await
            .unwrap();
        assert_eq!(cluster_tokens.len(), 1);
        eprintln!(
            "request {i}: cluster first token = {} (baseline = {})",
            cluster_tokens[0], baseline_token
        );
        assert_eq!(
            cluster_tokens[0], *baseline_token,
            "request {i}: cluster first token must match single-node greedy baseline"
        );
    }
}
