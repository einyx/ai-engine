//! Concurrent requests through one cluster must produce isolated outputs.
//!
//! Plan 4 Task 5 made two changes that together unlock concurrency:
//! 1. `ClusterLeader::generate` is now `&self`, so multiple tasks can drive
//!    the same leader at once.
//! 2. Activation exchanges go over bidi streams instead of paired uni
//!    streams, so concurrent requests' streams don't interleave on the
//!    receiver side.
//!
//! This test runs three distinct prompts through one 3-node cluster
//! concurrently and asserts each generated token sequence matches the
//! single-node greedy baseline for its prompt.

use ai_engine_cluster::capability::BackendKind;
use ai_engine_cluster::leader::{ClusterLeader, LeaderConfig, WorkerEndpoint};
use ai_engine_cluster::tls::generate_node_identity;
use ai_engine_cluster::transport::quic::server_endpoint;
use ai_engine_cluster::worker::run_worker_full;
use ai_engine_runtime::arch::model::Model;
use ai_engine_runtime::config::ModelConfig;
use ai_engine_runtime::kv_cache::KvCacheSlot;
use ai_engine_runtime::loader::load_range;
use ai_engine_runtime::sample::{sample, SamplingConfig};
use ai_engine_tokenizer::{HfTokenizer, Tokenizer};
use burn::tensor::{Int, Tensor, TensorData};
use std::path::{Path, PathBuf};
use std::sync::Arc;

type B = burn_ndarray::NdArray;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("ai-engine-runtime/fixtures/toy-llama-3")
}

/// Same greedy-5 reference path as the other end-to-end tests, duplicated
/// here because Rust integration tests can't share a `mod common`.
fn single_node_greedy_5(fix: &Path, cfg: &ModelConfig, prompt_ids: &[i32]) -> Vec<u32> {
    let dev = Default::default();
    let weights = load_range::<B>(
        &fix.join("model.safetensors"),
        cfg,
        0..cfg.n_layers,
        true,
        true,
        &dev,
    )
    .unwrap();
    let model = Model::<B>::from_loaded(cfg, weights, &dev).unwrap();

    let mut caches: Vec<KvCacheSlot<B>> = (0..cfg.n_layers)
        .map(|_| {
            KvCacheSlot::<B>::new(
                1,
                cfg.n_kv_heads,
                cfg.max_position_embeddings,
                cfg.head_dim,
                &dev,
            )
        })
        .collect();

    let prompt = Tensor::<B, 2, Int>::from_data(
        TensorData::new(prompt_ids.to_vec(), [1, prompt_ids.len()]),
        &dev,
    );
    let logits = model.forward_with_caches(prompt, 0, &mut caches);
    let last: Vec<f32> = logits
        .slice([
            0..1,
            (prompt_ids.len() - 1)..prompt_ids.len(),
            0..cfg.vocab_size,
        ])
        .reshape([cfg.vocab_size])
        .to_data()
        .to_vec()
        .unwrap();
    let scfg = SamplingConfig {
        temperature: 0.0,
        top_p: None,
        top_k: None,
        seed: 0,
    };
    let mut tokens = vec![sample(&last, &scfg)];
    for i in 1..5 {
        let next = Tensor::<B, 2, Int>::from_data(
            TensorData::new(vec![*tokens.last().unwrap() as i32], [1, 1]),
            &dev,
        );
        let logits = model.forward_with_caches(next, prompt_ids.len() + i - 1, &mut caches);
        let v: Vec<f32> = logits.reshape([cfg.vocab_size]).to_data().to_vec().unwrap();
        tokens.push(sample(&v, &scfg));
    }
    tokens
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn three_concurrent_requests_produce_isolated_outputs() {
    let fix = fixture();
    let cfg = ModelConfig::from_file(&fix.join("config.json")).unwrap();
    let tok = HfTokenizer::from_path(fix.join("tokenizer.json")).unwrap();
    let prompts = [
        "The quick brown fox",
        "Hello world",
        "ai engine cluster test",
    ];

    let prompt_ids: Vec<Vec<i32>> = prompts
        .iter()
        .map(|p| {
            tok.encode(p)
                .unwrap()
                .iter()
                .map(|x| *x as i32)
                .collect()
        })
        .collect();
    let baselines: Vec<Vec<u32>> = prompt_ids
        .iter()
        .map(|ids| single_node_greedy_5(&fix, &cfg, ids))
        .collect();

    // 3-node cluster: leader hosts 0..0; workers cover all 4 layers via
    // a partition_override that's stable across runs.
    let w1_id = generate_node_identity("w1").unwrap();
    let w1_ep = server_endpoint(&w1_id, "127.0.0.1:0".parse().unwrap()).unwrap();
    let w1_addr = w1_ep.local_addr().unwrap();
    let w2_id = generate_node_identity("w2").unwrap();
    let w2_ep = server_endpoint(&w2_id, "127.0.0.1:0".parse().unwrap()).unwrap();
    let w2_addr = w2_ep.local_addr().unwrap();

    let model_path = fix.join("model.safetensors");
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
        partition_override: Some(vec![("w1".into(), 0..2), ("w2".into(), 2..4)]),
    };

    let leader = Arc::new(ClusterLeader::start(&leader_id, lcfg).await.unwrap());

    // Fire all three requests concurrently against the same leader.
    let mut futures = Vec::new();
    for (i, ids) in prompt_ids.iter().cloned().enumerate() {
        let leader = leader.clone();
        let cfg = cfg.clone();
        let model_path = model_path.clone();
        futures.push(async move {
            let res = leader
                .generate::<B>(
                    &model_path,
                    &cfg,
                    0..0,
                    &ids,
                    5,
                    SamplingConfig {
                        temperature: 0.0,
                        top_p: None,
                        top_k: None,
                        seed: 0,
                    },
                )
                .await;
            (i, res)
        });
    }

    let results: Vec<(usize, anyhow::Result<Vec<u32>>)> =
        futures::future::join_all(futures).await;

    for (i, res) in results {
        let tokens = res.unwrap_or_else(|e| panic!("request {i} failed: {e:#}"));
        assert_eq!(
            tokens, baselines[i],
            "concurrent request {i} (prompt {:?}) did not match single-node baseline",
            prompts[i]
        );
    }
}
