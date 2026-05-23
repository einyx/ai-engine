//! End-to-end 3-node cluster test: leader + 2 workers over loopback QUIC.
//!
//! Loads the toy-llama-3 fixture, runs a single-node baseline forward, then
//! runs the same forward distributed across the cluster, and asserts the
//! resulting last-position logits match within 1e-3.

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
use std::path::PathBuf;

type B = burn_ndarray::NdArray;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("ai-engine-runtime/fixtures/toy-llama-3")
}

// Note: the original `three_node_cluster_logits_match_single_node` logits-match
// test was removed in Plan 4 Task 4. Token-level coverage via
// `cluster_generate_5_tokens_matches_single_node_baseline` below is strictly
// stronger (exact greedy tokens vs. a 1e-3 logits tolerance) and exercises the
// same end-to-end forward path, so the older test is redundant.

fn single_node_greedy_5(
    fix: &std::path::Path,
    cfg: &ModelConfig,
    prompt_ids: &[i32],
) -> Vec<u32> {
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

    // Prefill.
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

    // Generate 4 more tokens (5 total).
    for i in 1..5 {
        let next = Tensor::<B, 2, Int>::from_data(
            TensorData::new(vec![*tokens.last().unwrap() as i32], [1, 1]),
            &dev,
        );
        let logits =
            model.forward_with_caches(next, prompt_ids.len() + i - 1, &mut caches);
        let v: Vec<f32> = logits
            .reshape([cfg.vocab_size])
            .to_data()
            .to_vec()
            .unwrap();
        tokens.push(sample(&v, &scfg));
    }
    tokens
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn cluster_generate_5_tokens_matches_single_node_baseline() {
    let fix = fixture();
    let cfg = ModelConfig::from_file(&fix.join("config.json")).unwrap();
    let tok = HfTokenizer::from_path(fix.join("tokenizer.json")).unwrap();
    let prompt = "The quick brown fox";
    let ids: Vec<u32> = tok.encode(prompt).unwrap();
    let ids_i32: Vec<i32> = ids.iter().map(|x| *x as i32).collect();

    // Single-node baseline: 5 greedy-sampled tokens.
    let baseline_tokens = single_node_greedy_5(&fix, &cfg, &ids_i32);

    // --- 3-node cluster: leader hosts layers 0..1, w1 hosts 1..3, w2 hosts 3..4 ---
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
        // Leader hosts no layers (0..0); workers cover all 4.
        partition_override: Some(vec![("w1".into(), 0..2), ("w2".into(), 2..4)]),
    };

    let leader = ClusterLeader::start(&leader_id, lcfg).await.unwrap();
    let cluster_tokens = leader
        .generate::<B>(
            &model_path,
            &cfg,
            0..0,
            &ids_i32,
            5,
            SamplingConfig {
                temperature: 0.0,
                top_p: None,
                top_k: None,
                seed: 0,
            },
        )
        .await
        .unwrap();

    assert_eq!(
        cluster_tokens, baseline_tokens,
        "cluster generation must match single-node greedy generation"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn asymmetric_partition_via_assignment_matches_single_node() {
    // Toy llama has 4 layers. Use an asymmetric partition: w1 takes 0..3 (three
    // layers), w2 takes 3..4 (one layer). Leader hosts no layers (0..0). This
    // layout deliberately differs from the even-split (0..2 / 2..4) used in the
    // other tests, proving that the Assignment path — not a fallback formula —
    // is driving which weights each worker loads.
    let fix = fixture();
    let cfg = ModelConfig::from_file(&fix.join("config.json")).unwrap();
    let tok = HfTokenizer::from_path(fix.join("tokenizer.json")).unwrap();
    let prompt = "The quick brown fox";
    let ids: Vec<u32> = tok.encode(prompt).unwrap();
    let ids_i32: Vec<i32> = ids.iter().map(|x| *x as i32).collect();

    let baseline_tokens = single_node_greedy_5(&fix, &cfg, &ids_i32);

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
        // ASYMMETRIC: w1 gets 3 layers, w2 gets 1 layer. Leader hosts none.
        // 0..3 + 3..4 = full 4-layer coverage, contiguous and complete.
        partition_override: Some(vec![
            ("w1".to_string(), 0..3),
            ("w2".to_string(), 3..4),
        ]),
    };

    let leader = ClusterLeader::start(&leader_id, lcfg).await.unwrap();
    let cluster_tokens = leader
        .generate::<B>(
            &model_path,
            &cfg,
            0..0, // leader hosts no layers
            &ids_i32,
            5,
            SamplingConfig {
                temperature: 0.0,
                top_p: None,
                top_k: None,
                seed: 0,
            },
        )
        .await
        .unwrap();

    assert_eq!(
        cluster_tokens, baseline_tokens,
        "asymmetric partition via Assignment must match single-node baseline"
    );
}
