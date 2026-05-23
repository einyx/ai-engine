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

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn three_node_cluster_logits_match_single_node() {
    let fix = fixture();
    let cfg = ModelConfig::from_file(&fix.join("config.json")).unwrap();
    let tok = HfTokenizer::from_path(fix.join("tokenizer.json")).unwrap();
    let prompt = "The quick brown fox";
    let ids: Vec<u32> = tok.encode(prompt).unwrap();
    let ids_i32: Vec<i32> = ids.iter().map(|x| *x as i32).collect();

    // --- single-node baseline ---
    let dev = Default::default();
    let weights = load_range::<B>(
        &fix.join("model.safetensors"),
        &cfg,
        0..cfg.n_layers,
        true,
        true,
        &dev,
    )
    .unwrap();
    let model = Model::<B>::from_loaded(&cfg, weights, &dev).unwrap();
    let baseline_logits: Vec<f32> = model
        .forward(
            Tensor::<B, 2, Int>::from_data(
                TensorData::new(ids_i32.clone(), [1, ids.len()]),
                &dev,
            ),
            0,
        )
        .slice([0..1, (ids.len() - 1)..ids.len(), 0..cfg.vocab_size])
        .reshape([cfg.vocab_size])
        .to_data()
        .to_vec()
        .unwrap();

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
            1..3,
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
            3..4,
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
    };

    let mut leader = ClusterLeader::start(&leader_id, lcfg).await.unwrap();
    let cluster_logits = leader
        .full_forward_for_test::<B>(&model_path, &cfg, 0..1, &ids_i32)
        .await
        .unwrap();

    assert_eq!(cluster_logits.len(), baseline_logits.len());
    let max_diff: f32 = baseline_logits
        .iter()
        .zip(cluster_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0., f32::max);
    eprintln!("baseline vs cluster max diff = {max_diff}");
    assert!(
        max_diff < 1e-3,
        "cluster logits should match baseline within 1e-3 (got {max_diff})"
    );
}

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
            1..3,
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
            3..4,
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
    };

    let mut leader = ClusterLeader::start(&leader_id, lcfg).await.unwrap();
    let cluster_tokens = leader
        .generate::<B>(
            &model_path,
            &cfg,
            0..1,
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
