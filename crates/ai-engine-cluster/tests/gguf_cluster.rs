//! 3-node cluster loading the toy-llama-3-gguf fixture via load_weights dispatch.
//! Just verifies the full cluster forward runs without diverging — same shape as
//! the existing q8_cluster / q4_cluster tests, but pointing at the GGUF fixture.

use std::path::PathBuf;

type B = burn_ndarray::NdArray;

fn gguf_fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("ai-engine-runtime/fixtures/toy-llama-3-gguf")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn gguf_cluster_generation_runs_end_to_end() {
    use ai_engine_cluster::{
        capability::BackendKind,
        leader::{ClusterLeader, LeaderConfig, WorkerEndpoint},
        tls::generate_node_identity,
        transport::quic::server_endpoint,
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
    let leader = ClusterLeader::start(&leader_id, lcfg).await.unwrap();

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
