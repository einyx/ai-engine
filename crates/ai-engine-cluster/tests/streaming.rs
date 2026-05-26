//! Per-token streaming through `ClusterProvider::chat_stream`.
//!
//! Plan 4 Task 6: the leader spawns a background task that drives generation
//! and pushes each sampled token onto an mpsc channel; the provider adapts
//! that channel into a `BoxStream<ChatStreamEvent>` shaped like an OpenAI
//! `chat.completion.chunk` SSE.
//!
//! This test stands up a 3-node cluster (1 leader + 2 workers), calls
//! `provider.chat_stream(...)` with `max_tokens=3`, drives the returned
//! stream to completion, and asserts the shape: 3 content chunks + 1 final
//! chunk with `finish_reason: "stop"`.

use ai_engine_cluster::capability::BackendKind;
use ai_engine_cluster::leader::{ClusterLeader, LeaderConfig, WorkerEndpoint};
use ai_engine_cluster::provider::{ClusterProvider, LeaderState};
use ai_engine_cluster::tls::generate_node_identity;
use ai_engine_cluster::transport::quic::server_endpoint;
use ai_engine_cluster::worker::run_worker_full;
use ai_engine_provider::openai;
use ai_engine_provider::provider::{CallCtx, Credentials, Provider};
use ai_engine_runtime::config::ModelConfig;
use ai_engine_tokenizer::HfTokenizer;
use futures::StreamExt;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

type B = burn_ndarray::NdArray;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("ai-engine-runtime/fixtures/toy-llama-3")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn chat_stream_yields_per_token_chunks_and_finish() {
    let fix = fixture();
    let cfg = ModelConfig::from_file(&fix.join("config.json")).unwrap();
    let tok = HfTokenizer::from_path(fix.join("tokenizer.json")).unwrap();
    let model_path = fix.join("model.safetensors");

    // Workers.
    let w1_id = generate_node_identity("w1").unwrap();
    let w1_ep = server_endpoint(&w1_id, "127.0.0.1:0".parse().unwrap()).unwrap();
    let w1_addr = w1_ep.local_addr().unwrap();
    let w2_id = generate_node_identity("w2").unwrap();
    let w2_ep = server_endpoint(&w2_id, "127.0.0.1:0".parse().unwrap()).unwrap();
    let w2_addr = w2_ep.local_addr().unwrap();

    let cfg_for_w1 = cfg.clone();
    let mp1 = model_path.clone();
    let _w1_task = tokio::spawn(async move {
        run_worker_full::<B>(w1_ep, "w1".to_string(), BackendKind::Cpu, mp1, cfg_for_w1).await
    });
    let cfg_for_w2 = cfg.clone();
    let mp2 = model_path.clone();
    let _w2_task = tokio::spawn(async move {
        run_worker_full::<B>(w2_ep, "w2".to_string(), BackendKind::Cpu, mp2, cfg_for_w2).await
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

    let state = Arc::new(LeaderState {
        leader: leader.clone(),
        model_cfg: cfg.clone(),
        model_path: model_path.clone(),
        tokenizer: Arc::new(tok),
        leader_layers: 0..0,
    });

    let provider = ClusterProvider::new_leader_with_state("stream-test", state);

    let req = openai::ChatRequest {
        model: "toy".into(),
        messages: vec![openai::ChatMessage {
            role: "user".into(),
            content: openai::ChatContent::Text("hi".into()),
            extras: Default::default(),
        }],
        max_tokens: Some(3),
        temperature: Some(0.0),
        stream: Some(true),
        stream_options: None,
        extras: Default::default(),
    };

    let ctx = CallCtx {
        request_id: uuid::Uuid::now_v7(),
        deadline: None,
        upstream_model: "toy".into(),
    };

    let start = Instant::now();
    let mut stream = provider
        .chat_stream(req, &Credentials::none(), &ctx)
        .await
        .expect("chat_stream returns Ok");

    let mut events: Vec<openai::ChatStreamEvent> = Vec::new();
    while let Some(item) = stream.next().await {
        events.push(item.expect("no stream error"));
    }
    eprintln!(
        "got {} events in {:.2}s",
        events.len(),
        start.elapsed().as_secs_f64()
    );

    assert!(
        events.len() >= 4,
        "expected at least 4 events (3 content + 1 finish), got {}",
        events.len()
    );

    // First content chunk: delta.content present as a string.
    let first = &events[0];
    let first_delta = first.raw["choices"][0]["delta"]["content"].as_str();
    assert!(
        first_delta.is_some(),
        "first event missing delta.content string: {}",
        first.raw
    );

    // Last event: finish_reason == "stop".
    let last = events.last().unwrap();
    let finish = last.raw["choices"][0]["finish_reason"].as_str();
    assert_eq!(
        finish,
        Some("stop"),
        "last event finish_reason != stop: {}",
        last.raw
    );

    // Every event should be a chat.completion.chunk.
    for (i, ev) in events.iter().enumerate() {
        assert_eq!(
            ev.raw["object"].as_str(),
            Some("chat.completion.chunk"),
            "event[{i}] object != chat.completion.chunk: {}",
            ev.raw
        );
    }
}
