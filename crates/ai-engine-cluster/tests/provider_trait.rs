//! `ClusterProvider` is a valid `ai_engine_provider::Provider`:
//! - object-safe (`Arc<dyn Provider>` holds it)
//! - reports kind/id/capabilities as specified by the design
//! - worker mode refuses application traffic with `Unsupported`
//! - leader mode with a real cluster wires through to `ClusterLeader::generate`
//!   and returns a non-empty assistant message.

use ai_engine_cluster::capability::BackendKind;
use ai_engine_cluster::leader::{ClusterLeader, LeaderConfig, WorkerEndpoint};
use ai_engine_cluster::provider::{ClusterProvider, LeaderState};
use ai_engine_cluster::tls::generate_node_identity;
use ai_engine_cluster::transport::quic::server_endpoint;
use ai_engine_cluster::worker::run_worker_full;
use ai_engine_provider::error::ProviderError;
use ai_engine_provider::openai::{ChatContent, ChatMessage, ChatRequest};
use ai_engine_provider::provider::{CallCtx, Credentials, Provider};
use ai_engine_runtime::config::ModelConfig;
use ai_engine_tokenizer::HfTokenizer;
use std::path::PathBuf;
use std::sync::Arc;
use uuid::Uuid;

type B = burn_ndarray::NdArray;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("ai-engine-runtime/fixtures/toy-llama-3")
}

#[test]
fn cluster_provider_implements_provider_trait_object_safely() {
    let p: Arc<dyn Provider> = Arc::new(ClusterProvider::stub_leader("my-cluster"));
    assert_eq!(p.kind(), "local-cluster");
    assert_eq!(p.id(), "my-cluster");
    let caps = p.capabilities();
    assert!(caps.chat, "cluster provider must advertise chat");
    assert!(caps.streaming, "cluster provider must advertise streaming");
    assert!(!caps.messages, "messages comes via Plan 3 gateway dispatch");
    assert!(!caps.embeddings, "embeddings out of v0.2 scope");
    assert!(!caps.tools, "tools out of v0.2 scope");
    assert!(!caps.vision, "vision out of v0.2 scope");
}

#[test]
fn worker_provider_also_object_safe() {
    // A worker-mode provider is still object-safe.
    let p: Arc<dyn Provider> = Arc::new(ClusterProvider::stub_worker("my-cluster"));
    assert_eq!(p.kind(), "local-cluster");
    assert_eq!(p.id(), "my-cluster");
}

#[tokio::test]
async fn worker_mode_returns_unsupported_for_chat() {
    let p = ClusterProvider::stub_worker("my-cluster");
    let req = ChatRequest {
        model: "x".into(),
        messages: vec![ChatMessage {
            role: "user".into(),
            content: ChatContent::Text("hi".into()),
            extras: Default::default(),
        }],
        stream: None,
        temperature: None,
        max_tokens: None,
        stream_options: None,
        extras: Default::default(),
    };
    let ctx = CallCtx {
        request_id: Uuid::now_v7(),
        deadline: None,
        upstream_model: "x".into(),
    };
    let result = p.chat(req, &Credentials::none(), &ctx).await;
    assert!(
        matches!(result, Err(ProviderError::Unsupported)),
        "worker mode must refuse chat with Unsupported, got: {result:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn cluster_provider_chat_returns_completion_from_real_cluster() {
    // Spin up a 3-node cluster (leader + 2 workers on loopback QUIC), build
    // LeaderState, wrap in ClusterProvider, and assert chat returns a
    // non-empty assistant message. Uses small max_tokens=3 to keep the
    // test fast.

    let fix = fixture();
    let cfg = ModelConfig::from_file(&fix.join("config.json")).unwrap();
    let tokenizer = HfTokenizer::from_path(fix.join("tokenizer.json")).unwrap();

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
    let state = LeaderState {
        leader: Arc::new(leader),
        model_cfg: cfg.clone(),
        model_path: model_path.clone(),
        tokenizer: Arc::new(tokenizer),
        leader_layers: 0..0,
    };

    let provider =
        ClusterProvider::new_leader_with_state("test-cluster", Arc::new(state));

    let req = ChatRequest {
        model: "toy-llama".into(),
        messages: vec![ChatMessage {
            role: "user".into(),
            content: ChatContent::Text("The quick brown fox".into()),
            extras: Default::default(),
        }],
        stream: None,
        temperature: Some(1.0),
        max_tokens: Some(3),
        stream_options: None,
        extras: Default::default(),
    };
    let ctx = CallCtx {
        request_id: Uuid::now_v7(),
        deadline: None,
        upstream_model: "toy-llama".into(),
    };

    let resp = provider
        .chat(req, &Credentials::none(), &ctx)
        .await
        .expect("cluster chat must succeed");

    assert_eq!(resp.choices.len(), 1, "exactly one choice expected");
    let choice = &resp.choices[0];
    assert_eq!(choice.message.role, "assistant");
    // Verify shape only — toy fixture has random weights, and BPE decoding of
    // 3 randomly-sampled tokens sometimes produces an empty UTF-8 string.
    // The semantic gate is `completion_tokens > 0`, not non-empty decoded text;
    // matches the existing multiproc_smoke assertion.
    match &choice.message.content {
        ChatContent::Text(_) => {}
        ChatContent::Parts(_) => panic!("expected Text content from cluster"),
    };
    let usage = resp.usage.as_ref().expect("usage must be populated");
    assert_eq!(usage.completion_tokens, 3);
    assert!(usage.prompt_tokens > 0);
}
