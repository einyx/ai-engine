//! `ClusterProvider` is a valid `ai_engine_provider::Provider`:
//! - object-safe (`Arc<dyn Provider>` holds it)
//! - reports kind/id/capabilities as specified by the design
//! - worker mode refuses application traffic with `Unsupported`

use ai_engine_cluster::provider::ClusterProvider;
use ai_engine_provider::error::ProviderError;
use ai_engine_provider::openai::{ChatContent, ChatMessage, ChatRequest};
use ai_engine_provider::provider::{CallCtx, Credentials, Provider};
use std::sync::Arc;
use uuid::Uuid;

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
