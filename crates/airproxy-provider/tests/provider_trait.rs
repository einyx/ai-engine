use airproxy_provider::error::ProviderError;
use airproxy_provider::provider::{Capabilities, Credentials, CallCtx, Provider};
use std::sync::Arc;
use uuid::Uuid;

struct Dummy;

#[async_trait::async_trait]
impl Provider for Dummy {
    fn id(&self) -> &str { "dummy" }
    fn kind(&self) -> &'static str { "openai" }
    fn capabilities(&self) -> Capabilities { Capabilities::default() }
}

#[test]
fn trait_is_object_safe() {
    let p: Arc<dyn Provider> = Arc::new(Dummy);
    assert_eq!(p.id(), "dummy");
    assert_eq!(p.kind(), "openai");
}

#[tokio::test]
async fn default_methods_return_unsupported() {
    let p = Dummy;
    let creds = Credentials::none();
    let ctx = CallCtx { request_id: Uuid::now_v7(), deadline: None, upstream_model: "x".into() };

    let r = p.chat(
        airproxy_provider::openai::ChatRequest {
            model: "x".into(),
            messages: vec![],
            stream: None,
            temperature: None,
            max_tokens: None,
            stream_options: None,
            extras: Default::default(),
        },
        &creds,
        &ctx,
    ).await;
    assert!(matches!(r, Err(ProviderError::Unsupported)));
}

#[test]
fn credentials_none_has_no_keys() {
    let c = Credentials::none();
    assert!(c.api_key.is_none());
    assert!(c.raw_bearer.is_none());
    assert!(c.extra_headers.is_empty());
}

#[test]
fn capabilities_default_is_all_false() {
    let c = Capabilities::default();
    assert!(!c.chat);
    assert!(!c.messages);
    assert!(!c.embeddings);
    assert!(!c.streaming);
    assert!(!c.tools);
    assert!(!c.vision);
}
