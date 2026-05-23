use airproxy_core::ctx::{ProviderBinding, RequestBody, RequestCtx, ResponseSlot};
use airproxy_core::stage::{Stage, StageOutcome};
use airproxy_provider::{
    error::ProviderError, openai,
    provider::{CallCtx, Capabilities, Credentials, EventStream, Provider},
};
use airproxy_stages::ProviderRegistry;
use airproxy_stages::forward::ForwardStage;
use futures::stream;
use http::HeaderMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

struct MockOpenAi {
    chat_calls: Arc<AtomicUsize>,
}

#[async_trait::async_trait]
impl Provider for MockOpenAi {
    fn id(&self) -> &str {
        "mock-openai"
    }
    fn kind(&self) -> &'static str {
        "openai"
    }
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            chat: true,
            ..Default::default()
        }
    }

    async fn chat(
        &self,
        _req: openai::ChatRequest,
        _creds: &Credentials,
        _ctx: &CallCtx,
    ) -> Result<openai::ChatResponse, ProviderError> {
        self.chat_calls.fetch_add(1, Ordering::SeqCst);
        Ok(openai::ChatResponse {
            id: "x".into(),
            model: "gpt-4o".into(),
            choices: vec![],
            usage: Some(openai::Usage {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
            }),
            extras: Default::default(),
        })
    }

    async fn chat_stream(
        &self,
        _req: openai::ChatRequest,
        _creds: &Credentials,
        _ctx: &CallCtx,
    ) -> Result<EventStream<openai::ChatStreamEvent>, ProviderError> {
        let events = vec![
            Ok(openai::ChatStreamEvent {
                raw: serde_json::json!({"choices": [{"delta": {"content": "hi"}}]}),
            }),
            Ok(openai::ChatStreamEvent {
                raw: serde_json::json!({"usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}}),
            }),
        ];
        Ok(Box::pin(stream::iter(events)))
    }
}

fn make_ctx(stream: bool) -> RequestCtx {
    let mut ctx = RequestCtx::new(
        "/v1/chat/completions",
        HeaderMap::new(),
        0,
        RequestBody::OpenAiChat(openai::ChatRequest {
            model: "gpt-4o".into(),
            messages: vec![],
            stream: Some(stream),
            temperature: None,
            max_tokens: None,
            stream_options: None,
            extras: Default::default(),
        }),
    );
    ctx.binding = Some(ProviderBinding {
        provider_id: "mock-openai".into(),
        upstream_model: "gpt-4o".into(),
    });
    ctx.usage_slot = Some(Arc::new(std::sync::Mutex::new(None)));
    ctx
}

fn make_registry(provider: Arc<dyn Provider>) -> Arc<ProviderRegistry> {
    let mut r = ProviderRegistry::new();
    r.insert("mock-openai", provider, Credentials::none());
    Arc::new(r)
}

#[tokio::test]
async fn non_streaming_chat_fills_full_response_and_usage() {
    let calls = Arc::new(AtomicUsize::new(0));
    let provider = Arc::new(MockOpenAi {
        chat_calls: calls.clone(),
    });
    let stage = ForwardStage {
        providers: make_registry(provider),
    };
    let mut ctx = make_ctx(false);
    let r = stage.process(&mut ctx).await.unwrap();
    assert!(matches!(r, StageOutcome::Continue));
    assert!(matches!(ctx.response, ResponseSlot::Full(_)));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let usage = ctx
        .usage_slot
        .unwrap()
        .lock()
        .unwrap()
        .expect("usage recorded");
    assert_eq!(usage.prompt, 10);
    assert_eq!(usage.completion, 20);
    assert_eq!(usage.total, 30);
}

#[tokio::test]
async fn streaming_chat_taps_usage_from_terminal_event() {
    use futures::StreamExt;
    let calls = Arc::new(AtomicUsize::new(0));
    let provider = Arc::new(MockOpenAi {
        chat_calls: calls.clone(),
    });
    let stage = ForwardStage {
        providers: make_registry(provider),
    };
    let mut ctx = make_ctx(true);
    stage.process(&mut ctx).await.unwrap();
    let ResponseSlot::Stream(mut s) = std::mem::replace(&mut ctx.response, ResponseSlot::Pending)
    else {
        panic!("expected Stream");
    };
    // Drain the stream
    let mut events = 0;
    while let Some(_ev) = s.next().await {
        events += 1;
    }
    assert_eq!(events, 2);
    let usage = ctx
        .usage_slot
        .unwrap()
        .lock()
        .unwrap()
        .expect("usage tapped");
    assert_eq!(usage.prompt, 7);
    assert_eq!(usage.completion, 3);
    assert_eq!(usage.total, 10);
}

#[tokio::test]
async fn empty_body_passes_through_without_calling_provider() {
    let calls = Arc::new(AtomicUsize::new(0));
    let provider = Arc::new(MockOpenAi {
        chat_calls: calls.clone(),
    });
    let stage = ForwardStage {
        providers: make_registry(provider),
    };
    let mut ctx = RequestCtx::new("/v1/models", HeaderMap::new(), 0, RequestBody::Empty);
    ctx.binding = Some(ProviderBinding {
        provider_id: "mock-openai".into(),
        upstream_model: "_".into(),
    });
    stage.process(&mut ctx).await.unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(matches!(ctx.response, ResponseSlot::Pending));
}

#[tokio::test]
async fn missing_binding_produces_internal_error() {
    let provider = Arc::new(MockOpenAi {
        chat_calls: Arc::new(AtomicUsize::new(0)),
    });
    let stage = ForwardStage {
        providers: make_registry(provider),
    };
    let mut ctx = RequestCtx::new(
        "/v1/chat/completions",
        HeaderMap::new(),
        0,
        RequestBody::OpenAiChat(openai::ChatRequest {
            model: "gpt-4o".into(),
            messages: vec![],
            stream: None,
            temperature: None,
            max_tokens: None,
            stream_options: None,
            extras: Default::default(),
        }),
    );
    // No binding set
    let err = stage.process(&mut ctx).await.unwrap_err();
    assert!(matches!(
        err.error,
        airproxy_core::error::GatewayError::Internal(_)
    ));
}
