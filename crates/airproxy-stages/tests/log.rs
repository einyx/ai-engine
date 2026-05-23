use airproxy_core::ctx::{Identity, ProviderBinding, RecordedUsage, RequestBody, RequestCtx};
use airproxy_core::error::GatewayError;
use airproxy_core::stage::Stage;
use airproxy_provider::openai::{ChatContent, ChatMessage, ChatRequest};
use airproxy_stages::log::{LogSink, LogStage};
use http::HeaderMap;
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct CaptureSink {
    lines: Mutex<Vec<String>>,
}
impl LogSink for CaptureSink {
    fn write_line(&self, line: &str) {
        self.lines.lock().unwrap().push(line.to_string());
    }
}

fn body() -> RequestBody {
    RequestBody::OpenAiChat(ChatRequest {
        model: "gpt-4o".into(),
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
    })
}

fn ctx_full() -> (RequestCtx, Arc<CaptureSink>) {
    let mut ctx = RequestCtx::new("/v1/chat/completions", HeaderMap::new(), 42, body());
    ctx.identity = Some(Identity::Holder {
        name: "alice".into(),
    });
    ctx.binding = Some(ProviderBinding {
        provider_id: "openai-prod".into(),
        upstream_model: "gpt-4o-2024-08-06".into(),
    });
    let slot = Arc::new(Mutex::new(Some(RecordedUsage {
        prompt: 5,
        completion: 7,
        total: 12,
    })));
    ctx.usage_slot = Some(slot);
    let sink = Arc::new(CaptureSink::default());
    (ctx, sink)
}

#[tokio::test]
async fn emits_one_jsonl_line_with_required_fields() {
    let (mut ctx, sink) = ctx_full();
    let stage = LogStage {
        sink: Box::new(SinkRef(sink.clone())),
    };
    stage.process(&mut ctx).await.unwrap();
    let lines = sink.lines.lock().unwrap();
    assert_eq!(lines.len(), 1);
    let v: serde_json::Value = serde_json::from_str(&lines[0]).unwrap();
    assert_eq!(v["route"], "/v1/chat/completions");
    assert_eq!(v["model"], "gpt-4o");
    assert_eq!(v["provider"], "openai-prod");
    assert_eq!(v["upstream_model"], "gpt-4o-2024-08-06");
    assert_eq!(v["identity"], "alice");
    assert_eq!(v["status"], 200);
    assert_eq!(v["tokens"]["prompt"], 5);
    assert_eq!(v["tokens"]["completion"], 7);
    assert_eq!(v["tokens"]["total"], 12);
    assert_eq!(v["error"], serde_json::Value::Null);
}

#[tokio::test]
async fn logs_even_with_error_set() {
    let (mut ctx, sink) = ctx_full();
    ctx.error = Some(GatewayError::Unauthorized);
    let stage = LogStage {
        sink: Box::new(SinkRef(sink.clone())),
    };
    stage.process(&mut ctx).await.unwrap();
    let lines = sink.lines.lock().unwrap();
    let v: serde_json::Value = serde_json::from_str(&lines[0]).unwrap();
    assert_eq!(v["status"], 401);
    assert!(v["error"].is_string());
}

// Forwards write_line calls to an Arc<CaptureSink>. LogStage takes
// Box<dyn LogSink>, which requires 'static; this wrapper makes that work
// without moving ownership of the Arc.
struct SinkRef(Arc<CaptureSink>);
impl LogSink for SinkRef {
    fn write_line(&self, line: &str) {
        self.0.write_line(line);
    }
}
