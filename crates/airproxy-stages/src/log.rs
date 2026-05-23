use airproxy_core::ctx::{RequestBody, RequestCtx, ResponseSlot};
use airproxy_core::error::GatewayError;
use airproxy_core::stage::{Stage, StageError, StageOutcome};
use serde_json::json;

pub trait LogSink: Send + Sync + 'static {
    fn write_line(&self, line: &str);
}

pub struct StdoutSink;
impl LogSink for StdoutSink {
    fn write_line(&self, line: &str) {
        println!("{line}");
    }
}

pub struct LogStage {
    pub sink: Box<dyn LogSink>,
}

impl LogStage {
    pub fn stdout() -> Self {
        Self {
            sink: Box::new(StdoutSink),
        }
    }
}

#[async_trait::async_trait]
impl Stage for LogStage {
    fn name(&self) -> &'static str {
        "log"
    }
    fn is_terminal(&self) -> bool {
        true
    }

    async fn process(&self, ctx: &mut RequestCtx) -> Result<StageOutcome, StageError> {
        let status = ctx
            .error
            .as_ref()
            .map(|e| e.http_status())
            .unwrap_or_else(|| match &ctx.response {
                ResponseSlot::Full(r) if r.status != 0 => r.status,
                _ => 200,
            });
        let duration_ms = ctx.started_at.elapsed().as_millis() as u64;
        let is_stream = matches!(&ctx.response, ResponseSlot::Stream(_));
        let usage = ctx
            .usage_slot
            .as_ref()
            .and_then(|u| u.lock().ok().and_then(|g| *g));
        let model = model_of(&ctx.body);
        let identity = identity_label(ctx.identity.as_ref());
        let line = json!({
            "ts": time::OffsetDateTime::now_utc()
                .format(&time::format_description::well_known::Rfc3339)
                .unwrap_or_default(),
            "request_id": ctx.request_id.to_string(),
            "route": ctx.route,
            "model": model,
            "provider": ctx.binding.as_ref().map(|b| b.provider_id.as_str()),
            "upstream_model": ctx.binding.as_ref().map(|b| b.upstream_model.as_str()),
            "status": status,
            "duration_ms": duration_ms,
            "stream": is_stream,
            "tokens": usage.map(|u| json!({
                "prompt": u.prompt,
                "completion": u.completion,
                "total": u.total,
            })),
            "identity": identity,
            "error": ctx.error.as_ref().map(error_label),
        });
        self.sink.write_line(&line.to_string());
        Ok(StageOutcome::Continue)
    }
}

fn model_of(body: &RequestBody) -> Option<&str> {
    match body {
        RequestBody::OpenAiChat(c) => Some(c.model.as_str()),
        RequestBody::AnthropicMessages(m) => Some(m.model.as_str()),
        RequestBody::OpenAiEmbeddings(e) => Some(e.model.as_str()),
        RequestBody::Empty => None,
    }
}

fn identity_label(identity: Option<&airproxy_core::ctx::Identity>) -> Option<String> {
    use airproxy_core::ctx::Identity;
    match identity? {
        Identity::Anonymous { .. } => Some("anonymous".to_string()),
        Identity::Holder { name } => Some(name.clone()),
    }
}

fn error_label(e: &GatewayError) -> String {
    e.to_string()
}
