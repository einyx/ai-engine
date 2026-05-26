use ai_engine_core::activity::{ActivityLog, ChatEvent};
use ai_engine_core::ctx::{RequestBody, RequestCtx, ResponseSlot};
use ai_engine_core::metrics::GatewayMetrics;
use ai_engine_core::error::GatewayError;
use ai_engine_core::stage::{Stage, StageError, StageOutcome};
use serde_json::json;
use std::sync::Arc;

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
    /// When set, each inference request is recorded for the activity graph.
    pub activity: Option<Arc<ActivityLog>>,
    /// When set, per-provider request stats (count/errors/latency) are recorded.
    pub metrics: Option<Arc<GatewayMetrics>>,
}

impl LogStage {
    pub fn stdout() -> Self {
        Self {
            sink: Box::new(StdoutSink),
            activity: None,
            metrics: None,
        }
    }

    /// Attach an activity log so chat → model → provider edges appear in `/graph`.
    pub fn with_activity(mut self, activity: Arc<ActivityLog>) -> Self {
        self.activity = Some(activity);
        self
    }

    /// Attach gateway metrics so per-provider request stats are recorded.
    pub fn with_metrics(mut self, metrics: Arc<GatewayMetrics>) -> Self {
        self.metrics = Some(metrics);
        self
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
        let ts = time::OffsetDateTime::now_utc()
            .format(&time::format_description::well_known::Rfc3339)
            .unwrap_or_default();
        // Record inference requests for the activity graph. The model comes from
        // the binding (NOT ctx.body — forward consumed it), and provider_id is
        // the upstream that actually served the request. Tokens may be 0 for
        // streaming since usage fills lazily after this stage.
        // Per-provider request stats for the cluster dashboard (status + latency).
        if let (Some(m), Some(b)) = (&self.metrics, ctx.binding.as_ref()) {
            if !b.provider_id.is_empty() {
                m.record_request(&b.provider_id, status >= 400, duration_ms);
            }
        }

        if let (Some(act), Some(b)) = (&self.activity, ctx.binding.as_ref()) {
            if !b.provider_id.is_empty() && !b.upstream_model.is_empty() {
                let prompt = ctx
                    .metadata
                    .get("prompt_preview")
                    .and_then(|v| v.as_str())
                    .map(str::to_string);
                act.record(ChatEvent {
                    request_id: ctx.request_id.to_string(),
                    ts: ts.clone(),
                    model: b.upstream_model.clone(),
                    provider: b.provider_id.clone(),
                    tokens: usage.map(|u| u.completion).unwrap_or(0),
                    duration_ms,
                    status,
                    prompt,
                });
            }
        }

        let line = json!({
            "ts": ts,
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

fn identity_label(identity: Option<&ai_engine_core::ctx::Identity>) -> Option<String> {
    use ai_engine_core::ctx::Identity;
    match identity? {
        Identity::Anonymous { .. } => Some("anonymous".to_string()),
        Identity::Holder { name } => Some(name.clone()),
    }
}

fn error_label(e: &GatewayError) -> String {
    e.to_string()
}
