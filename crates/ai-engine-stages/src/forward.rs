use std::sync::Arc;

use ai_engine_core::ctx::{
    GatewayResponse, Identity, RecordedUsage, RequestBody, RequestCtx, ResponseSlot, StreamItem,
};
use ai_engine_core::error::GatewayError;
use ai_engine_core::stage::{Stage, StageError, StageOutcome};
use ai_engine_provider::{
    error::ProviderError,
    openai,
    provider::{CallCtx, Credentials},
};
use futures::StreamExt;
use http::HeaderMap;

use ai_engine_core::metrics::GatewayMetrics;

use crate::{LoadTracker, ProviderRegistry};

pub struct ForwardStage {
    pub providers: Arc<ProviderRegistry>,
    pub tracker: Arc<LoadTracker>,
    pub metrics: Arc<GatewayMetrics>,
}

/// Errors worth retrying against another provider in the pool: transport
/// failures and upstream 5xx. Client errors (4xx), `Unsupported`, and malformed
/// responses are deterministic — retrying elsewhere won't help.
fn retryable(e: &ProviderError) -> bool {
    match e {
        ProviderError::Connect(_) | ProviderError::Timeout | ProviderError::Stream(_) => true,
        ProviderError::Status { status, .. } => *status >= 500,
        _ => false,
    }
}

impl ForwardStage {
    /// Index of the least-busy candidate not already tried, or `None` when the
    /// pool is exhausted. Ties break toward the earliest in route order.
    fn pick(&self, candidates: &[String], tried: &[usize]) -> Option<usize> {
        candidates
            .iter()
            .enumerate()
            .filter(|(i, _)| !tried.contains(i))
            .min_by_key(|(_, id)| self.tracker.load(id))
            .map(|(i, _)| i)
    }
}

#[async_trait::async_trait]
impl Stage for ForwardStage {
    fn name(&self) -> &'static str {
        "forward"
    }

    async fn process(&self, ctx: &mut RequestCtx) -> Result<StageOutcome, StageError> {
        let stage = self.name();

        let binding = ctx.binding.as_ref().ok_or_else(|| StageError {
            stage,
            error: GatewayError::Internal(anyhow::anyhow!(
                "forward: no binding (model_route must run first)"
            )),
        })?;
        // Pool of providers serving this model, in route order. (Fall back to a
        // lone resolved provider_id for any caller that set it directly.)
        let candidates: Vec<String> = if !binding.candidates.is_empty() {
            binding.candidates.clone()
        } else if !binding.provider_id.is_empty() {
            vec![binding.provider_id.clone()]
        } else {
            Vec::new()
        };
        let upstream_model = binding.upstream_model.clone();

        // Take ownership of the request body; clone per attempt so a failover
        // can re-send it. Replace with `Empty` to keep `ctx.body` valid later.
        let body = std::mem::replace(&mut ctx.body, RequestBody::Empty);
        if let RequestBody::Empty = body {
            // Nothing to forward (e.g. /v1/models, /healthz).
            return Ok(StageOutcome::Continue);
        }

        let mut tried: Vec<usize> = Vec::new();
        let mut last_err: Option<ProviderError> = None;

        loop {
            let Some(idx) = self.pick(&candidates, &tried) else {
                // Pool exhausted: surface the last upstream error, or an
                // internal error if there were never any candidates.
                return Err(match last_err {
                    Some(e) => stage_err(stage, e),
                    None => StageError {
                        stage,
                        error: GatewayError::Internal(anyhow::anyhow!(
                            "forward: no providers in pool"
                        )),
                    },
                });
            };
            let id = candidates[idx].clone();
            let Some((provider, default_creds)) = self.providers.get(&id) else {
                tried.push(idx);
                last_err = Some(ProviderError::Connect(format!("unknown provider {id}")));
                continue;
            };

            // Credentials: passthrough prefers the caller's bearer; otherwise
            // the provider default.
            let creds = match ctx.identity.as_ref() {
                Some(Identity::Anonymous {
                    raw_bearer: Some(b),
                }) => Credentials {
                    api_key: default_creds.api_key.clone(),
                    raw_bearer: Some(b.clone()),
                    extra_headers: default_creds.extra_headers.clone(),
                },
                _ => default_creds.clone(),
            };
            let call_ctx = CallCtx {
                request_id: ctx.request_id,
                deadline: None,
                upstream_model: upstream_model.clone(),
            };
            // Record the provider we're actually dispatching to (for the log
            // stage) and count it as in-flight.
            if let Some(b) = ctx.binding.as_mut() {
                b.provider_id = id.clone();
            }
            let guard = self.tracker.acquire(&id);

            match &body {
                RequestBody::OpenAiChat(req) if req.stream.unwrap_or(false) => {
                    match provider.chat_stream(req.clone(), &creds, &call_ctx).await {
                        Ok(stream) => {
                            let tapped = tap_openai_usage(
                                stream,
                                ctx.usage_slot.clone(),
                                self.metrics.clone(),
                                id.clone(),
                            );
                            // Hold the in-flight guard for the stream's lifetime.
                            let counted = tapped.map(move |it| {
                                let _hold = &guard;
                                it
                            });
                            ctx.response = ResponseSlot::Stream(Box::pin(counted));
                            return Ok(StageOutcome::Continue);
                        }
                        Err(e) if retryable(&e) => {
                            tried.push(idx);
                            last_err = Some(e);
                        }
                        Err(e) => return Err(stage_err(stage, e)),
                    }
                }
                RequestBody::OpenAiChat(req) => {
                    match provider.chat(req.clone(), &creds, &call_ctx).await {
                        Ok(resp) => {
                            if let Some(u) = resp.usage.as_ref() {
                                record_openai_usage(&ctx.usage_slot, u);
                                self.metrics.add_output(&id, u.completion_tokens as u64);
                            }
                            ctx.response = full_json(stage, &resp)?;
                            return Ok(StageOutcome::Continue);
                        }
                        Err(e) if retryable(&e) => {
                            tried.push(idx);
                            last_err = Some(e);
                        }
                        Err(e) => return Err(stage_err(stage, e)),
                    }
                }
                RequestBody::AnthropicMessages(req) if req.stream.unwrap_or(false) => {
                    match provider.messages_stream(req.clone(), &creds, &call_ctx).await {
                        Ok(stream) => {
                            let tapped = tap_anthropic_usage(
                                stream,
                                ctx.usage_slot.clone(),
                                self.metrics.clone(),
                                id.clone(),
                            );
                            let counted = tapped.map(move |it| {
                                let _hold = &guard;
                                it
                            });
                            ctx.response = ResponseSlot::Stream(Box::pin(counted));
                            return Ok(StageOutcome::Continue);
                        }
                        Err(e) if retryable(&e) => {
                            tried.push(idx);
                            last_err = Some(e);
                        }
                        Err(e) => return Err(stage_err(stage, e)),
                    }
                }
                RequestBody::AnthropicMessages(req) => {
                    match provider.messages(req.clone(), &creds, &call_ctx).await {
                        Ok(resp) => {
                            record_anthropic_usage(&ctx.usage_slot, &resp.usage);
                            self.metrics
                                .add_output(&id, resp.usage.output_tokens as u64);
                            ctx.response = full_json(stage, &resp)?;
                            return Ok(StageOutcome::Continue);
                        }
                        Err(e) if retryable(&e) => {
                            tried.push(idx);
                            last_err = Some(e);
                        }
                        Err(e) => return Err(stage_err(stage, e)),
                    }
                }
                RequestBody::OpenAiEmbeddings(req) => {
                    match provider.embeddings(req.clone(), &creds, &call_ctx).await {
                        Ok(resp) => {
                            if let Some(u) = resp.usage.as_ref() {
                                record_openai_usage(&ctx.usage_slot, u);
                            }
                            ctx.response = full_json(stage, &resp)?;
                            return Ok(StageOutcome::Continue);
                        }
                        Err(e) if retryable(&e) => {
                            tried.push(idx);
                            last_err = Some(e);
                        }
                        Err(e) => return Err(stage_err(stage, e)),
                    }
                }
                RequestBody::Empty => return Ok(StageOutcome::Continue),
            }
            // Reached only on a retryable error: drop this provider's guard
            // before looping to the next candidate.
            drop(guard);
        }
    }
}

#[cfg(test)]
mod failover_tests {
    use super::{retryable, ForwardStage};
    use crate::{LoadTracker, ProviderRegistry};
    use ai_engine_provider::error::ProviderError;
    use std::sync::Arc;

    #[test]
    fn retryable_classification() {
        assert!(retryable(&ProviderError::Connect("x".into())));
        assert!(retryable(&ProviderError::Timeout));
        assert!(retryable(&ProviderError::Stream("x".into())));
        assert!(retryable(&ProviderError::Status { status: 503, body: Default::default() }));
        // Client errors and deterministic failures are not retried.
        assert!(!retryable(&ProviderError::Status { status: 400, body: Default::default() }));
        assert!(!retryable(&ProviderError::Unsupported));
        assert!(!retryable(&ProviderError::InvalidResponse("x".into())));
    }

    #[test]
    fn pick_prefers_least_loaded_then_route_order() {
        let tracker = Arc::new(LoadTracker::new(["a".to_string(), "b".to_string(), "c".to_string()]));
        let stage = ForwardStage {
            providers: Arc::new(ProviderRegistry::new()),
            tracker: tracker.clone(),
            metrics: Arc::new(ai_engine_core::metrics::GatewayMetrics::default()),
        };
        let pool = vec!["a".to_string(), "b".to_string(), "c".to_string()];

        // All equal → earliest in route order.
        assert_eq!(stage.pick(&pool, &[]), Some(0));

        // Load up `a`; `b` (next) should win.
        let _g = tracker.acquire("a");
        assert_eq!(stage.pick(&pool, &[]), Some(1));

        // Exclude tried indices 0 and 1 → only `c` remains.
        assert_eq!(stage.pick(&pool, &[0, 1]), Some(2));

        // Everything tried → none.
        assert_eq!(stage.pick(&pool, &[0, 1, 2]), None);
    }
}

/// Serialize a provider response into a 200 `Full` response slot.
fn full_json<T: serde::Serialize>(
    stage: &'static str,
    resp: &T,
) -> Result<ResponseSlot, StageError> {
    let body = serde_json::to_vec(resp).map_err(|e| StageError {
        stage,
        error: GatewayError::Internal(anyhow::Error::new(e)),
    })?;
    Ok(ResponseSlot::Full(GatewayResponse {
        status: 200,
        headers: HeaderMap::new(),
        body: body.into(),
    }))
}

fn stage_err(stage: &'static str, e: ProviderError) -> StageError {
    StageError {
        stage,
        error: GatewayError::Provider(e),
    }
}

fn record_openai_usage(slot: &Option<ai_engine_core::ctx::UsageSlot>, u: &openai::Usage) {
    if let Some(s) = slot {
        if let Ok(mut g) = s.lock() {
            *g = Some(RecordedUsage {
                prompt: u.prompt_tokens,
                completion: u.completion_tokens,
                total: u.total_tokens,
            });
        }
    }
}

fn record_anthropic_usage(
    slot: &Option<ai_engine_core::ctx::UsageSlot>,
    u: &ai_engine_provider::anthropic::AnthropicUsage,
) {
    if let Some(s) = slot {
        if let Ok(mut g) = s.lock() {
            *g = Some(RecordedUsage {
                prompt: u.input_tokens,
                completion: u.output_tokens,
                total: u.input_tokens + u.output_tokens,
            });
        }
    }
}

/// Wrap the upstream stream and snoop into `usage`/`message_delta` events so the
/// terminal LogStage can read token counts. The wrapper passes through every
/// event unchanged.
fn tap_openai_usage(
    stream: ai_engine_provider::provider::EventStream<openai::ChatStreamEvent>,
    slot: Option<ai_engine_core::ctx::UsageSlot>,
    metrics: Arc<GatewayMetrics>,
    provider_id: String,
) -> impl futures::Stream<Item = Result<StreamItem, ProviderError>> + Send + 'static {
    // Track completion tokens already credited to the provider so repeated
    // usage events only add the delta.
    let mut credited: u32 = 0;
    stream.map(move |ev| match ev {
        Ok(e) => {
            if let Some(u) = e.raw.get("usage") {
                let prompt = u.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                let completion = u
                    .get("completion_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as u32;
                let total = u
                    .get("total_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or((prompt + completion) as u64) as u32;
                if completion > credited {
                    metrics.add_output(&provider_id, (completion - credited) as u64);
                    credited = completion;
                }
                if let Some(s) = slot.as_ref() {
                    if let Ok(mut g) = s.lock() {
                        *g = Some(RecordedUsage {
                            prompt,
                            completion,
                            total,
                        });
                    }
                }
            }
            Ok(StreamItem::OpenAiChat(e))
        }
        Err(e) => Err(e),
    })
}

fn tap_anthropic_usage(
    stream: ai_engine_provider::provider::EventStream<ai_engine_provider::anthropic::MessagesEvent>,
    slot: Option<ai_engine_core::ctx::UsageSlot>,
    metrics: Arc<GatewayMetrics>,
    provider_id: String,
) -> impl futures::Stream<Item = Result<StreamItem, ProviderError>> + Send + 'static {
    let mut credited: u32 = 0;
    stream.map(move |ev| match ev {
        Ok(e) => {
            // Anthropic emits usage in message_start (input_tokens) and message_delta (output_tokens).
            // Accumulate.
            if let Some(u) = e
                .raw
                .get("usage")
                .or_else(|| e.raw.get("message").and_then(|m| m.get("usage")))
            {
                let output_now = u.get("output_tokens").and_then(|v| v.as_u64()).map(|n| n as u32);
                if let Some(output) = output_now {
                    if output > credited {
                        metrics.add_output(&provider_id, (output - credited) as u64);
                        credited = output;
                    }
                }
                if let Some(s) = slot.as_ref() {
                    if let Ok(mut g) = s.lock() {
                        let current = g.unwrap_or_default();
                        let input = u
                            .get("input_tokens")
                            .and_then(|v| v.as_u64())
                            .map(|n| n as u32)
                            .unwrap_or(current.prompt);
                        let output = output_now.unwrap_or(current.completion);
                        *g = Some(RecordedUsage {
                            prompt: input,
                            completion: output,
                            total: input + output,
                        });
                    }
                }
            }
            Ok(StreamItem::AnthropicMessages(e))
        }
        Err(e) => Err(e),
    })
}
