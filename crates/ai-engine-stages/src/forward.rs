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

use crate::ProviderRegistry;

pub struct ForwardStage {
    pub providers: Arc<ProviderRegistry>,
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

        let (provider, default_creds) =
            self.providers
                .get(&binding.provider_id)
                .ok_or_else(|| StageError {
                    stage,
                    error: GatewayError::Internal(anyhow::anyhow!(
                        "forward: unknown provider {}",
                        binding.provider_id
                    )),
                })?;

        // Credentials selection: passthrough mode prefers raw_bearer, shared-key uses
        // the provider default. Identity::Holder users get the provider default.
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
            upstream_model: binding.upstream_model.clone(),
        };

        // Take ownership of the request body so we can move it into the provider call.
        // Replace with `Empty` to keep `ctx.body` valid for later stages.
        let body = std::mem::replace(&mut ctx.body, RequestBody::Empty);

        match body {
            RequestBody::OpenAiChat(req) => {
                let streaming = req.stream.unwrap_or(false);
                if streaming {
                    let stream = provider
                        .chat_stream(req, &creds, &call_ctx)
                        .await
                        .map_err(|e| stage_err(stage, e))?;
                    let tapped = tap_openai_usage(stream, ctx.usage_slot.clone());
                    ctx.response = ResponseSlot::Stream(Box::pin(tapped));
                } else {
                    let resp = provider
                        .chat(req, &creds, &call_ctx)
                        .await
                        .map_err(|e| stage_err(stage, e))?;
                    if let Some(u) = resp.usage.as_ref() {
                        record_openai_usage(&ctx.usage_slot, u);
                    }
                    let body = serde_json::to_vec(&resp).map_err(|e| StageError {
                        stage,
                        error: GatewayError::Internal(anyhow::Error::new(e)),
                    })?;
                    ctx.response = ResponseSlot::Full(GatewayResponse {
                        status: 200,
                        headers: HeaderMap::new(),
                        body: body.into(),
                    });
                }
            }
            RequestBody::AnthropicMessages(req) => {
                let streaming = req.stream.unwrap_or(false);
                if streaming {
                    let stream = provider
                        .messages_stream(req, &creds, &call_ctx)
                        .await
                        .map_err(|e| stage_err(stage, e))?;
                    let tapped = tap_anthropic_usage(stream, ctx.usage_slot.clone());
                    ctx.response = ResponseSlot::Stream(Box::pin(tapped));
                } else {
                    let resp = provider
                        .messages(req, &creds, &call_ctx)
                        .await
                        .map_err(|e| stage_err(stage, e))?;
                    record_anthropic_usage(&ctx.usage_slot, &resp.usage);
                    let body = serde_json::to_vec(&resp).map_err(|e| StageError {
                        stage,
                        error: GatewayError::Internal(anyhow::Error::new(e)),
                    })?;
                    ctx.response = ResponseSlot::Full(GatewayResponse {
                        status: 200,
                        headers: HeaderMap::new(),
                        body: body.into(),
                    });
                }
            }
            RequestBody::OpenAiEmbeddings(req) => {
                let resp = provider
                    .embeddings(req, &creds, &call_ctx)
                    .await
                    .map_err(|e| stage_err(stage, e))?;
                if let Some(u) = resp.usage.as_ref() {
                    record_openai_usage(&ctx.usage_slot, u);
                }
                let body = serde_json::to_vec(&resp).map_err(|e| StageError {
                    stage,
                    error: GatewayError::Internal(anyhow::Error::new(e)),
                })?;
                ctx.response = ResponseSlot::Full(GatewayResponse {
                    status: 200,
                    headers: HeaderMap::new(),
                    body: body.into(),
                });
            }
            RequestBody::Empty => {
                // Nothing to forward (e.g. /v1/models, /healthz). Skip.
            }
        }
        Ok(StageOutcome::Continue)
    }
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
) -> impl futures::Stream<Item = Result<StreamItem, ProviderError>> + Send + 'static {
    stream.map(move |ev| match ev {
        Ok(e) => {
            if let Some(u) = e.raw.get("usage") {
                if let Some(s) = slot.as_ref() {
                    if let Ok(mut g) = s.lock() {
                        let prompt =
                            u.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                        let completion = u
                            .get("completion_tokens")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0) as u32;
                        let total = u
                            .get("total_tokens")
                            .and_then(|v| v.as_u64())
                            .unwrap_or((prompt + completion) as u64)
                            as u32;
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
) -> impl futures::Stream<Item = Result<StreamItem, ProviderError>> + Send + 'static {
    stream.map(move |ev| match ev {
        Ok(e) => {
            // Anthropic emits usage in message_start (input_tokens) and message_delta (output_tokens).
            // Accumulate.
            if let Some(u) = e
                .raw
                .get("usage")
                .or_else(|| e.raw.get("message").and_then(|m| m.get("usage")))
            {
                if let Some(s) = slot.as_ref() {
                    if let Ok(mut g) = s.lock() {
                        let current = g.unwrap_or_default();
                        let input = u
                            .get("input_tokens")
                            .and_then(|v| v.as_u64())
                            .map(|n| n as u32)
                            .unwrap_or(current.prompt);
                        let output = u
                            .get("output_tokens")
                            .and_then(|v| v.as_u64())
                            .map(|n| n as u32)
                            .unwrap_or(current.completion);
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
