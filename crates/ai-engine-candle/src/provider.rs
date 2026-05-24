//! `CandleProvider`: implements `ai_engine_provider::Provider` for the
//! candle-backed native-quantized local GPU inference path
//! (`kind = "candle-local"`).
//!
//! Holds an `Arc<ReplicaPool>` of independently-loaded GGUF replicas plus a
//! shared `Arc<HfTokenizer>`. `chat` acquires a replica and runs the blocking
//! autoregressive loop inside `block_in_place`. `chat_stream` clones the pool
//! and tokenizer into a spawned task that streams per-token deltas over an
//! mpsc channel, mirroring the cluster provider's SSE chunk shape.

use ai_engine_provider::{
    error::ProviderError,
    openai,
    provider::{CallCtx, Capabilities, Credentials, EventStream, Provider},
};
use ai_engine_tokenizer::{HfTokenizer, Tokenizer};
use async_trait::async_trait;
use std::path::Path;
use std::sync::Arc;

use crate::model::GenParams;
use crate::pool::ReplicaPool;

pub struct CandleProvider {
    id: String,
    pool: Arc<ReplicaPool>,
    tokenizer: Arc<HfTokenizer>,
}

impl CandleProvider {
    /// Load `pool_size` replicas of the GGUF at `gguf_path` onto the device
    /// resolved from `device_spec` (e.g. `"cpu"`, `"cuda:0"`, `"metal"`).
    pub fn new(
        id: impl Into<String>,
        gguf_path: &Path,
        device_spec: &str,
        pool_size: usize,
    ) -> anyhow::Result<Self> {
        let device = crate::device::resolve_device(device_spec)?;
        let tokenizer =
            Arc::new(ai_engine_runtime::load_tokenizer_from_gguf(gguf_path)?);
        let pool = Arc::new(ReplicaPool::new(
            gguf_path,
            device,
            tokenizer.clone(),
            pool_size,
        )?);
        Ok(Self {
            id: id.into(),
            pool,
            tokenizer,
        })
    }
}

#[async_trait]
impl Provider for CandleProvider {
    fn id(&self) -> &str {
        &self.id
    }

    fn kind(&self) -> &'static str {
        "candle-local"
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            chat: true,
            streaming: true,
            tools: false,
            vision: false,
            messages: false,
            embeddings: false,
        }
    }

    async fn chat(
        &self,
        req: openai::ChatRequest,
        _creds: &Credentials,
        ctx: &CallCtx,
    ) -> Result<openai::ChatResponse, ProviderError> {
        let params = GenParams {
            max_tokens: req.max_tokens.unwrap_or(256) as usize,
            temperature: req.temperature.unwrap_or(0.0),
        };

        // Generation is blocking CPU/GPU work. Hold the replica guard across
        // the blocking call via `block_in_place` so we don't stall the async
        // runtime's worker thread.
        let mut guard = self.pool.acquire().await;
        let prompt = build_prompt(&guard, &req);
        let mut prompt_tokens = 0usize;
        let result = tokio::task::block_in_place(|| {
            let ids = guard.generate(&prompt, &params, |_| {}, &mut prompt_tokens)?;
            let text = guard.decode(&ids)?;
            Ok::<_, anyhow::Error>((text, ids.len()))
        });
        drop(guard);
        let (content, completion_tokens) =
            result.map_err(|e| ProviderError::InvalidResponse(format!("generate: {e}")))?;

        Ok(build_chat_response(
            &req,
            ctx,
            content,
            prompt_tokens,
            completion_tokens,
        ))
    }

    async fn chat_stream(
        &self,
        req: openai::ChatRequest,
        _creds: &Credentials,
        ctx: &CallCtx,
    ) -> Result<EventStream<openai::ChatStreamEvent>, ProviderError> {
        let params = GenParams {
            max_tokens: req.max_tokens.unwrap_or(256) as usize,
            temperature: req.temperature.unwrap_or(0.0),
        };

        let id = format!("chatcmpl-{}", ctx.request_id);
        let model = req.model.clone();
        let pool = self.pool.clone();
        let tokenizer = self.tokenizer.clone();

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Result<String, String>>();

        // Spawn the generation task. It acquires a replica independently of
        // `&self`'s lifetime, runs the blocking loop, and pushes one decoded
        // piece per token. Byte-level BPE makes single-token decode lossy, so
        // we decode the cumulative id vector with the shared tokenizer (not the
        // guard, which is borrowed mutably by `generate`) and emit only the new
        // suffix.
        tokio::spawn(async move {
            let mut guard = pool.acquire().await;
            let prompt = build_prompt(&guard, &req);
            let mut prompt_tokens = 0usize;
            let result = tokio::task::block_in_place(|| {
                let mut emitted: Vec<u32> = Vec::new();
                let mut prev_text = String::new();
                let mut send_err: Option<String> = None;
                let ids = guard.generate(
                    &prompt,
                    &params,
                    |tok| {
                        if send_err.is_some() {
                            return;
                        }
                        emitted.push(tok);
                        match tokenizer.decode(&emitted) {
                            Ok(full) => {
                                if full.len() > prev_text.len() {
                                    let suffix = full[prev_text.len()..].to_string();
                                    if !suffix.is_empty() && tx.send(Ok(suffix)).is_err() {
                                        send_err = Some("receiver dropped".into());
                                    }
                                    prev_text = full;
                                }
                            }
                            Err(e) => send_err = Some(format!("decode: {e}")),
                        }
                    },
                    &mut prompt_tokens,
                );
                if let Some(e) = send_err {
                    return Err(e);
                }
                ids.map(|_| ()).map_err(|e| format!("generate: {e}"))
            });
            if let Err(e) = result {
                let _ = tx.send(Err(e));
            }
        });

        let stream = async_stream::stream! {
            while let Some(item) = rx.recv().await {
                match item {
                    Ok(piece) => {
                        let raw = serde_json::json!({
                            "id": id,
                            "object": "chat.completion.chunk",
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": { "content": piece },
                                "finish_reason": serde_json::Value::Null,
                            }],
                        });
                        yield Ok(openai::ChatStreamEvent { raw });
                    }
                    Err(e) => {
                        yield Err(ProviderError::Stream(e));
                        return;
                    }
                }
            }
            // End-of-stream sentinel chunk.
            let raw = serde_json::json!({
                "id": id,
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }],
            });
            yield Ok(openai::ChatStreamEvent { raw });
        };

        Ok(Box::pin(stream))
    }
}

fn build_chat_response(
    req: &openai::ChatRequest,
    ctx: &CallCtx,
    content: String,
    prompt_tokens: usize,
    completion_tokens: usize,
) -> openai::ChatResponse {
    openai::ChatResponse {
        id: format!("chatcmpl-{}", ctx.request_id),
        model: req.model.clone(),
        choices: vec![openai::ChatChoice {
            index: 0,
            message: openai::ChatMessage {
                role: "assistant".into(),
                content: openai::ChatContent::Text(content),
                extras: Default::default(),
            },
            finish_reason: Some("stop".into()),
            extras: Default::default(),
        }],
        usage: Some(openai::Usage {
            prompt_tokens: prompt_tokens as u32,
            completion_tokens: completion_tokens as u32,
            total_tokens: (prompt_tokens + completion_tokens) as u32,
        }),
        extras: Default::default(),
    }
}

/// Convert OpenAI chat messages into template messages.
fn to_template_messages(req: &openai::ChatRequest) -> Vec<crate::template::TemplateMessage> {
    req.messages
        .iter()
        .map(|m| {
            let content = match &m.content {
                openai::ChatContent::Text(s) => s.clone(),
                openai::ChatContent::Parts(parts) => parts
                    .iter()
                    .filter_map(|p| p.get("text").and_then(|t| t.as_str()).map(String::from))
                    .collect::<Vec<_>>()
                    .join("\n"),
            };
            crate::template::TemplateMessage {
                role: m.role.clone(),
                content,
            }
        })
        .collect()
}

/// Build the prompt for `req` using `model`'s embedded chat template, falling
/// back to the plain `render_prompt` format when no template is present or
/// rendering fails.
fn build_prompt(model: &crate::model::CandleModel, req: &openai::ChatRequest) -> String {
    let msgs = to_template_messages(req);
    match model.render_with_chat_template(&msgs) {
        Some(Ok(p)) => p,
        Some(Err(e)) => {
            tracing::warn!("chat_template render failed ({e}); falling back to plain prompt");
            render_prompt(req)
        }
        None => render_prompt(req),
    }
}

fn render_prompt(req: &openai::ChatRequest) -> String {
    let mut out = String::new();
    for m in &req.messages {
        let role = &m.role;
        let text = match &m.content {
            openai::ChatContent::Text(s) => s.clone(),
            openai::ChatContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| p.get("text").and_then(|t| t.as_str()).map(String::from))
                .collect::<Vec<_>>()
                .join("\n"),
        };
        out.push_str(role);
        out.push_str(": ");
        out.push_str(&text);
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_prompt_single_user_message() {
        let req = openai::ChatRequest {
            model: "x".into(),
            messages: vec![openai::ChatMessage {
                role: "user".into(),
                content: openai::ChatContent::Text("Hello".into()),
                extras: Default::default(),
            }],
            stream: None,
            temperature: None,
            max_tokens: None,
            stream_options: None,
            extras: Default::default(),
        };
        assert_eq!(render_prompt(&req), "user: Hello\n");
    }
}
