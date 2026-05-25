//! `CandleProvider`: implements `ai_engine_provider::Provider` for the
//! candle-backed native-quantized local GPU inference path
//! (`kind = "candle-local"`).
//!
//! Supports two backends:
//! - `Backend::Pool`: pool of N replicas (original path, `engine = "pool"`).
//! - `Backend::Paged`: continuous-batching paged engine (default, `engine = "paged"`).

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
use crate::paged::engine::{Engine, EngineConfig, GenRequest};

enum Backend {
    Pool(Arc<ReplicaPool>),
    Paged(Arc<Engine>),
}

pub struct CandleProvider {
    id: String,
    backend: Backend,
    tokenizer: Arc<HfTokenizer>,
    chat_template: Option<String>,
    bos_token: String,
    eos_token: String,
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
        let (_, chat_template, bos_token, eos_token) =
            crate::model::read_gguf_meta(gguf_path, &tokenizer)?;
        let pool = Arc::new(ReplicaPool::new(
            gguf_path,
            device,
            tokenizer.clone(),
            pool_size,
        )?);
        Ok(Self {
            id: id.into(),
            backend: Backend::Pool(pool),
            tokenizer,
            chat_template,
            bos_token,
            eos_token,
        })
    }

    /// Spawn a paged continuous-batching engine for the GGUF at `gguf_path`.
    pub fn new_paged(
        id: impl Into<String>,
        gguf_path: &Path,
        device_spec: &str,
        max_num_seqs: usize,
        block_size: usize,
        kv_cache_blocks: usize,
    ) -> anyhow::Result<Self> {
        let device = crate::device::resolve_device(device_spec)?;
        let tokenizer =
            Arc::new(ai_engine_runtime::load_tokenizer_from_gguf(gguf_path)?);
        let (eos_token_id, chat_template, bos_token, eos_token) =
            crate::model::read_gguf_meta(gguf_path, &tokenizer)?;
        let engine = Engine::spawn(
            gguf_path,
            device,
            EngineConfig { max_num_seqs, block_size, kv_cache_blocks, max_seq: 4096, eos_token_id },
        )?;
        Ok(Self {
            id: id.into(),
            backend: Backend::Paged(engine),
            tokenizer,
            chat_template,
            bos_token,
            eos_token,
        })
    }

    /// Build the prompt for `req` using stored template metadata.
    fn build_prompt(&self, req: &openai::ChatRequest) -> String {
        let msgs = to_template_messages(req);
        match &self.chat_template {
            Some(t) => {
                match crate::template::render_chat_template(t, &msgs, &self.bos_token, &self.eos_token) {
                    Ok(p) => p,
                    Err(e) => {
                        tracing::warn!("chat_template render failed ({e}); falling back to plain prompt");
                        render_prompt(req)
                    }
                }
            }
            None => render_prompt(req),
        }
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
        let prompt = self.build_prompt(&req);
        let params = GenParams {
            max_tokens: req.max_tokens.unwrap_or(256) as usize,
            temperature: req.temperature.unwrap_or(0.0),
        };

        match &self.backend {
            Backend::Pool(pool) => {
                let mut guard = pool.acquire().await;
                let mut prompt_tokens = 0usize;
                let result = tokio::task::block_in_place(|| {
                    let ids = guard.generate(&prompt, &params, |_| {}, &mut prompt_tokens)?;
                    let text = guard.decode(&ids)?;
                    Ok::<_, anyhow::Error>((text, ids.len()))
                });
                drop(guard);
                let (content, completion_tokens) =
                    result.map_err(|e| ProviderError::InvalidResponse(format!("generate: {e}")))?;
                Ok(build_chat_response(&req, ctx, content, prompt_tokens, completion_tokens))
            }
            Backend::Paged(engine) => {
                let prompt_ids = self.tokenizer.encode(&prompt)
                    .map_err(|e| ProviderError::InvalidResponse(format!("tokenize: {e}")))?;
                let prompt_token_count = prompt_ids.len();
                let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
                engine.submit(GenRequest {
                    prompt_ids,
                    max_tokens: params.max_tokens,
                    temperature: params.temperature,
                    tx,
                });
                let mut ids: Vec<u32> = Vec::new();
                while let Some(item) = rx.recv().await {
                    match item {
                        Ok(id) => ids.push(id),
                        Err(e) => return Err(ProviderError::InvalidResponse(format!("paged engine: {e}"))),
                    }
                }
                let completion_tokens = ids.len();
                let content = self.tokenizer.decode(&ids)
                    .map_err(|e| ProviderError::InvalidResponse(format!("decode: {e}")))?;
                Ok(build_chat_response(&req, ctx, content, prompt_token_count, completion_tokens))
            }
        }
    }

    async fn chat_stream(
        &self,
        req: openai::ChatRequest,
        _creds: &Credentials,
        ctx: &CallCtx,
    ) -> Result<EventStream<openai::ChatStreamEvent>, ProviderError> {
        let prompt = self.build_prompt(&req);
        let params = GenParams {
            max_tokens: req.max_tokens.unwrap_or(256) as usize,
            temperature: req.temperature.unwrap_or(0.0),
        };

        let id = format!("chatcmpl-{}", ctx.request_id);
        let model = req.model.clone();

        match &self.backend {
            Backend::Pool(pool) => {
                let pool = pool.clone();
                let tokenizer = self.tokenizer.clone();
                let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Result<String, String>>();

                tokio::spawn(async move {
                    let mut guard = pool.acquire().await;
                    let mut prompt_tokens = 0usize;
                    let result = tokio::task::block_in_place(|| {
                        let mut emitted: Vec<u32> = Vec::new();
                        let mut prev_text = String::new();
                        let mut send_err: Option<String> = None;
                        let ids = guard.generate(
                            &prompt,
                            &params,
                            |tok| {
                                if send_err.is_some() { return; }
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
                        if let Some(e) = send_err { return Err(e); }
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
                                    "id": id, "object": "chat.completion.chunk", "model": model,
                                    "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": serde_json::Value::Null}],
                                });
                                yield Ok(openai::ChatStreamEvent { raw });
                            }
                            Err(e) => { yield Err(ProviderError::Stream(e)); return; }
                        }
                    }
                    let raw = serde_json::json!({
                        "id": id, "object": "chat.completion.chunk", "model": model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    });
                    yield Ok(openai::ChatStreamEvent { raw });
                };
                Ok(Box::pin(stream))
            }
            Backend::Paged(engine) => {
                let prompt_ids = self.tokenizer.encode(&prompt)
                    .map_err(|e| ProviderError::InvalidResponse(format!("tokenize: {e}")))?;
                let tokenizer = self.tokenizer.clone();
                let (token_tx, mut token_rx) = tokio::sync::mpsc::unbounded_channel::<Result<u32, String>>();
                engine.submit(GenRequest {
                    prompt_ids,
                    max_tokens: params.max_tokens,
                    temperature: params.temperature,
                    tx: token_tx,
                });

                let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Result<String, String>>();
                tokio::spawn(async move {
                    let mut emitted: Vec<u32> = Vec::new();
                    let mut prev_text = String::new();
                    while let Some(item) = token_rx.recv().await {
                        match item {
                            Ok(tok) => {
                                emitted.push(tok);
                                match tokenizer.decode(&emitted) {
                                    Ok(full) => {
                                        if full.len() > prev_text.len() {
                                            let suffix = full[prev_text.len()..].to_string();
                                            if !suffix.is_empty() && tx.send(Ok(suffix)).is_err() {
                                                return;
                                            }
                                            prev_text = full;
                                        }
                                    }
                                    Err(e) => {
                                        let _ = tx.send(Err(format!("decode: {e}")));
                                        return;
                                    }
                                }
                            }
                            Err(e) => { let _ = tx.send(Err(e)); return; }
                        }
                    }
                });

                let stream = async_stream::stream! {
                    while let Some(item) = rx.recv().await {
                        match item {
                            Ok(piece) => {
                                let raw = serde_json::json!({
                                    "id": id, "object": "chat.completion.chunk", "model": model,
                                    "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": serde_json::Value::Null}],
                                });
                                yield Ok(openai::ChatStreamEvent { raw });
                            }
                            Err(e) => { yield Err(ProviderError::Stream(e)); return; }
                        }
                    }
                    let raw = serde_json::json!({
                        "id": id, "object": "chat.completion.chunk", "model": model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    });
                    yield Ok(openai::ChatStreamEvent { raw });
                };
                Ok(Box::pin(stream))
            }
        }
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
