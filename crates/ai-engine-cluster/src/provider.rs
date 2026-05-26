//! `ClusterProvider`: implements `ai_engine_provider::Provider` so the
//! existing gateway pipeline can route requests at a cluster without any
//! trait changes.
//!
//! Two modes:
//! - **Leader**: holds an `Arc<LeaderState>` and drives the cluster
//!   autoregressive generation loop from `chat`. (Plan 4: no Mutex —
//!   `ClusterLeader::generate` is `&self`, supporting concurrent requests.)
//! - **Worker**: never receives inbound HTTP, so every Provider method
//!   returns `ProviderError::Unsupported`. The cluster member still exposes
//!   a Provider so that the same binary configuration shape works for both
//!   roles — the worker just refuses application traffic.

use ai_engine_provider::{
    error::ProviderError,
    openai,
    provider::{CallCtx, Capabilities, Credentials, EventStream, Provider},
};
use ai_engine_runtime::config::ModelConfig;
use ai_engine_runtime::sample::SamplingConfig;
use ai_engine_tokenizer::{HfTokenizer, Tokenizer};
use async_trait::async_trait;
use std::path::PathBuf;
use std::sync::Arc;

use crate::coordinator::Coordinator;
use crate::leader::ClusterLeader;

/// Live coordinator-side state for a leaderless (Phase B) `ClusterProvider`.
/// Holds the mesh-backed `Coordinator` for this node plus the tokenizer used
/// to bridge chat text ↔ token ids. The node's hosted stages are served to
/// peers by a separate `serve_peer` loop spawned at startup.
pub struct CoordinatorState {
    pub coordinator: Arc<Coordinator>,
    pub tokenizer: Arc<HfTokenizer>,
}

/// Whether this provider runs as the cluster leader (drives requests) or as
/// a worker (advertises capabilities but refuses application traffic).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    Leader,
    Worker,
    /// Leaderless p2p: this node ingests requests and drives them over the mesh
    /// via a local `Coordinator`.
    Coordinator,
}

/// Live leader-side state attached to a `ClusterProvider` in leader mode.
/// Owns the `ClusterLeader` plus everything required to drive a generation
/// request end-to-end (model config, tokenizer, weights path, leader's
/// layer range).
///
/// Plan 4 Task 4: the leader is held as `Arc<ClusterLeader>` (shareable
/// into spawned generation tasks) and the tokenizer as `Arc<HfTokenizer>`
/// (cheap clone for the same reason). `ClusterLeader::generate` is now
/// `&self`, so `LeaderState` itself no longer needs a `Mutex`.
pub struct LeaderState {
    pub leader: Arc<ClusterLeader>,
    pub model_cfg: ModelConfig,
    pub model_path: PathBuf,
    pub tokenizer: Arc<HfTokenizer>,
    pub leader_layers: std::ops::Range<usize>,
}

pub struct ClusterProvider {
    id: String,
    role: Role,
    /// Live cluster handle, populated for `new_leader_with_state`. The
    /// `chat` impl drives the autoregressive loop through this. `None` for
    /// worker mode and for the `stub_leader` test helper.
    state: Option<Arc<LeaderState>>,
    /// Live coordinator state, populated for `new_coordinator` (leaderless
    /// mode). The `chat` impl drives the mesh forward pass through this.
    coordinator: Option<Arc<CoordinatorState>>,
}

impl ClusterProvider {
    /// Production constructor: leader mode with live cluster state.
    pub fn new_leader_with_state(
        id: impl Into<String>,
        state: Arc<LeaderState>,
    ) -> Self {
        Self {
            id: id.into(),
            role: Role::Leader,
            state: Some(state),
            coordinator: None,
        }
    }

    /// Production constructor: leaderless coordinator mode. Drives requests
    /// over the full mesh via a local `Coordinator`.
    pub fn new_coordinator(
        id: impl Into<String>,
        state: Arc<CoordinatorState>,
    ) -> Self {
        Self {
            id: id.into(),
            role: Role::Coordinator,
            state: None,
            coordinator: Some(state),
        }
    }

    /// Production constructor: worker mode (no state; every Provider method
    /// returns `Unsupported`).
    pub fn new_worker(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            role: Role::Worker,
            state: None,
            coordinator: None,
        }
    }

    /// Test helper: leader mode without live cluster state. Used by the
    /// trait-impl smoke test; production code uses `new_leader_with_state`.
    pub fn stub_leader(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            role: Role::Leader,
            state: None,
            coordinator: None,
        }
    }

    /// Test helper: worker mode (equivalent to `new_worker`, named
    /// consistently with `stub_leader`).
    pub fn stub_worker(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            role: Role::Worker,
            state: None,
            coordinator: None,
        }
    }
}

#[async_trait]
impl Provider for ClusterProvider {
    fn id(&self) -> &str {
        &self.id
    }

    fn kind(&self) -> &'static str {
        "local-cluster"
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
        if self.role == Role::Worker {
            return Err(ProviderError::Unsupported);
        }

        // Leaderless p2p: drive the request over the mesh via the local
        // Coordinator. Mirrors the leader path's token-ids bridging.
        if self.role == Role::Coordinator {
            let cs = self.coordinator.as_ref().ok_or_else(|| {
                ProviderError::InvalidResponse("coordinator provider has no state".into())
            })?;
            let prompt = render_prompt(&req);
            let max_tokens = req.max_tokens.unwrap_or(256) as usize;
            let sampling = SamplingConfig {
                temperature: req.temperature.unwrap_or(1.0),
                top_p: None,
                top_k: None,
                seed: ctx.request_id.as_u128() as u64,
            };
            let prompt_ids: Vec<u32> = cs
                .tokenizer
                .encode(&prompt)
                .map_err(|e| ProviderError::InvalidResponse(format!("tokenize: {e}")))?;
            let prompt_ids_i32: Vec<i32> = prompt_ids.iter().map(|x| *x as i32).collect();
            let tokens = cs
                .coordinator
                .generate(&prompt_ids_i32, max_tokens, sampling)
                .await
                .map_err(|e| ProviderError::InvalidResponse(format!("generate: {e}")))?;
            let content = cs
                .tokenizer
                .decode(&tokens)
                .map_err(|e| ProviderError::InvalidResponse(format!("decode: {e}")))?;
            return Ok(openai::ChatResponse {
                id: format!("chatcmpl-{}", ctx.request_id),
                model: req.model,
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
                    prompt_tokens: prompt_ids.len() as u32,
                    completion_tokens: tokens.len() as u32,
                    total_tokens: (prompt_ids.len() + tokens.len()) as u32,
                }),
                extras: Default::default(),
            });
        }

        let state = self.state.as_ref().ok_or_else(|| {
            ProviderError::InvalidResponse("cluster provider has no leader state".into())
        })?;

        // Render chat messages as a flat prompt. v0.2 doesn't apply chat
        // templates — that's deferred. We concatenate role+content with
        // newlines, which matches what most local models accept for
        // completion-style use.
        let prompt = render_prompt(&req);

        let max_tokens = req.max_tokens.unwrap_or(256) as usize;
        let sampling = SamplingConfig {
            temperature: req.temperature.unwrap_or(1.0),
            top_p: None,
            top_k: None,
            seed: ctx.request_id.as_u128() as u64,
        };

        let st = state.as_ref();
        let prompt_ids: Vec<u32> = st
            .tokenizer
            .encode(&prompt)
            .map_err(|e| ProviderError::InvalidResponse(format!("tokenize: {e}")))?;
        let prompt_ids_i32: Vec<i32> = prompt_ids.iter().map(|x| *x as i32).collect();

        // v0.2 wires NdArray only — multi-backend dispatch is deferred.
        let leader_layers = st.leader_layers.clone();
        let model_path = st.model_path.clone();
        let model_cfg = st.model_cfg.clone();
        let tokens = st
            .leader
            .generate::<burn_ndarray::NdArray>(
                &model_path,
                &model_cfg,
                leader_layers,
                &prompt_ids_i32,
                max_tokens,
                sampling,
            )
            .await
            .map_err(|e| ProviderError::InvalidResponse(format!("generate: {e}")))?;

        let content = st
            .tokenizer
            .decode(&tokens)
            .map_err(|e| ProviderError::InvalidResponse(format!("decode: {e}")))?;

        Ok(openai::ChatResponse {
            id: format!("chatcmpl-{}", ctx.request_id),
            model: req.model,
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
                prompt_tokens: prompt_ids.len() as u32,
                completion_tokens: tokens.len() as u32,
                total_tokens: (prompt_ids.len() + tokens.len()) as u32,
            }),
            extras: Default::default(),
        })
    }

    async fn chat_stream(
        &self,
        req: openai::ChatRequest,
        _creds: &Credentials,
        ctx: &CallCtx,
    ) -> Result<EventStream<openai::ChatStreamEvent>, ProviderError> {
        if self.role == Role::Worker {
            return Err(ProviderError::Unsupported);
        }

        // Leaderless p2p: drive the mesh and stream tokens incrementally as
        // they're produced (true per-token streaming). Re-decode the full
        // emitted prefix each step and emit only the new suffix, so multi-token
        // unicode flushes correctly.
        if self.role == Role::Coordinator {
            let cs = self.coordinator.as_ref().ok_or_else(|| {
                ProviderError::InvalidResponse("coordinator provider has no state".into())
            })?;
            let prompt = render_prompt(&req);
            let max_tokens = req.max_tokens.unwrap_or(256) as usize;
            let sampling = SamplingConfig {
                temperature: req.temperature.unwrap_or(1.0),
                top_p: None,
                top_k: None,
                seed: ctx.request_id.as_u128() as u64,
            };
            let prompt_ids: Vec<u32> = cs
                .tokenizer
                .encode(&prompt)
                .map_err(|e| ProviderError::InvalidResponse(format!("tokenize: {e}")))?;
            let prompt_ids_i32: Vec<i32> = prompt_ids.iter().map(|x| *x as i32).collect();
            let tokenizer = cs.tokenizer.clone();
            let id = format!("chatcmpl-{}", ctx.request_id);
            let model = req.model.clone();
            let mut rx = cs
                .coordinator
                .clone()
                .generate_stream(prompt_ids_i32, max_tokens, sampling);

            let stream = async_stream::stream! {
                let mut emitted: Vec<u32> = Vec::new();
                let mut prev = String::new();
                while let Some(item) = rx.recv().await {
                    let tok = match item {
                        Ok(t) => t,
                        Err(e) => {
                            yield Err(ProviderError::InvalidResponse(format!("generate: {e}")));
                            return;
                        }
                    };
                    emitted.push(tok);
                    let full = match tokenizer.decode(&emitted) {
                        Ok(s) => s,
                        Err(e) => {
                            yield Err(ProviderError::InvalidResponse(format!("decode: {e}")));
                            return;
                        }
                    };
                    if full.len() > prev.len() {
                        let suffix = full[prev.len()..].to_string();
                        prev = full;
                        if !suffix.is_empty() {
                            let raw = serde_json::json!({
                                "id": id,
                                "object": "chat.completion.chunk",
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": { "content": suffix },
                                    "finish_reason": serde_json::Value::Null,
                                }],
                            });
                            yield Ok(openai::ChatStreamEvent { raw });
                        }
                    }
                }
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
            return Ok(Box::pin(stream));
        }

        let state = self.state.as_ref().ok_or_else(|| {
            ProviderError::InvalidResponse("cluster provider has no leader state".into())
        })?;

        let prompt = render_prompt(&req);
        let max_tokens = req.max_tokens.unwrap_or(256) as usize;
        let sampling = SamplingConfig {
            temperature: req.temperature.unwrap_or(1.0),
            top_p: None,
            top_k: None,
            seed: ctx.request_id.as_u128() as u64,
        };

        let st = state.clone();
        let prompt_ids: Vec<u32> = st
            .tokenizer
            .encode(&prompt)
            .map_err(|e| ProviderError::InvalidResponse(format!("tokenize: {e}")))?;
        let prompt_ids_i32: Vec<i32> = prompt_ids.iter().map(|x| *x as i32).collect();

        let leader_layers = st.leader_layers.clone();
        let model_path = st.model_path.clone();
        let model_cfg = st.model_cfg.clone();

        // Spawn the generation task; it pushes tokens onto an mpsc channel.
        let mut rx = st.leader.clone().generate_stream::<burn_ndarray::NdArray>(
            &model_path,
            &model_cfg,
            leader_layers,
            &prompt_ids_i32,
            max_tokens,
            sampling,
        );

        let id = format!("chatcmpl-{}", ctx.request_id);
        let model = req.model.clone();
        let tokenizer = st.tokenizer.clone();

        let stream = async_stream::stream! {
            while let Some(item) = rx.recv().await {
                match item {
                    Ok(token) => {
                        let piece = match tokenizer.decode(&[token]) {
                            Ok(s) => s,
                            Err(e) => {
                                yield Err(ProviderError::InvalidResponse(format!(
                                    "decode: {e}"
                                )));
                                return;
                            }
                        };
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
                        yield Err(ProviderError::InvalidResponse(format!(
                            "generate_stream: {e}"
                        )));
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
