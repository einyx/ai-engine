//! `ClusterProvider`: implements `ai_engine_provider::Provider` so the
//! existing gateway pipeline can route requests at a cluster without any
//! trait changes.
//!
//! Two modes:
//! - **Leader**: holds an `Arc<Mutex<LeaderState>>` and drives the cluster
//!   autoregressive generation loop from `chat`.
//! - **Worker**: never receives inbound HTTP, so every Provider method
//!   returns `ProviderError::Unsupported`. The cluster member still exposes
//!   a Provider so that the same binary configuration shape works for both
//!   roles — the worker just refuses application traffic.

use ai_engine_provider::{
    error::ProviderError,
    openai,
    provider::{CallCtx, Capabilities, Credentials, Provider},
};
use ai_engine_runtime::config::ModelConfig;
use ai_engine_runtime::sample::SamplingConfig;
use ai_engine_tokenizer::{HfTokenizer, Tokenizer};
use async_trait::async_trait;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::leader::ClusterLeader;

/// Whether this provider runs as the cluster leader (drives requests) or as
/// a worker (advertises capabilities but refuses application traffic).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    Leader,
    Worker,
}

/// Live leader-side state attached to a `ClusterProvider` in leader mode.
/// Owns the `ClusterLeader` plus everything required to drive a generation
/// request end-to-end (model config, tokenizer, weights path, leader's
/// layer range).
pub struct LeaderState {
    pub leader: ClusterLeader,
    pub model_cfg: ModelConfig,
    pub model_path: PathBuf,
    pub tokenizer: HfTokenizer,
    pub leader_layers: std::ops::Range<usize>,
}

pub struct ClusterProvider {
    id: String,
    role: Role,
    /// Live cluster handle, populated for `new_leader_with_state`. The
    /// `chat` impl drives the autoregressive loop through this. `None` for
    /// worker mode and for the `stub_leader` test helper.
    state: Option<Arc<Mutex<LeaderState>>>,
}

impl ClusterProvider {
    /// Production constructor: leader mode with live cluster state.
    pub fn new_leader_with_state(
        id: impl Into<String>,
        state: Arc<Mutex<LeaderState>>,
    ) -> Self {
        Self {
            id: id.into(),
            role: Role::Leader,
            state: Some(state),
        }
    }

    /// Production constructor: worker mode (no state; every Provider method
    /// returns `Unsupported`).
    pub fn new_worker(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            role: Role::Worker,
            state: None,
        }
    }

    /// Test helper: leader mode without live cluster state. Used by the
    /// trait-impl smoke test; production code uses `new_leader_with_state`.
    pub fn stub_leader(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            role: Role::Leader,
            state: None,
        }
    }

    /// Test helper: worker mode (equivalent to `new_worker`, named
    /// consistently with `stub_leader`).
    pub fn stub_worker(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            role: Role::Worker,
            state: None,
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

        let mut st = state.lock().await;
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
