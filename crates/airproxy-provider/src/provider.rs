use crate::{anthropic, error::ProviderError, openai};
use futures::stream::BoxStream;
use std::time::Instant;
use uuid::Uuid;

#[derive(Debug, Clone, Default)]
pub struct Capabilities {
    pub chat: bool,
    pub messages: bool,
    pub embeddings: bool,
    pub streaming: bool,
    pub tools: bool,
    pub vision: bool,
}

/// Per-request credentials passed to a provider call.
///
/// - `api_key` is the provider-default key from `[[provider]].api_key` in config.
///   Optional because Ollama, vLLM, LM Studio, etc. typically have no key set.
/// - `raw_bearer` is the unmodified `Authorization: Bearer …` from the inbound
///   request, populated only in passthrough auth mode.
/// - `extra_headers` are static per-provider headers like `anthropic-version`.
#[derive(Debug, Clone, Default)]
pub struct Credentials {
    pub api_key: Option<String>,
    pub raw_bearer: Option<String>,
    pub extra_headers: Vec<(String, String)>,
}

impl Credentials {
    pub fn none() -> Self {
        Self::default()
    }
}

pub struct CallCtx {
    pub request_id: Uuid,
    pub deadline: Option<Instant>,
    pub upstream_model: String,
}

pub type EventStream<T> = BoxStream<'static, Result<T, ProviderError>>;

#[async_trait::async_trait]
pub trait Provider: Send + Sync + 'static {
    fn id(&self) -> &str;
    fn kind(&self) -> &'static str;
    fn capabilities(&self) -> Capabilities;

    async fn chat(
        &self,
        req: openai::ChatRequest,
        creds: &Credentials,
        ctx: &CallCtx,
    ) -> Result<openai::ChatResponse, ProviderError> {
        let _ = (req, creds, ctx);
        Err(ProviderError::Unsupported)
    }

    async fn chat_stream(
        &self,
        req: openai::ChatRequest,
        creds: &Credentials,
        ctx: &CallCtx,
    ) -> Result<EventStream<openai::ChatStreamEvent>, ProviderError> {
        let _ = (req, creds, ctx);
        Err(ProviderError::Unsupported)
    }

    async fn messages(
        &self,
        req: anthropic::MessagesRequest,
        creds: &Credentials,
        ctx: &CallCtx,
    ) -> Result<anthropic::MessagesResponse, ProviderError> {
        let _ = (req, creds, ctx);
        Err(ProviderError::Unsupported)
    }

    async fn messages_stream(
        &self,
        req: anthropic::MessagesRequest,
        creds: &Credentials,
        ctx: &CallCtx,
    ) -> Result<EventStream<anthropic::MessagesEvent>, ProviderError> {
        let _ = (req, creds, ctx);
        Err(ProviderError::Unsupported)
    }

    async fn embeddings(
        &self,
        req: openai::EmbeddingsRequest,
        creds: &Credentials,
        ctx: &CallCtx,
    ) -> Result<openai::EmbeddingsResponse, ProviderError> {
        let _ = (req, creds, ctx);
        Err(ProviderError::Unsupported)
    }
}
