//! ai-engine-openai
//!
//! Concrete `Provider` for OpenAI-shape HTTP APIs. The same impl serves
//! Ollama, vLLM, LM Studio, and OpenRouter — toggled by `base_url` and the
//! presence of credentials.

mod client;
mod stream;

use ai_engine_provider::{
    error::ProviderError,
    openai,
    provider::{CallCtx, Capabilities, Credentials, EventStream, Provider},
};

pub struct OpenAiProvider {
    id: String,
    base_url: String,
    http: reqwest::Client,
}

impl OpenAiProvider {
    pub fn new(id: String, base_url: impl Into<String>, timeout_secs: u64, http2: bool) -> Self {
        let http = client::build(&client::ClientConfig { timeout_secs, http2 });
        Self { id, base_url: base_url.into(), http }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url.trim_end_matches('/'), path)
    }
}

#[async_trait::async_trait]
impl Provider for OpenAiProvider {
    fn id(&self) -> &str { &self.id }
    fn kind(&self) -> &'static str { "openai" }
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            chat: true, embeddings: true, streaming: true,
            tools: true, vision: true, messages: false,
        }
    }

    async fn chat(
        &self,
        mut req: openai::ChatRequest,
        creds: &Credentials,
        ctx: &CallCtx,
    ) -> Result<openai::ChatResponse, ProviderError> {
        req.model = ctx.upstream_model.clone();
        req.stream = Some(false);
        let resp = self.http
            .post(self.url("/chat/completions"))
            .headers(client::auth_headers(creds, &[]))
            .json(&req)
            .send()
            .await
            .map_err(map_send_err)?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.bytes().await.unwrap_or_default();
            return Err(ProviderError::Status { status: status.as_u16(), body });
        }
        resp.json::<openai::ChatResponse>().await
            .map_err(|e| ProviderError::InvalidResponse(e.to_string()))
    }

    async fn chat_stream(
        &self,
        mut req: openai::ChatRequest,
        creds: &Credentials,
        ctx: &CallCtx,
    ) -> Result<EventStream<openai::ChatStreamEvent>, ProviderError> {
        req.model = ctx.upstream_model.clone();
        req.stream = Some(true);
        let mut opts = req.stream_options.clone().unwrap_or_default();
        if opts.include_usage.is_none() {
            opts.include_usage = Some(true);
        }
        req.stream_options = Some(opts);

        let resp = self.http
            .post(self.url("/chat/completions"))
            .headers(client::auth_headers(creds, &[("accept".into(), "text/event-stream".into())]))
            .json(&req)
            .send()
            .await
            .map_err(map_send_err)?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.bytes().await.unwrap_or_default();
            return Err(ProviderError::Status { status: status.as_u16(), body });
        }
        Ok(Box::pin(stream::parse(resp.bytes_stream())))
    }

    async fn embeddings(
        &self,
        mut req: openai::EmbeddingsRequest,
        creds: &Credentials,
        ctx: &CallCtx,
    ) -> Result<openai::EmbeddingsResponse, ProviderError> {
        req.model = ctx.upstream_model.clone();
        let resp = self.http
            .post(self.url("/embeddings"))
            .headers(client::auth_headers(creds, &[]))
            .json(&req)
            .send()
            .await
            .map_err(map_send_err)?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.bytes().await.unwrap_or_default();
            return Err(ProviderError::Status { status: status.as_u16(), body });
        }
        resp.json::<openai::EmbeddingsResponse>().await
            .map_err(|e| ProviderError::InvalidResponse(e.to_string()))
    }
}

fn map_send_err(e: reqwest::Error) -> ProviderError {
    if e.is_timeout() {
        ProviderError::Timeout
    } else {
        ProviderError::Connect(e.to_string())
    }
}
