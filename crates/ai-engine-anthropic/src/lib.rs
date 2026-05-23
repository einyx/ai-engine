//! ai-engine-anthropic
//!
//! Concrete `Provider` for the Anthropic Messages API.

mod client;
mod stream;

use ai_engine_provider::{
    anthropic,
    error::ProviderError,
    provider::{CallCtx, Capabilities, Credentials, EventStream, Provider},
};

pub struct AnthropicProvider {
    id: String,
    base_url: String,
    http: reqwest::Client,
}

impl AnthropicProvider {
    pub fn new(id: String, base_url: impl Into<String>, timeout_secs: u64) -> Self {
        let http = client::build(&client::ClientConfig { timeout_secs });
        Self { id, base_url: base_url.into(), http }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url.trim_end_matches('/'), path)
    }
}

#[async_trait::async_trait]
impl Provider for AnthropicProvider {
    fn id(&self) -> &str { &self.id }
    fn kind(&self) -> &'static str { "anthropic" }
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            chat: false, embeddings: false,
            messages: true, streaming: true,
            tools: true, vision: true,
        }
    }

    async fn messages(
        &self,
        mut req: anthropic::MessagesRequest,
        creds: &Credentials,
        ctx: &CallCtx,
    ) -> Result<anthropic::MessagesResponse, ProviderError> {
        req.model = ctx.upstream_model.clone();
        req.stream = Some(false);
        let resp = self.http
            .post(self.url("/v1/messages"))
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
        resp.json::<anthropic::MessagesResponse>().await
            .map_err(|e| ProviderError::InvalidResponse(e.to_string()))
    }

    async fn messages_stream(
        &self,
        mut req: anthropic::MessagesRequest,
        creds: &Credentials,
        ctx: &CallCtx,
    ) -> Result<EventStream<anthropic::MessagesEvent>, ProviderError> {
        req.model = ctx.upstream_model.clone();
        req.stream = Some(true);
        let resp = self.http
            .post(self.url("/v1/messages"))
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
}

fn map_send_err(e: reqwest::Error) -> ProviderError {
    if e.is_timeout() { ProviderError::Timeout } else { ProviderError::Connect(e.to_string()) }
}
