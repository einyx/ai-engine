use airproxy_provider::{anthropic, error::ProviderError, openai};
use bytes::Bytes;
use futures::stream::BoxStream;
use http::HeaderMap;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use uuid::Uuid;

use crate::error::GatewayError;

/// Recognized request payloads. Each variant maps to one route family.
pub enum RequestBody {
    OpenAiChat(openai::ChatRequest),
    AnthropicMessages(anthropic::MessagesRequest),
    OpenAiEmbeddings(openai::EmbeddingsRequest),
    /// `/v1/models`, `/healthz`, `/readyz`.
    Empty,
}

#[derive(Debug, Default)]
pub struct GatewayResponse {
    pub status: u16,            // 0 means "default 200"
    pub headers: HeaderMap,
    pub body: Bytes,
}

/// A streaming event flowing back to the client.
/// Tagged because terminal stages may peek at one or the other shape.
pub enum StreamItem {
    OpenAiChat(openai::ChatStreamEvent),
    AnthropicMessages(anthropic::MessagesEvent),
}

#[derive(Default)]
pub enum ResponseSlot {
    #[default]
    Pending,
    Full(GatewayResponse),
    Stream(BoxStream<'static, Result<StreamItem, ProviderError>>),
}

pub enum Identity {
    Anonymous { raw_bearer: Option<String> },
    Holder { name: String },
}

pub struct ProviderBinding {
    pub provider_id: String,
    pub upstream_model: String,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RecordedUsage {
    pub prompt: u32,
    pub completion: u32,
    pub total: u32,
}

pub type UsageSlot = Arc<Mutex<Option<RecordedUsage>>>;

pub struct RequestCtx {
    pub request_id: Uuid,
    pub started_at: Instant,
    pub route: &'static str,
    pub headers: HeaderMap,
    pub raw_body_len: usize,
    pub body: RequestBody,
    pub identity: Option<Identity>,
    pub binding: Option<ProviderBinding>,
    pub response: ResponseSlot,
    pub error: Option<GatewayError>,
    /// Filled by `ForwardStage` (it's the one that can tap streams); read by `LogStage`.
    pub usage_slot: Option<UsageSlot>,
    pub metadata: HashMap<&'static str, Value>,
}

impl RequestCtx {
    pub fn new(
        route: &'static str,
        headers: HeaderMap,
        raw_body_len: usize,
        body: RequestBody,
    ) -> Self {
        Self {
            request_id: Uuid::now_v7(),
            started_at: Instant::now(),
            route,
            headers,
            raw_body_len,
            body,
            identity: None,
            binding: None,
            response: ResponseSlot::Pending,
            error: None,
            usage_slot: None,
            metadata: HashMap::new(),
        }
    }
}
