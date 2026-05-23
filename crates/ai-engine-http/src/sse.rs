use ai_engine_core::ctx::StreamItem;
use ai_engine_provider::error::ProviderError;
use axum::response::sse::Event;
use futures::stream::{BoxStream, Stream, StreamExt};
use std::convert::Infallible;

/// Adapt a `ResponseSlot::Stream` into an axum SSE event stream.
///
/// Differences from the upstream wire SSE we received:
/// - We re-emit each event with axum's `Event::default().data(...)`, which
///   produces canonical `data: <line>\n\n` framing.
/// - For Anthropic events, we set `event: <type>` from the JSON's `type` field.
/// - For OpenAI streams, we append a trailing `data: [DONE]\n\n` since the
///   upstream provider parser strips it but real SDKs require it.
/// - On stream error, we emit a final `event: error` frame, then close.
pub fn encode_openai(
    inner: BoxStream<'static, Result<StreamItem, ProviderError>>,
) -> impl Stream<Item = Result<Event, Infallible>> + Send + 'static {
    async_stream_emit(inner, EncodeKind::OpenAi)
}

pub fn encode_anthropic(
    inner: BoxStream<'static, Result<StreamItem, ProviderError>>,
) -> impl Stream<Item = Result<Event, Infallible>> + Send + 'static {
    async_stream_emit(inner, EncodeKind::Anthropic)
}

#[derive(Clone, Copy)]
enum EncodeKind {
    OpenAi,
    Anthropic,
}

fn async_stream_emit(
    inner: BoxStream<'static, Result<StreamItem, ProviderError>>,
    kind: EncodeKind,
) -> impl Stream<Item = Result<Event, Infallible>> + Send + 'static {
    let inner_with_done = inner
        .map(Some)
        .chain(futures::stream::once(async { None }));
    inner_with_done.map(move |maybe_item| {
        Ok(match maybe_item {
            Some(Ok(StreamItem::OpenAiChat(ev))) => Event::default().data(ev.raw.to_string()),
            Some(Ok(StreamItem::AnthropicMessages(ev))) => {
                let ty = ev
                    .raw
                    .get("type")
                    .and_then(|t| t.as_str())
                    .unwrap_or("event")
                    .to_string();
                Event::default().event(ty).data(ev.raw.to_string())
            }
            Some(Err(e)) => Event::default()
                .event("error")
                .data(serde_json::json!({ "message": e.to_string() }).to_string()),
            None => match kind {
                // OpenAI requires a [DONE] sentinel; Anthropic stream ends naturally.
                EncodeKind::OpenAi => Event::default().data("[DONE]"),
                EncodeKind::Anthropic => Event::default().comment("done"),
            },
        })
    })
}
