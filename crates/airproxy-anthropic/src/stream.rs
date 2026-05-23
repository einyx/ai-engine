use airproxy_provider::{anthropic::MessagesEvent, error::ProviderError};
use bytes::Bytes;
use futures::stream::{Stream, StreamExt};
use serde_json::Value;

/// Parse a byte stream of Anthropic SSE frames into typed events.
///
/// Each SSE frame contains both an `event: <name>` line and a `data: {...}` line.
/// We only parse the data line; the event type is also present inside the JSON
/// as the `type` field. The stream ends when the upstream connection closes
/// (after `message_stop`) — there is no `[DONE]` sentinel.
pub fn parse(
    byte_stream: impl Stream<Item = reqwest::Result<Bytes>> + Send + 'static,
) -> impl Stream<Item = Result<MessagesEvent, ProviderError>> + Send + 'static {
    async_stream::stream! {
        let mut buf: Vec<u8> = Vec::with_capacity(4096);
        let mut byte_stream = Box::pin(byte_stream);
        while let Some(chunk) = byte_stream.next().await {
            let chunk = match chunk {
                Ok(b) => b,
                Err(e) => { yield Err(ProviderError::Stream(e.to_string())); return; }
            };
            buf.extend_from_slice(&chunk);
            while let Some(idx) = find_frame_end(&buf) {
                let frame: Vec<u8> = buf.drain(..idx + 2).collect();
                let frame = match std::str::from_utf8(&frame) {
                    Ok(s) => s,
                    Err(e) => { yield Err(ProviderError::Stream(e.to_string())); return; }
                };
                for line in frame.lines() {
                    // Anthropic frames carry both `event: <name>` and `data: {...}` lines.
                    // We only parse the data line; the `type` field is in the JSON.
                    let Some(data) = line.strip_prefix("data: ").or_else(|| line.strip_prefix("data:")) else { continue; };
                    let data = data.trim_start();
                    if data.is_empty() { continue; }
                    match serde_json::from_str::<Value>(data) {
                        Ok(raw) => yield Ok(MessagesEvent { raw }),
                        Err(e) => { yield Err(ProviderError::InvalidResponse(e.to_string())); return; }
                    }
                }
            }
        }
    }
}

fn find_frame_end(b: &[u8]) -> Option<usize> {
    b.windows(2).position(|w| w == b"\n\n")
}
