use ai_engine_provider::{error::ProviderError, openai::ChatStreamEvent};
use bytes::Bytes;
use futures::stream::{Stream, StreamExt};
use serde_json::Value;

/// Parse a byte stream of SSE frames into typed events. Terminates on `data: [DONE]`.
pub fn parse(
    byte_stream: impl Stream<Item = reqwest::Result<Bytes>> + Send + 'static,
) -> impl Stream<Item = Result<ChatStreamEvent, ProviderError>> + Send + 'static {
    async_stream::stream! {
        let mut buf: Vec<u8> = Vec::with_capacity(4096);
        let mut byte_stream = Box::pin(byte_stream);
        while let Some(chunk) = byte_stream.next().await {
            let chunk = match chunk {
                Ok(b) => b,
                Err(e) => { yield Err(ProviderError::Stream(e.to_string())); return; }
            };
            buf.extend_from_slice(&chunk);
            // Process completed frames (terminated by \n\n)
            while let Some(idx) = find_frame_end(&buf) {
                let frame: Vec<u8> = buf.drain(..idx + 2).collect();
                let frame = match std::str::from_utf8(&frame) {
                    Ok(s) => s,
                    Err(e) => { yield Err(ProviderError::Stream(e.to_string())); return; }
                };
                for line in frame.lines() {
                    let Some(data) = line.strip_prefix("data: ").or_else(|| line.strip_prefix("data:")) else { continue; };
                    let data = data.trim_start();
                    if data == "[DONE]" { return; }
                    if data.is_empty() { continue; }
                    match serde_json::from_str::<Value>(data) {
                        Ok(raw) => yield Ok(ChatStreamEvent { raw }),
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
