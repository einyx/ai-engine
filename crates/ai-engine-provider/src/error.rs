use bytes::Bytes;
use thiserror::Error;

// Variant payloads are owned `String`/`Bytes` rather than the underlying
// `reqwest::Error` / `serde_json::Error` / `io::Error` so this crate stays
// runtime-agnostic — concrete providers stringify at their boundary.
#[derive(Debug, Error)]
pub enum ProviderError {
    #[error("connect: {0}")]
    Connect(String),
    #[error("timeout")]
    Timeout,
    // Body is forwarded verbatim by ForwardStage so SDK clients see the
    // upstream's original error envelope.
    #[error("upstream status {status}")]
    Status { status: u16, body: Bytes },
    #[error("invalid response: {0}")]
    InvalidResponse(String),
    // The 502 status this maps to is a fallback only — by the time
    // a stream error fires we have already written the 200 SSE headers,
    // so the HTTP layer emits a final `event: error` chunk instead.
    #[error("stream error: {0}")]
    Stream(String),
    #[error("unsupported")]
    Unsupported,
}
