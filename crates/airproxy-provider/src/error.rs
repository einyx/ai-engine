use bytes::Bytes;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProviderError {
    #[error("connect: {0}")]
    Connect(String),
    #[error("timeout")]
    Timeout,
    #[error("upstream status {status}")]
    Status { status: u16, body: Bytes },
    #[error("invalid response: {0}")]
    InvalidResponse(String),
    #[error("stream error: {0}")]
    Stream(String),
    #[error("unsupported")]
    Unsupported,
}
