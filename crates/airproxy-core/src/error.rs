use airproxy_provider::error::ProviderError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GatewayError {
    #[error("bad request: {0}")]
    BadRequest(String),
    #[error("unauthorized")]
    Unauthorized,
    #[error("payload too large")]
    PayloadTooLarge,
    #[error("no route for model {model}")]
    NoRouteForModel { model: String },
    #[error(transparent)]
    Provider(#[from] ProviderError),
    #[error("internal: {0}")]
    Internal(String),
}

impl GatewayError {
    pub fn http_status(&self) -> u16 {
        match self {
            Self::BadRequest(_) => 400,
            Self::Unauthorized => 401,
            Self::PayloadTooLarge => 413,
            Self::NoRouteForModel { .. } => 502,
            Self::Provider(p) => match p {
                ProviderError::Connect(_) => 502,
                ProviderError::Timeout => 504,
                ProviderError::Status { status, .. } => *status,
                ProviderError::InvalidResponse(_) => 502,
                ProviderError::Stream(_) => 502,
                ProviderError::Unsupported => 502,
            },
            Self::Internal(_) => 500,
        }
    }
}
