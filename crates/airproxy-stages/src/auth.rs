use airproxy_core::ctx::{Identity, RequestCtx};
use airproxy_core::error::GatewayError;
use airproxy_core::stage::{Stage, StageError, StageOutcome};
use http::{HeaderMap, HeaderValue};

/// Authentication modes for v1. Future sub-projects add DB-backed keys / OIDC.
pub enum AuthMode {
    /// Forward whatever the client sent. Useful when airproxy is acting purely
    /// as a transport (e.g., per-user BYOK against OpenAI/Anthropic upstream).
    Passthrough,
    /// Accept only listed keys. Each entry is `(key, holder_name)`.
    /// First matching key wins; the holder name lands in `Identity::Holder`.
    SharedKey { keys: Vec<(String, String)> },
}

pub struct AuthStage {
    pub mode: AuthMode,
}

#[async_trait::async_trait]
impl Stage for AuthStage {
    fn name(&self) -> &'static str { "auth" }

    async fn process(&self, ctx: &mut RequestCtx) -> Result<StageOutcome, StageError> {
        let bearer = extract_bearer(&ctx.headers);
        match &self.mode {
            AuthMode::Passthrough => {
                ctx.identity = Some(Identity::Anonymous {
                    raw_bearer: bearer.map(str::to_string),
                });
                Ok(StageOutcome::Continue)
            }
            AuthMode::SharedKey { keys } => {
                let Some(b) = bearer else {
                    return Err(StageError {
                        stage: self.name(),
                        error: GatewayError::Unauthorized,
                    });
                };
                if let Some((_, name)) = keys.iter().find(|(k, _)| k == b) {
                    ctx.identity = Some(Identity::Holder { name: name.clone() });
                    Ok(StageOutcome::Continue)
                } else {
                    Err(StageError {
                        stage: self.name(),
                        error: GatewayError::Unauthorized,
                    })
                }
            }
        }
    }
}

fn extract_bearer(headers: &HeaderMap) -> Option<&str> {
    headers
        .get("authorization")
        .and_then(|v: &HeaderValue| v.to_str().ok())
        .and_then(|s: &str| s.strip_prefix("Bearer ").or_else(|| s.strip_prefix("bearer ")))
}
