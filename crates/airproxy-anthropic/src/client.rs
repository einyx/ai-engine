use airproxy_provider::provider::Credentials;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};

#[derive(Clone)]
pub struct ClientConfig {
    pub timeout_secs: u64,
}

pub fn build(client_cfg: &ClientConfig) -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(client_cfg.timeout_secs))
        .pool_idle_timeout(std::time::Duration::from_secs(90))
        .build()
        .expect("reqwest client")
}

/// Build auth + extra headers for an outbound Anthropic call.
///
/// - `x-api-key` is sourced from `api_key`, else `raw_bearer` (passthrough mode
///   may carry the user's key in the inbound `Authorization` header — we lift
///   it directly into x-api-key for the upstream, stripping any "Bearer " prefix).
/// - `anthropic-version: 2023-06-01` is always added (callers can override via
///   `extra_headers` if they need a different API version).
pub fn auth_headers(creds: &Credentials, extra: &[(String, String)]) -> HeaderMap {
    let mut h = HeaderMap::new();
    let key: Option<&str> = if let Some(k) = creds.api_key.as_deref() {
        Some(k)
    } else {
        creds.raw_bearer.as_deref().map(strip_bearer_prefix)
    };
    if let Some(k) = key {
        if let Ok(v) = HeaderValue::from_str(k) {
            h.insert("x-api-key", v);
        }
    }
    h.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
    for (k, v) in creds.extra_headers.iter().chain(extra) {
        if let (Ok(name), Ok(value)) = (HeaderName::try_from(k.as_str()), HeaderValue::from_str(v)) {
            h.insert(name, value);
        }
    }
    h
}

fn strip_bearer_prefix(s: &str) -> &str {
    s.strip_prefix("Bearer ").or_else(|| s.strip_prefix("bearer ")).unwrap_or(s)
}
