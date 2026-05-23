use airproxy_provider::provider::Credentials;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};

#[derive(Clone)]
pub struct ClientConfig {
    pub timeout_secs: u64,
    pub http2: bool,
}

pub fn build(client_cfg: &ClientConfig) -> reqwest::Client {
    let b = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(client_cfg.timeout_secs))
        .pool_idle_timeout(std::time::Duration::from_secs(90));
    if client_cfg.http2 {
        // Don't use http2_prior_knowledge — that breaks plain HTTP/1.1
        // Ollama (which is the whole point of this provider). Instead let
        // hyper negotiate via ALPN on TLS; on plaintext, stay on /1.1.
    }
    b.build().expect("reqwest client")
}

/// Build the auth + extra headers for an outbound request.
///
/// Header precedence:
/// 1. `raw_bearer` — verbatim passthrough mode wins (preserves SDK keys).
/// 2. `api_key`    — wrap in `Bearer <key>`.
/// 3. Neither set  — omit `Authorization` (Ollama / local mode).
///
/// `extra_headers` from creds and `extra` arg both append; later writes win
/// per `HeaderMap::insert` semantics.
pub fn auth_headers(creds: &Credentials, extra: &[(String, String)]) -> HeaderMap {
    let mut h = HeaderMap::new();
    if let Some(bearer) = creds.raw_bearer.as_ref() {
        if let Ok(v) = HeaderValue::from_str(bearer) {
            h.insert("authorization", v);
        }
    } else if let Some(k) = creds.api_key.as_ref() {
        if let Ok(v) = HeaderValue::from_str(&format!("Bearer {k}")) {
            h.insert("authorization", v);
        }
    }
    for (k, v) in creds.extra_headers.iter().chain(extra) {
        if let (Ok(name), Ok(value)) = (HeaderName::try_from(k.as_str()), HeaderValue::from_str(v)) {
            h.insert(name, value);
        }
    }
    h
}
