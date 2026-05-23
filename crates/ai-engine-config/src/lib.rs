//! ai-engine-config

mod interpolate;
mod validate;

use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub server: Server,
    pub auth: Auth,
    #[serde(default)]
    pub content_policy: ContentPolicy,
    #[serde(rename = "provider", default)]
    pub providers: Vec<Provider>,
    #[serde(rename = "route", default)]
    pub routes: Vec<Route>,
    #[serde(default)]
    pub pipeline: HashMap<String, Pipeline>,
}

#[derive(Debug, Deserialize)]
pub struct Server {
    pub bind: String,
    #[serde(default = "default_grace")]
    pub shutdown_grace_secs: u64,
    #[serde(default = "default_log_format")]
    pub log_format: String,
    #[serde(default = "default_log_level")]
    pub log_level: String,
}
fn default_grace() -> u64 {
    30
}
fn default_log_format() -> String {
    "json".into()
}
fn default_log_level() -> String {
    "info".into()
}

#[derive(Debug, Deserialize)]
pub struct Auth {
    pub mode: String,
    #[serde(default)]
    pub master_keys: Vec<MasterKey>,
}

#[derive(Debug, Deserialize)]
pub struct MasterKey {
    pub key: String,
    pub name: String,
}

#[derive(Debug, Deserialize)]
pub struct ContentPolicy {
    #[serde(default = "default_max_bytes")]
    pub max_request_bytes: usize,
    #[serde(default)]
    pub prompt_injection_patterns: Vec<String>,
}

impl Default for ContentPolicy {
    fn default() -> Self {
        Self {
            max_request_bytes: default_max_bytes(),
            prompt_injection_patterns: Vec::new(),
        }
    }
}

fn default_max_bytes() -> usize {
    1_048_576
}

#[derive(Debug, Deserialize)]
pub struct Provider {
    pub id: String,
    pub kind: String,
    pub base_url: String,
    /// Optional: omit for Ollama, vLLM, LM Studio, and other local OpenAI-
    /// compatible servers that don't require authentication.
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    #[serde(default = "default_true")]
    pub http2: bool,
    #[serde(default)]
    pub extra_headers: HashMap<String, String>,
}
fn default_timeout() -> u64 {
    120
}
fn default_true() -> bool {
    true
}

#[derive(Debug, Deserialize)]
pub struct Route {
    pub r#match: RouteMatch,
    pub provider: String,
    #[serde(default)]
    pub upstream_model: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct RouteMatch {
    pub model: String,
}

#[derive(Debug, Deserialize)]
pub struct Pipeline {
    pub stages: Vec<String>,
}

impl Config {
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(src: &str) -> anyhow::Result<Self> {
        let interpolated = interpolate::env_substitute(src)?;
        let cfg: Self = toml::from_str(&interpolated)
            .map_err(|e| anyhow::anyhow!("toml parse: {e}"))?;
        validate::validate(&cfg)?;
        Ok(cfg)
    }

    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let src = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("read {}: {e}", path.display()))?;
        Self::from_str(&src)
    }
}
