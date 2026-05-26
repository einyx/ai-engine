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
    #[serde(default, rename = "cluster")]
    pub clusters: Vec<Cluster>,
    #[serde(default)]
    pub discovery: Option<Discovery>,
}

/// LAN auto-discovery of upstreams. Currently covers Ollama instances
/// advertised on mDNS; discovered endpoints are merged into the provider/route
/// set at startup.
#[derive(Debug, Clone, Deserialize)]
pub struct Discovery {
    /// Browse mDNS for advertised Ollama endpoints and auto-register them.
    #[serde(default)]
    pub ollama_mdns: bool,
    /// How long the startup browse listens before building the pipeline.
    #[serde(default = "default_ollama_discovery_timeout")]
    pub timeout_secs: u64,
}
fn default_ollama_discovery_timeout() -> u64 {
    4
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
    /// Required for HTTP-based providers (`openai`, `anthropic`). Omitted for
    /// `local-cluster` providers which target a [[cluster]] by id instead.
    #[serde(default)]
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
    /// References a `[[cluster]] id` when `kind = "local-cluster"`.
    #[serde(default)]
    pub cluster: Option<String>,
    /// Path to model weights file (only for `kind = "candle"`).
    /// Must point to a `.gguf` file.
    #[serde(default)]
    pub weights_path: Option<String>,
    /// Candle device spec (only for `kind = "candle"`). auto|cpu|metal|cuda:N.
    #[serde(default)]
    pub device: Option<String>,
    /// Number of model replicas for concurrency (only for `kind = "candle"`).
    #[serde(default)]
    pub pool_size: Option<usize>,
    /// Engine for candle: "paged" (default, continuous batching) or "pool" (replica pool).
    #[serde(default)]
    pub engine: Option<String>,
    /// Paged engine: max concurrent sequences per batch (default 32).
    #[serde(default)]
    pub max_num_seqs: Option<usize>,
    /// Paged engine: KV block size in tokens (default 16).
    #[serde(default)]
    pub block_size: Option<usize>,
    /// Paged engine: KV block pool size, caps total KV memory (default 4096).
    #[serde(default)]
    pub kv_cache_blocks: Option<usize>,
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

#[derive(Debug, Clone, Deserialize)]
pub struct Cluster {
    pub id: String,
    pub leader: String,
    pub quic_bind: String,
    #[serde(default = "default_protocol_version")]
    pub protocol_version: u16,
    #[serde(default = "default_join_timeout")]
    pub join_timeout_secs: u64,
    #[serde(default = "default_heartbeat")]
    pub heartbeat_interval_secs: u64,
    pub model: ClusterModel,
    #[serde(default, rename = "node")]
    pub nodes: Vec<ClusterNode>,
    #[serde(default, rename = "partition_override")]
    pub partition_override: Vec<PartitionOverride>,
    #[serde(default)]
    pub discover: Option<ClusterDiscover>,
    /// Leaderless p2p mode (Phase B): every node forms a full mesh, serves its
    /// hosted stages to peers, and can ingest+coordinate HTTP requests via a
    /// local `Coordinator`. When false (default) the legacy star path is used
    /// (one leader orchestrates; workers only listen).
    #[serde(default)]
    pub leaderless: bool,
}
fn default_protocol_version() -> u16 {
    1
}
fn default_join_timeout() -> u64 {
    30
}
fn default_heartbeat() -> u64 {
    5
}

#[derive(Debug, Clone, Deserialize)]
pub struct ClusterDiscover {
    pub expected_workers: usize,
    #[serde(default = "default_discover_timeout")]
    pub timeout_secs: u64,
}
fn default_discover_timeout() -> u64 {
    30
}

#[derive(Debug, Clone, Deserialize)]
pub struct ClusterModel {
    pub id: String,
    pub weights_path: String,
    #[serde(default)]
    pub config_path: Option<String>,
    #[serde(default)]
    pub tokenizer_path: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ClusterNode {
    pub id: String,
    pub addr: String,
    pub cert_fingerprint: String,
    pub backend: String,
    #[serde(default)]
    pub device_index: usize,
    #[serde(default)]
    pub max_memory_mib: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PartitionOverride {
    pub node: String,
    pub layers: String,
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
