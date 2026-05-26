//! ai-engine-http

mod error;
mod routes;
mod sse;

use ai_engine_core::cluster_view::ClusterView;
use ai_engine_core::pipeline::Pipeline;
use arc_swap::ArcSwap;
use std::collections::HashMap;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

/// One configured upstream AI service and the models routed to it, for the UI.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct ServiceInfo {
    pub id: String,
    pub kind: String,
    /// HTTP base_url, or `cluster:<id>` for local-cluster providers; None if neither.
    pub endpoint: Option<String>,
    pub models: Vec<String>,
    /// Device spec for in-process local backends (candle/rustyllm): cpu|cuda:N|metal.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
    /// Basename of the weights path for local backends, for display.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weights: Option<String>,
    /// True when inference runs in-process or against a localhost endpoint
    /// (vs. a remote HTTP API), so the UI can distinguish local from remote.
    pub local: bool,
}

/// Application state. `pipelines` are held behind `ArcSwap` so they can be
/// atomically replaced at runtime by the SIGHUP reload path in the binary
/// crate (Task 13).
pub struct AppState {
    pub pipelines: HashMap<&'static str, ArcSwap<Pipeline>>,
    pub openai_models: Vec<String>, // populated from route table for /v1/models
    pub ready: AtomicBool,
    /// Present only when this process is a cluster leader; `None` for
    /// gateway-only mode. Backs `/cluster/topology` and `/cluster/metrics`.
    pub cluster: Option<std::sync::Arc<dyn ClusterView>>,
    /// Configured upstream AI services for the UI `/cluster/services` endpoint.
    pub services: Vec<ServiceInfo>,
    /// Per-provider output-token counters backing the `/gateway/metrics` SSE.
    pub gateway_metrics: std::sync::Arc<ai_engine_core::metrics::GatewayMetrics>,
    /// Recent gateway requests for the activity graph (chat → model → provider).
    pub activity: std::sync::Arc<ai_engine_core::activity::ActivityLog>,
    /// Per-provider health from the background prober (cluster dashboard).
    pub health: std::sync::Arc<ai_engine_core::health::HealthStore>,
    /// Host resources for discovered (remote) providers, captured at discovery.
    /// Local providers are sampled live in the metrics SSE.
    pub resources:
        std::sync::Arc<std::collections::HashMap<String, ai_engine_core::resources::NodeResources>>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            pipelines: HashMap::new(),
            openai_models: vec![],
            ready: AtomicBool::new(false),
            cluster: None,
            services: vec![],
            gateway_metrics: std::sync::Arc::new(
                ai_engine_core::metrics::GatewayMetrics::default(),
            ),
            activity: std::sync::Arc::new(ai_engine_core::activity::ActivityLog::default()),
            health: std::sync::Arc::new(ai_engine_core::health::HealthStore::new()),
            resources: std::sync::Arc::new(std::collections::HashMap::new()),
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

pub fn build_router(state: Arc<AppState>) -> axum::Router {
    axum::Router::new()
        .route(
            "/v1/chat/completions",
            axum::routing::post(routes::chat_completions),
        )
        .route("/v1/messages", axum::routing::post(routes::messages))
        .route("/v1/embeddings", axum::routing::post(routes::embeddings))
        .route("/v1/models", axum::routing::get(routes::models))
        .route("/healthz", axum::routing::get(routes::healthz))
        .route("/readyz", axum::routing::get(routes::readyz))
        .route("/cluster/topology", axum::routing::get(routes::cluster_topology))
        .route("/cluster/metrics", axum::routing::get(routes::cluster_metrics))
        .route("/gateway/metrics", axum::routing::get(routes::gateway_metrics))
        .route("/cluster/services", axum::routing::get(routes::cluster_services))
        .route("/graph", axum::routing::get(routes::graph))
        .with_state(state)
}
