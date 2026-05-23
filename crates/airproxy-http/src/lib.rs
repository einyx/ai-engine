//! airproxy-http

mod error;
mod routes;
mod sse;

use airproxy_core::pipeline::Pipeline;
use arc_swap::ArcSwap;
use std::collections::HashMap;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

/// Application state. `pipelines` are held behind `ArcSwap` so they can be
/// atomically replaced at runtime by the SIGHUP reload path in the binary
/// crate (Task 13).
pub struct AppState {
    pub pipelines: HashMap<&'static str, ArcSwap<Pipeline>>,
    pub openai_models: Vec<String>, // populated from route table for /v1/models
    pub ready: AtomicBool,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            pipelines: HashMap::new(),
            openai_models: vec![],
            ready: AtomicBool::new(false),
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
        .with_state(state)
}
