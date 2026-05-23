//! `ClusterProvider`: implements `ai_engine_provider::Provider` so the
//! existing gateway pipeline can route requests at a cluster without any
//! trait changes.
//!
//! Two modes:
//! - **Leader**: holds an `Arc<Mutex<ClusterLeader>>` and (in Plan 3) drives
//!   the cluster generation loop on `chat` / `chat_stream`.
//! - **Worker**: never receives inbound HTTP, so every Provider method
//!   returns `ProviderError::Unsupported`. The cluster member still exposes
//!   a Provider so that the same binary configuration shape works for both
//!   roles — the worker just refuses application traffic.
//!
//! Plan 2 scope is the trait surface only; the actual cluster dispatch in
//! `chat` is a stub returning `Unsupported`. Plan 3 wires the leader path
//! end-to-end.

use ai_engine_provider::{
    error::ProviderError,
    openai,
    provider::{CallCtx, Capabilities, Credentials, Provider},
};
use async_trait::async_trait;
use std::sync::Arc;

use crate::leader::ClusterLeader;

/// Whether this provider runs as the cluster leader (drives requests) or as
/// a worker (advertises capabilities but refuses application traffic).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    Leader,
    Worker,
}

pub struct ClusterProvider {
    id: String,
    role: Role,
    /// Live cluster handle, populated for `new_leader`. Plan 3 uses this to
    /// drive `full_forward_for_test`-style flows from `chat`. Unused in Plan
    /// 2 (the trait surface lands here, the dispatch lands in Plan 3).
    #[allow(dead_code)]
    inner: Option<Arc<tokio::sync::Mutex<ClusterLeader>>>,
}

impl ClusterProvider {
    /// Production constructor: leader mode with a live cluster handle.
    pub fn new_leader(
        id: impl Into<String>,
        leader: Arc<tokio::sync::Mutex<ClusterLeader>>,
    ) -> Self {
        Self {
            id: id.into(),
            role: Role::Leader,
            inner: Some(leader),
        }
    }

    /// Production constructor: worker mode (no inner handle; every Provider
    /// method returns `Unsupported`).
    pub fn new_worker(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            role: Role::Worker,
            inner: None,
        }
    }

    /// Test helper: leader mode without a live cluster. Used in the
    /// trait-impl smoke test; production code should use `new_leader`.
    pub fn stub_leader(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            role: Role::Leader,
            inner: None,
        }
    }

    /// Test helper: worker mode (equivalent to `new_worker`, named
    /// consistently with `stub_leader`).
    pub fn stub_worker(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            role: Role::Worker,
            inner: None,
        }
    }
}

#[async_trait]
impl Provider for ClusterProvider {
    fn id(&self) -> &str {
        &self.id
    }

    fn kind(&self) -> &'static str {
        "local-cluster"
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            chat: true,
            streaming: true,
            tools: false,
            vision: false,
            messages: false,
            embeddings: false,
        }
    }

    async fn chat(
        &self,
        _req: openai::ChatRequest,
        _creds: &Credentials,
        _ctx: &CallCtx,
    ) -> Result<openai::ChatResponse, ProviderError> {
        if self.role == Role::Worker {
            return Err(ProviderError::Unsupported);
        }
        // Plan 2 scope: surface the trait. Real dispatch (tokenize, run
        // cluster generation loop via `inner.lock().await`, sample, build
        // ChatResponse) lands in Plan 3.
        Err(ProviderError::Unsupported)
    }
}
