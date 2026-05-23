//! airproxy-stages

pub mod auth;
pub mod content_policy;
pub mod forward;
pub mod log;
pub mod model_route;

use airproxy_core::stage::Stage;
use airproxy_provider::provider::{Credentials, Provider};
use std::collections::HashMap;
use std::sync::Arc;

/// Maps `provider_id` → (provider impl, default credentials).
#[derive(Default)]
pub struct ProviderRegistry {
    inner: HashMap<String, (Arc<dyn Provider>, Credentials)>,
}

impl ProviderRegistry {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn insert(
        &mut self,
        id: impl Into<String>,
        provider: Arc<dyn Provider>,
        creds: Credentials,
    ) {
        self.inner.insert(id.into(), (provider, creds));
    }
    pub fn get(&self, id: &str) -> Option<&(Arc<dyn Provider>, Credentials)> {
        self.inner.get(id)
    }
}

/// Maps stage id (as in TOML config) → constructed stage instance.
pub struct StageRegistry {
    pub by_id: HashMap<&'static str, Arc<dyn Stage>>,
}

impl StageRegistry {
    pub fn new() -> Self {
        Self {
            by_id: HashMap::new(),
        }
    }
    pub fn insert(&mut self, id: &'static str, stage: Arc<dyn Stage>) {
        self.by_id.insert(id, stage);
    }
    pub fn build_pipeline(
        &self,
        ids: &[String],
    ) -> anyhow::Result<airproxy_core::pipeline::Pipeline> {
        let mut stages = Vec::with_capacity(ids.len());
        for id in ids {
            let s = self
                .by_id
                .get(id.as_str())
                .ok_or_else(|| anyhow::anyhow!("unknown stage id `{id}`"))?
                .clone();
            stages.push(s);
        }
        Ok(airproxy_core::pipeline::Pipeline::new(stages))
    }
}

impl Default for StageRegistry {
    fn default() -> Self {
        Self::new()
    }
}
