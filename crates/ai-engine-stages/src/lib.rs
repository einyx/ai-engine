//! ai-engine-stages

pub mod auth;
pub mod content_policy;
pub mod forward;
pub mod log;
pub mod model_route;

use ai_engine_core::stage::Stage;
use ai_engine_provider::provider::{Credentials, Provider};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
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
    /// All registered provider ids (used to pre-populate the load tracker).
    pub fn ids(&self) -> impl Iterator<Item = &String> {
        self.inner.keys()
    }
}

/// Tracks in-flight request counts per provider so `forward` can route each
/// request to the least-busy provider in a model's pool.
#[derive(Default)]
pub struct LoadTracker {
    inflight: HashMap<String, Arc<AtomicUsize>>,
}

impl LoadTracker {
    /// Pre-populate a counter for every known provider id.
    pub fn new(ids: impl IntoIterator<Item = String>) -> Self {
        Self {
            inflight: ids
                .into_iter()
                .map(|id| (id, Arc::new(AtomicUsize::new(0))))
                .collect(),
        }
    }
    /// Current in-flight count for `id` (0 if unknown).
    pub fn load(&self, id: &str) -> usize {
        self.inflight
            .get(id)
            .map(|c| c.load(Ordering::Relaxed))
            .unwrap_or(0)
    }
    /// Increment the in-flight count for `id`; the returned guard decrements on
    /// drop. Unknown ids yield a no-op guard.
    pub fn acquire(&self, id: &str) -> InflightGuard {
        let counter = self.inflight.get(id).cloned();
        if let Some(c) = &counter {
            c.fetch_add(1, Ordering::Relaxed);
        }
        InflightGuard { counter }
    }
}

/// Decrements its provider's in-flight count when dropped. For streaming
/// responses it is moved into the response stream so the count stays held for
/// the stream's whole lifetime.
pub struct InflightGuard {
    counter: Option<Arc<AtomicUsize>>,
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        if let Some(c) = &self.counter {
            c.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod load_tracker_tests {
    use super::LoadTracker;

    #[test]
    fn acquire_increments_and_guard_drop_decrements() {
        let t = LoadTracker::new(["a".to_string(), "b".to_string()]);
        assert_eq!(t.load("a"), 0);
        let g1 = t.acquire("a");
        let g2 = t.acquire("a");
        assert_eq!(t.load("a"), 2);
        assert_eq!(t.load("b"), 0);
        drop(g1);
        assert_eq!(t.load("a"), 1);
        drop(g2);
        assert_eq!(t.load("a"), 0);
    }

    #[test]
    fn unknown_id_is_zero_and_noop_guard() {
        let t = LoadTracker::new(["a".to_string()]);
        let g = t.acquire("missing");
        assert_eq!(t.load("missing"), 0);
        drop(g); // must not panic
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
    ) -> anyhow::Result<ai_engine_core::pipeline::Pipeline> {
        let mut stages = Vec::with_capacity(ids.len());
        for id in ids {
            let s = self
                .by_id
                .get(id.as_str())
                .ok_or_else(|| anyhow::anyhow!("unknown stage id `{id}`"))?
                .clone();
            stages.push(s);
        }
        Ok(ai_engine_core::pipeline::Pipeline::new(stages))
    }
}

impl Default for StageRegistry {
    fn default() -> Self {
        Self::new()
    }
}
