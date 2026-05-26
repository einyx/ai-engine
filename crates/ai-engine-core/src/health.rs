//! Per-provider health, written by the gateway's background prober and read by
//! the `/gateway/metrics` SSE. Lives in `-core` so the binary (prober, with an
//! HTTP client) and `-http` (reader) share it without a dependency cycle.

use std::collections::HashMap;
use std::sync::Mutex;

#[derive(Debug, Clone, Default)]
pub struct Health {
    /// Whether the last probe succeeded.
    pub up: bool,
    /// Round-trip latency of the last successful probe, milliseconds.
    pub latency_ms: u64,
    /// Whether this provider has been probed at all yet.
    pub checked: bool,
}

/// Thread-safe map of `provider_id -> Health`.
#[derive(Debug, Default)]
pub struct HealthStore {
    inner: Mutex<HashMap<String, Health>>,
}

impl HealthStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&self, provider_id: &str, up: bool, latency_ms: u64) {
        if let Ok(mut m) = self.inner.lock() {
            m.insert(
                provider_id.to_string(),
                Health {
                    up,
                    latency_ms,
                    checked: true,
                },
            );
        }
    }

    pub fn snapshot(&self) -> HashMap<String, Health> {
        self.inner.lock().map(|m| m.clone()).unwrap_or_default()
    }
}
