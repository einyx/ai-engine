//! Per-provider gateway metrics.
//!
//! The `forward` stage credits output tokens as they're produced; the terminal
//! `log` stage records one request (status + latency) per call. The
//! `/gateway/metrics` SSE derives per-second rates from deltas of these
//! monotonic counters and reports the cumulative totals. Lives in `-core` so
//! both `-stages` (writer) and `-http` (reader) can depend on it without a
//! cycle.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Default)]
struct Counters {
    out_tokens: AtomicU64,
    requests: AtomicU64,
    errors: AtomicU64,
    latency_ms_sum: AtomicU64,
}

/// Immutable per-provider snapshot for the SSE layer.
#[derive(Debug, Clone, Default)]
pub struct ProviderSnapshot {
    pub out_tokens: u64,
    pub requests: u64,
    pub errors: u64,
    pub latency_ms_sum: u64,
}

/// Cumulative per-provider counters since startup.
#[derive(Debug, Default)]
pub struct GatewayMetrics {
    providers: HashMap<String, Counters>,
}

impl GatewayMetrics {
    /// Pre-populate counters for every known provider id. Counters are fixed at
    /// construction; unknown ids passed to the recorders are ignored.
    pub fn new(ids: impl IntoIterator<Item = String>) -> Self {
        Self {
            providers: ids.into_iter().map(|id| (id, Counters::default())).collect(),
        }
    }

    /// Add `n` completion tokens to `provider_id` (no-op if unknown).
    pub fn add_output(&self, provider_id: &str, n: u64) {
        if let Some(c) = self.providers.get(provider_id) {
            c.out_tokens.fetch_add(n, Ordering::Relaxed);
        }
    }

    /// Record one completed request: bumps request count, error count (when
    /// `is_error`), and the latency sum. No-op for unknown providers.
    pub fn record_request(&self, provider_id: &str, is_error: bool, latency_ms: u64) {
        if let Some(c) = self.providers.get(provider_id) {
            c.requests.fetch_add(1, Ordering::Relaxed);
            if is_error {
                c.errors.fetch_add(1, Ordering::Relaxed);
            }
            c.latency_ms_sum.fetch_add(latency_ms, Ordering::Relaxed);
        }
    }

    /// Snapshot of every provider's cumulative counters.
    pub fn snapshot(&self) -> Vec<(String, ProviderSnapshot)> {
        self.providers
            .iter()
            .map(|(id, c)| {
                (
                    id.clone(),
                    ProviderSnapshot {
                        out_tokens: c.out_tokens.load(Ordering::Relaxed),
                        requests: c.requests.load(Ordering::Relaxed),
                        errors: c.errors.load(Ordering::Relaxed),
                        latency_ms_sum: c.latency_ms_sum.load(Ordering::Relaxed),
                    },
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::GatewayMetrics;

    #[test]
    fn records_tokens_requests_errors_latency() {
        let m = GatewayMetrics::new(["a".to_string(), "b".to_string()]);
        m.add_output("a", 10);
        m.add_output("a", 5);
        m.record_request("a", false, 120);
        m.record_request("a", true, 80);
        m.add_output("ghost", 100); // ignored
        m.record_request("ghost", false, 1); // ignored
        let snap: std::collections::HashMap<_, _> = m.snapshot().into_iter().collect();
        assert_eq!(snap["a"].out_tokens, 15);
        assert_eq!(snap["a"].requests, 2);
        assert_eq!(snap["a"].errors, 1);
        assert_eq!(snap["a"].latency_ms_sum, 200);
        assert_eq!(snap["b"].requests, 0);
        assert!(!snap.contains_key("ghost"));
    }
}
