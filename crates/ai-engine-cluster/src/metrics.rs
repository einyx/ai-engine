//! Lock-free token counter shared between the leader's generation loop and
//! the web metrics endpoint. tokens/sec is derived by the reader from deltas.

use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Default)]
pub struct ClusterMetrics {
    total_tokens: AtomicU64,
}

impl ClusterMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one produced output token. Called on the generation hot path,
    /// so it is a single relaxed atomic add.
    pub fn record_token(&self) {
        self.total_tokens.fetch_add(1, Ordering::Relaxed);
    }

    pub fn total_tokens(&self) -> u64 {
        self.total_tokens.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_increments_total() {
        let m = ClusterMetrics::new();
        assert_eq!(m.total_tokens(), 0);
        m.record_token();
        m.record_token();
        assert_eq!(m.total_tokens(), 2);
    }
}
