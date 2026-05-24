//! Replica pool: N independently-loaded `CandleModel`s for concurrent requests.

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::{Mutex, MutexGuard};

use ai_engine_tokenizer::HfTokenizer;
use candle_core::Device;

use crate::model::CandleModel;

/// Round-robin replica index. Pure function for testability.
pub(crate) fn next_index(counter: &AtomicUsize, n: usize) -> usize {
    counter.fetch_add(1, Ordering::Relaxed) % n
}

/// A pool of `n` model replicas. Each replica is an independently-loaded
/// `CandleModel` (weights NOT shared — candle-transformers bundles
/// weights+KV-cache with no sharing API). The tokenizer IS shared via `Arc`.
pub struct ReplicaPool {
    replicas: Vec<Mutex<CandleModel>>,
    counter: AtomicUsize,
}

impl ReplicaPool {
    /// Load the GGUF `n` times into `n` replicas on `device`.
    pub fn new(
        gguf_path: &Path,
        device: Device,
        tokenizer: Arc<HfTokenizer>,
        n: usize,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(n >= 1, "pool_size must be >= 1");
        let mut replicas = Vec::with_capacity(n);
        for i in 0..n {
            tracing::info!("candle: loading replica {}/{}", i + 1, n);
            let m = CandleModel::load(gguf_path, device.clone(), tokenizer.clone())?;
            replicas.push(Mutex::new(m));
        }
        Ok(Self { replicas, counter: AtomicUsize::new(0) })
    }

    /// Acquire a free replica. Tries each replica's `try_lock`; if all are busy,
    /// awaits the lock on a round-robin-chosen replica.
    pub async fn acquire(&self) -> MutexGuard<'_, CandleModel> {
        for r in &self.replicas {
            if let Ok(guard) = r.try_lock() {
                return guard;
            }
        }
        let idx = next_index(&self.counter, self.replicas.len());
        self.replicas[idx].lock().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_robin_index_wraps() {
        let counter = std::sync::atomic::AtomicUsize::new(0);
        assert_eq!(next_index(&counter, 2), 0);
        assert_eq!(next_index(&counter, 2), 1);
        assert_eq!(next_index(&counter, 2), 0);
        assert_eq!(next_index(&counter, 2), 1);
    }
}
