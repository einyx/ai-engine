use crate::config::ModelConfig;
use crate::kv_cache::KvCacheSlot;
use burn::tensor::backend::Backend;

/// All the per-request state that persists across token generations.
/// Used by both single-node serving (Plan 1) and cluster leader path (Plan 2).
pub struct RequestState<B: Backend> {
    pub caches: Vec<KvCacheSlot<B>>,
    pub current_pos: usize,
}

impl<B: Backend> RequestState<B> {
    pub fn new(cfg: &ModelConfig, batch: usize, max_tokens: usize, device: &B::Device) -> Self {
        let caches = (0..cfg.n_layers).map(|_| {
            KvCacheSlot::<B>::new(batch, cfg.n_kv_heads, max_tokens, cfg.head_dim, device)
        }).collect();
        Self { caches, current_pos: 0 }
    }

    pub fn advance(&mut self, n: usize) { self.current_pos += n; }
}
