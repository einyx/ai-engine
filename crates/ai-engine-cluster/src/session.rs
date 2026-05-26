//! Per-request leader-side state.
//!
//! Plan 4 Task 4 splits what used to be the local variables in
//! `ClusterLeader::generate(&mut self, ...)` into two pieces:
//!
//! - [`LeaderModel`]: immutable model artifacts (embedding, leader's
//!   decoder blocks, final norm, output projection, model config). Shared
//!   across many concurrent requests via `Arc<LeaderModel<B>>`.
//! - [`RequestSession`]: the mutable per-request state — KV caches, the
//!   current sequence position, the cloned per-worker `quinn::Connection`s,
//!   and a unique `request_id` used by workers to key their own KV maps.
//!
//! With this split, `ClusterLeader::generate` is now `&self`: each call
//! builds a fresh `RequestSession` and drives the forward loop against it.
//! Multiple sessions can coexist on the same `ClusterLeader`.

use ai_engine_runtime::{
    arch::{
        block::DecoderBlock,
        embedding::{OutputProjection, TokenEmbedding},
        rmsnorm::RmsNorm,
    },
    config::ModelConfig,
    kv_cache::KvCacheSlot,
};
use burn::tensor::backend::Backend;
use quinn::Connection;
use std::sync::Arc;

/// Shared, immutable model artifacts loaded once per leader for a given
/// backend. Multiple `RequestSession`s share the same `LeaderModel` cheaply
/// via `Arc`.
pub struct LeaderModel<B: Backend> {
    pub embedding: TokenEmbedding<B>,
    pub blocks: Vec<DecoderBlock<B>>,
    pub final_norm: RmsNorm<B>,
    pub output: OutputProjection<B>,
    pub cfg: ModelConfig,
}

/// Per-request leader-side state. One `RequestSession` per concurrent request.
///
/// Owns its own KV caches and its own clones of the worker `quinn::Connection`s
/// (cheap; quinn connections are internally Arc-backed). Each session gets a
/// distinct `request_id`, which workers use as the key in their per-request
/// KV cache map.
pub struct RequestSession<B: Backend> {
    pub model: Arc<LeaderModel<B>>,
    pub leader_caches: Vec<KvCacheSlot<B>>,
    pub current_pos: usize,
    pub worker_conns: Vec<Connection>,
    pub request_id: uuid::Uuid,
}

impl<B: Backend> RequestSession<B>
where
    B::Device: Default,
{
    /// Construct a fresh per-request session with empty KV caches.
    pub fn new(
        model: Arc<LeaderModel<B>>,
        worker_conns: Vec<Connection>,
        device: &B::Device,
    ) -> Self {
        let cfg = &model.cfg;
        let leader_caches = (0..model.blocks.len())
            .map(|_| {
                KvCacheSlot::<B>::new(
                    1,
                    cfg.n_kv_heads,
                    cfg.max_position_embeddings,
                    cfg.head_dim,
                    device,
                )
            })
            .collect();
        Self {
            model,
            leader_caches,
            current_pos: 0,
            worker_conns,
            request_id: uuid::Uuid::now_v7(),
        }
    }
}
