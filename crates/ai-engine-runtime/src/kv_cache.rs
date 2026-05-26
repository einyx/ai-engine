//! Per-request, per-layer KV cache for autoregressive decoding.
//!
//! Each `KvCacheSlot` pre-allocates `[batch, n_kv_heads, max_seq, head_dim]` k/v
//! tensors and tracks how many positions on the `seq` axis have been filled.
//! `append` writes the new tokens into the pre-allocated buffer using
//! `slice_assign` — we never grow via `cat`, which would defeat the purpose
//! of the cache.

use burn::tensor::{Tensor, backend::Backend};

/// Per-layer KV cache for a single request. Holds k/v tensors with the
/// `seq` dimension growing as tokens are produced.
pub struct KvCacheSlot<B: Backend> {
    pub k: Tensor<B, 4>, // [batch, n_kv_heads, max_seq, head_dim]
    pub v: Tensor<B, 4>,
    pub max_seq: usize,
    current_len: usize,
}

impl<B: Backend> KvCacheSlot<B> {
    pub fn new(
        batch: usize,
        n_kv_heads: usize,
        max_seq: usize,
        head_dim: usize,
        device: &B::Device,
    ) -> Self {
        let k = Tensor::<B, 4>::zeros([batch, n_kv_heads, max_seq, head_dim], device);
        let v = Tensor::<B, 4>::zeros([batch, n_kv_heads, max_seq, head_dim], device);
        Self {
            k,
            v,
            max_seq,
            current_len: 0,
        }
    }

    pub fn current_len(&self) -> usize {
        self.current_len
    }

    /// Append `k_new`, `v_new` of shape `[batch, n_kv_heads, new_tokens, head_dim]`
    /// into the pre-allocated buffer at positions
    /// `[current_len..current_len + new_tokens]` on the seq dim.
    ///
    /// Panics if appending would exceed `max_seq`.
    pub fn append(&mut self, k_new: Tensor<B, 4>, v_new: Tensor<B, 4>) {
        let n = k_new.dims()[2];
        assert!(
            self.current_len + n <= self.max_seq,
            "KV cache overflow: have {}, adding {}, max {}",
            self.current_len,
            n,
            self.max_seq,
        );
        let [b, nh, _, hd] = self.k.dims();
        let start = self.current_len;
        let end = self.current_len + n;
        // burn 0.21: `slice_assign` consumes self and returns a new tensor.
        self.k = self
            .k
            .clone()
            .slice_assign([0..b, 0..nh, start..end, 0..hd], k_new);
        self.v = self
            .v
            .clone()
            .slice_assign([0..b, 0..nh, start..end, 0..hd], v_new);
        self.current_len = end;
    }

    /// Returns slices of k and v covering `[0..current_len]` on the seq dim.
    pub fn read(&self) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [b, nh, _, hd] = self.k.dims();
        let k = self
            .k
            .clone()
            .slice([0..b, 0..nh, 0..self.current_len, 0..hd]);
        let v = self
            .v
            .clone()
            .slice([0..b, 0..nh, 0..self.current_len, 0..hd]);
        (k, v)
    }
}
