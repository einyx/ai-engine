//! Multi-head attention with Grouped-Query Attention (GQA), Rotary Positional
//! Embeddings (RoPE), and KV cache integration.
//!
//! Forward pass:
//! 1. Linear projections produce Q, K, V.
//! 2. Reshape into heads and apply RoPE to Q and K.
//! 3. Append K, V into the per-layer KV cache; read the full cached sequence.
//! 4. Broadcast cached K/V from `n_kv_heads` to `n_heads` (CONSECUTIVE
//!    repetition, Llama convention).
//! 5. Scaled dot-product attention with an additive causal mask.
//! 6. Merge heads and apply the output projection.

use crate::arch::rope::RotaryEmbedding;
use crate::kv_cache::KvCacheSlot;
use burn::tensor::{Distribution, Tensor, TensorData, activation::softmax, backend::Backend};

/// GQA attention block.
///
/// Weight shapes:
/// - `q_proj`: `[hidden, n_heads * head_dim]`
/// - `k_proj`: `[hidden, n_kv_heads * head_dim]`
/// - `v_proj`: `[hidden, n_kv_heads * head_dim]`
/// - `o_proj`: `[n_heads * head_dim, hidden]`
pub struct Attention<B: Backend> {
    pub q_proj: Tensor<B, 2>,
    pub k_proj: Tensor<B, 2>,
    pub v_proj: Tensor<B, 2>,
    pub o_proj: Tensor<B, 2>,
    pub rope: RotaryEmbedding<B>,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub scale: f32,
}

impl<B: Backend> Attention<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        q_proj: Tensor<B, 2>,
        k_proj: Tensor<B, 2>,
        v_proj: Tensor<B, 2>,
        o_proj: Tensor<B, 2>,
        rope: RotaryEmbedding<B>,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        assert!(
            n_heads % n_kv_heads == 0,
            "n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})",
        );
        let scale = 1.0 / (head_dim as f32).sqrt();
        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rope,
            n_heads,
            n_kv_heads,
            head_dim,
            scale,
        }
    }

    pub fn with_random_weights(
        hidden: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        rope_theta: f32,
        device: &B::Device,
    ) -> Self {
        let q_proj = Tensor::<B, 2>::random(
            [hidden, n_heads * head_dim],
            Distribution::Default,
            device,
        );
        let k_proj = Tensor::<B, 2>::random(
            [hidden, n_kv_heads * head_dim],
            Distribution::Default,
            device,
        );
        let v_proj = Tensor::<B, 2>::random(
            [hidden, n_kv_heads * head_dim],
            Distribution::Default,
            device,
        );
        let o_proj = Tensor::<B, 2>::random(
            [n_heads * head_dim, hidden],
            Distribution::Default,
            device,
        );
        let rope = RotaryEmbedding::<B>::new(head_dim, max_seq, rope_theta, device);
        Self::new(q_proj, k_proj, v_proj, o_proj, rope, n_heads, n_kv_heads, head_dim)
    }

    /// Forward pass.
    ///
    /// - `x`: `[batch, seq, hidden]`
    /// - `positions[i]` is the absolute sequence position of token `i`.
    /// - `cache`: per-layer KV cache; mutated in place.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        positions: &[i32],
        cache: &mut KvCacheSlot<B>,
    ) -> Tensor<B, 3> {
        let [batch, seq, _hidden] = x.dims();

        // 1. Linear projections.
        let q_w: Tensor<B, 3> = self.q_proj.clone().unsqueeze();
        let k_w: Tensor<B, 3> = self.k_proj.clone().unsqueeze();
        let v_w: Tensor<B, 3> = self.v_proj.clone().unsqueeze();
        let q = x.clone().matmul(q_w);
        let k = x.clone().matmul(k_w);
        let v = x.matmul(v_w);

        // 2. Reshape to [batch, heads, seq, head_dim].
        let q = q
            .reshape([batch, seq, self.n_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch, seq, self.n_kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch, seq, self.n_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        // 3. Apply RoPE to Q and K.
        let q = self.rope.apply(q, positions);
        let k = self.rope.apply(k, positions);

        // 4. Append into KV cache; read back full cached sequence.
        cache.append(k, v);
        let (k_all, v_all) = cache.read();
        let total_seq = cache.current_len();

        // 5. Broadcast cached K/V from n_kv_heads to n_heads (Llama: consecutive).
        let repeat = self.n_heads / self.n_kv_heads;
        let k_all = repeat_heads(k_all, repeat, batch, self.n_kv_heads, total_seq, self.head_dim);
        let v_all = repeat_heads(v_all, repeat, batch, self.n_kv_heads, total_seq, self.head_dim);

        // 6. Scaled dot-product attention.
        // scores: [batch, n_heads, seq, total_seq]
        let scores = q.matmul(k_all.swap_dims(2, 3)).mul_scalar(self.scale);
        let scores = apply_causal_mask::<B>(scores, positions, total_seq);
        let probs = softmax(scores, 3);
        let ctx = probs.matmul(v_all); // [batch, n_heads, seq, head_dim]

        // 7. Merge heads.
        let ctx = ctx
            .swap_dims(1, 2)
            .reshape([batch, seq, self.n_heads * self.head_dim]);

        // 8. Output projection.
        let o_w: Tensor<B, 3> = self.o_proj.clone().unsqueeze();
        ctx.matmul(o_w)
    }
}

/// Repeat each KV head `repeat` times CONSECUTIVELY along the head dim.
///
/// Llama convention: for `n_kv_heads=2` broadcast to `n_heads=4` with
/// `repeat=2`, the result has head ordering `[kv0, kv0, kv1, kv1]` (NOT
/// interleaved).
///
/// Input shape:  `[batch, n_kv, seq, head_dim]`.
/// Output shape: `[batch, n_kv * repeat, seq, head_dim]`.
fn repeat_heads<B: Backend>(
    x: Tensor<B, 4>,
    repeat: usize,
    batch: usize,
    n_kv: usize,
    seq: usize,
    head_dim: usize,
) -> Tensor<B, 4> {
    if repeat == 1 {
        return x;
    }
    // Promote to rank 5 with a singleton "repeat" axis between n_kv and seq,
    // expand it to `repeat`, then collapse n_kv and repeat into a single head
    // axis. Because the repeat axis is immediately after n_kv, collapsing
    // yields consecutive repetition: kv0, kv0, kv1, kv1, ...
    let expanded: Tensor<B, 5> = x
        .reshape([batch, n_kv, 1, seq, head_dim])
        .expand([batch, n_kv, repeat, seq, head_dim]);
    expanded.reshape([batch, n_kv * repeat, seq, head_dim])
}

/// Additive causal mask.
///
/// `scores: [batch, n_heads, q_seq, k_seq]`. Position `j` along `k_seq` is
/// allowed for query `i` iff `j <= positions[i]`. Disallowed positions get
/// `-inf` added so softmax zeroes them out.
fn apply_causal_mask<B: Backend>(
    scores: Tensor<B, 4>,
    positions: &[i32],
    k_seq: usize,
) -> Tensor<B, 4> {
    let q_seq = positions.len();
    let mut mask_data: Vec<f32> = Vec::with_capacity(q_seq * k_seq);
    for &p in positions.iter().take(q_seq) {
        let allowed_max = p as i64;
        for j in 0..k_seq {
            if (j as i64) <= allowed_max {
                mask_data.push(0.0_f32);
            } else {
                mask_data.push(f32::NEG_INFINITY);
            }
        }
    }
    let device = scores.device();
    let mask = Tensor::<B, 2>::from_data(TensorData::new(mask_data, [q_seq, k_seq]), &device)
        .reshape([1usize, 1, q_seq, k_seq]);
    scores.add(mask)
}
