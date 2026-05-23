//! Rotary Positional Embeddings (HF Llama "split-halves" convention).

use burn::tensor::{Int, Tensor, TensorData, backend::Backend};

/// Rotary Positional Embeddings (HF Llama "split-halves" convention).
///
/// Precomputes cos/sin tables of shape `[max_seq, head_dim/2]` at construction.
/// `apply(x, positions)` rotates the last dim of `x` per-position.
///
/// CRITICAL: HF Llama uses "split-halves" rotation:
///
/// ```text
/// first_half  = x[..., :head_dim/2]
/// second_half = x[..., head_dim/2:]
/// out_first   = first_half * cos - second_half * sin
/// out_second  = first_half * sin + second_half * cos
/// ```
///
/// NOT the interleaved evens/odds variant from the original RoFormer paper.
pub struct RotaryEmbedding<B: Backend> {
    pub cos: Tensor<B, 2>, // [max_seq, head_dim/2]
    pub sin: Tensor<B, 2>,
    pub head_dim: usize,
    pub max_seq: usize,
}

impl<B: Backend> RotaryEmbedding<B> {
    pub fn new(head_dim: usize, max_seq: usize, theta: f32, device: &B::Device) -> Self {
        assert!(
            head_dim % 2 == 0,
            "RoPE requires even head_dim, got {head_dim}",
        );
        let half = head_dim / 2;
        let freqs: Vec<f32> = (0..half)
            .map(|k| 1.0 / theta.powf((2.0 * k as f32) / head_dim as f32))
            .collect();
        let mut cos_data: Vec<f32> = Vec::with_capacity(max_seq * half);
        let mut sin_data: Vec<f32> = Vec::with_capacity(max_seq * half);
        for t in 0..max_seq {
            for freq in freqs.iter().take(half) {
                let angle = t as f32 * *freq;
                cos_data.push(angle.cos());
                sin_data.push(angle.sin());
            }
        }
        let cos = Tensor::<B, 2>::from_data(TensorData::new(cos_data, [max_seq, half]), device);
        let sin = Tensor::<B, 2>::from_data(TensorData::new(sin_data, [max_seq, half]), device);
        Self {
            cos,
            sin,
            head_dim,
            max_seq,
        }
    }

    pub fn cos_table_shape(&self) -> [usize; 2] {
        self.cos.dims()
    }

    pub fn sin_table_shape(&self) -> [usize; 2] {
        self.sin.dims()
    }

    /// Rotate `x: [batch, n_heads, seq, head_dim]` by RoPE.
    /// `positions[i]` is the absolute seq position of token `i`.
    pub fn apply(&self, x: Tensor<B, 4>, positions: &[i32]) -> Tensor<B, 4> {
        let [batch, n_heads, seq, head_dim] = x.dims();
        let half = head_dim / 2;
        assert_eq!(positions.len(), seq, "positions length must equal seq");
        assert_eq!(
            head_dim, self.head_dim,
            "x.head_dim ({head_dim}) != rope.head_dim ({})",
            self.head_dim
        );

        // 1. Split x into first half and second half along the last dim.
        let x_first = x
            .clone()
            .slice([0..batch, 0..n_heads, 0..seq, 0..half]);
        let x_second = x.slice([0..batch, 0..n_heads, 0..seq, half..head_dim]);

        // 2. Gather cos/sin at positions: result shape [seq, half].
        let device = self.cos.device();
        let positions_tensor = Tensor::<B, 1, Int>::from_data(
            TensorData::new(positions.to_vec(), [seq]),
            &device,
        );
        let cos_at = self.cos.clone().select(0, positions_tensor.clone());
        let sin_at = self.sin.clone().select(0, positions_tensor);

        // 3. Reshape cos_at, sin_at to broadcast over [batch, n_heads, seq, half].
        let cos_at = cos_at.reshape([1usize, 1, seq, half]);
        let sin_at = sin_at.reshape([1usize, 1, seq, half]);

        // 4. Rotation:
        //    out_first  = x_first * cos - x_second * sin
        //    out_second = x_first * sin + x_second * cos
        let out_first = x_first
            .clone()
            .mul(cos_at.clone())
            .sub(x_second.clone().mul(sin_at.clone()));
        let out_second = x_first.mul(sin_at).add(x_second.mul(cos_at));

        // 5. Concat back along the last dim.
        Tensor::cat(vec![out_first, out_second], 3)
    }
}
