//! RMSNorm: `out = x * rsqrt(mean(x^2, dim=-1) + eps) * weight`.

use burn::tensor::{Tensor, TensorData, backend::Backend};

/// RMSNorm: `out = x * rsqrt(mean(x^2, dim=-1) + eps) * weight`.
pub struct RmsNorm<B: Backend> {
    pub weight: Tensor<B, 1>,
    pub eps: f32,
}

impl<B: Backend> RmsNorm<B> {
    pub fn new(weight: Tensor<B, 1>, eps: f32) -> Self {
        Self { weight, eps }
    }

    pub fn with_weights<W: AsRef<[f32]>>(
        hidden: usize,
        weights: W,
        eps: f32,
        device: &B::Device,
    ) -> Self {
        let w_vec: Vec<f32> = weights.as_ref().to_vec();
        assert_eq!(w_vec.len(), hidden);
        let weight = Tensor::<B, 1>::from_data(TensorData::new(w_vec, [hidden]), device);
        Self { weight, eps }
    }

    /// `x: [..., hidden]` — normalizes over the last dim.
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        // sq = x * x; mean over last dim (mean_dim keeps the dim with size 1 in burn 0.21).
        let sq = x.clone().powf_scalar(2.0_f32);
        let mean = sq.mean_dim(D - 1);
        // rsqrt(mean + eps)
        let rsqrt = mean.add_scalar(self.eps).sqrt().recip();
        // normalize: x * rsqrt (broadcasts on last dim of size 1).
        let normed = x.mul(rsqrt);
        // scale by weight (broadcast over last dim). weight is [hidden] -> [1, ..., hidden].
        let mut target_shape = [1usize; D];
        target_shape[D - 1] = self.weight.dims()[0];
        let w = self.weight.clone().reshape(target_shape);
        normed.mul(w)
    }
}
