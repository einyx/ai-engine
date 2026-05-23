//! SwiGLU FFN: `down(silu(gate(x)) * up(x))`.

use burn::tensor::{Distribution, Tensor, activation::silu, backend::Backend};

/// SwiGLU FFN: `down(silu(gate(x)) * up(x))`.
pub struct SwiGluFfn<B: Backend> {
    pub gate_proj: Tensor<B, 2>, // [hidden, inter]
    pub up_proj: Tensor<B, 2>,   // [hidden, inter]
    pub down_proj: Tensor<B, 2>, // [inter, hidden]
}

impl<B: Backend> SwiGluFfn<B> {
    pub fn new(
        gate_proj: Tensor<B, 2>,
        up_proj: Tensor<B, 2>,
        down_proj: Tensor<B, 2>,
    ) -> Self {
        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    pub fn with_random_weights(hidden: usize, inter: usize, device: &B::Device) -> Self {
        Self {
            gate_proj: Tensor::<B, 2>::random(
                [hidden, inter],
                Distribution::Default,
                device,
            ),
            up_proj: Tensor::<B, 2>::random(
                [hidden, inter],
                Distribution::Default,
                device,
            ),
            down_proj: Tensor::<B, 2>::random(
                [inter, hidden],
                Distribution::Default,
                device,
            ),
        }
    }

    /// `x: [batch, seq, hidden]` -> `[batch, seq, hidden]`.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Expand 2D weights to 3D with leading dim 1 so they broadcast against [batch, seq, *].
        let gate_w: Tensor<B, 3> = self.gate_proj.clone().unsqueeze();
        let up_w: Tensor<B, 3> = self.up_proj.clone().unsqueeze();
        let down_w: Tensor<B, 3> = self.down_proj.clone().unsqueeze();

        let gate = x.clone().matmul(gate_w);
        let up = x.matmul(up_w);
        silu(gate).mul(up).matmul(down_w)
    }
}
