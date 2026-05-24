//! SwiGLU FFN: `down(silu(gate(x)) * up(x))`.

use crate::arch::linear::LinearWeight;
use burn::tensor::{Distribution, Tensor, activation::silu, backend::Backend};

/// SwiGLU FFN: `down(silu(gate(x)) * up(x))`.
pub struct SwiGluFfn<B: Backend> {
    pub gate_proj: LinearWeight<B>, // [hidden, inter]
    pub up_proj: LinearWeight<B>,   // [hidden, inter]
    pub down_proj: LinearWeight<B>, // [inter, hidden]
}

impl<B: Backend> SwiGluFfn<B> {
    pub fn new(
        gate_proj: LinearWeight<B>,
        up_proj: LinearWeight<B>,
        down_proj: LinearWeight<B>,
    ) -> Self {
        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    pub fn with_random_weights(hidden: usize, inter: usize, device: &B::Device) -> Self {
        let gate_proj = Tensor::<B, 2>::random(
            [hidden, inter],
            Distribution::Default,
            device,
        );
        let up_proj = Tensor::<B, 2>::random(
            [hidden, inter],
            Distribution::Default,
            device,
        );
        let down_proj = Tensor::<B, 2>::random(
            [inter, hidden],
            Distribution::Default,
            device,
        );
        Self {
            gate_proj: LinearWeight::Dense(gate_proj),
            up_proj: LinearWeight::Dense(up_proj),
            down_proj: LinearWeight::Dense(down_proj),
        }
    }

    /// `x: [batch, seq, hidden]` -> `[batch, seq, hidden]`.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = self.gate_proj.matmul(x.clone());
        let up = self.up_proj.matmul(x);
        self.down_proj.matmul(silu(gate).mul(up))
    }
}
