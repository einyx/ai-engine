use crate::quant::QuantizedTensor;
use burn::tensor::{backend::Backend, Tensor};

/// A Linear's weight matrix — either dense or Q8-quantized.
///
/// Both forms produce the same `[in, out]`-shaped weight from the caller's
/// perspective. `matmul(x: [batch, seq, in]) -> [batch, seq, out]` handles
/// the dispatch.
pub enum LinearWeight<B: Backend> {
    Dense(Tensor<B, 2>),
    Quantized(QuantizedTensor<B>),
}

impl<B: Backend> LinearWeight<B> {
    pub fn shape(&self) -> [usize; 2] {
        match self {
            Self::Dense(t) => t.dims(),
            Self::Quantized(q) => q.shape(),
        }
    }

    /// `x: [batch, seq, in]` -> `[batch, seq, out]`. For quantized weights,
    /// dequantizes the weight matrix once before the matmul.
    pub fn matmul(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            Self::Dense(w) => x.matmul(w.clone().unsqueeze()),
            Self::Quantized(q) => x.matmul(q.dequantize().unsqueeze()),
        }
    }
}
