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

    /// Transpose a 2D linear weight (swap rows/cols). Used to convert
    /// safetensors' `[out, in]` layout to the `[in, out]` layout our matmul
    /// expects.
    ///
    /// For the Dense variant this is just `Tensor::swap_dims`. For the
    /// Quantized variant we dequantize, swap, and requantize — losing a
    /// tiny bit of precision per transpose. Acceptable because this happens
    /// once per load.
    pub fn swap_dims(self, a: usize, b: usize) -> Self {
        match self {
            Self::Dense(t) => Self::Dense(t.swap_dims(a, b)),
            Self::Quantized(q) => {
                let dq = q.dequantize().swap_dims(a, b);
                Self::Quantized(QuantizedTensor::quantize_from(dq))
            }
        }
    }
}
