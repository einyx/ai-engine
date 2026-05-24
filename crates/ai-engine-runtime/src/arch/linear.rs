use crate::gguf::q4_0::Q4GgufTensor;
use crate::quant::{Q4Tensor, QuantizedTensor};
use burn::tensor::{backend::Backend, Tensor};

/// A Linear's weight matrix — dense, Q8-quantized, Q4-quantized, or native
/// GGUF Q4_0-quantized.
///
/// All forms produce the same `[in, out]`-shaped weight from the caller's
/// perspective. `matmul(x: [batch, seq, in]) -> [batch, seq, out]` handles
/// the dispatch by dequantizing the quantized variants on each call.
pub enum LinearWeight<B: Backend> {
    Dense(Tensor<B, 2>),
    Quantized(QuantizedTensor<B>),
    Q4(Q4Tensor<B>),
    Q4Gguf(Q4GgufTensor<B>),
}

impl<B: Backend> LinearWeight<B> {
    pub fn shape(&self) -> [usize; 2] {
        match self {
            Self::Dense(t) => t.dims(),
            Self::Quantized(q) => q.shape(),
            Self::Q4(q) => q.shape(),
            Self::Q4Gguf(q) => q.shape(),
        }
    }

    /// `x: [batch, seq, in]` -> `[batch, seq, out]`. For quantized weights,
    /// dequantizes the weight matrix once before the matmul.
    pub fn matmul(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            Self::Dense(w) => x.matmul(w.clone().unsqueeze()),
            Self::Quantized(q) => x.matmul(q.dequantize().unsqueeze()),
            Self::Q4(q) => x.matmul(q.dequantize().unsqueeze()),
            Self::Q4Gguf(q) => x.matmul(q.dequantize_cached().unsqueeze()),
        }
    }

    /// Transpose a 2D linear weight (swap rows/cols). Used to convert
    /// safetensors' `[out, in]` layout to the `[in, out]` layout our matmul
    /// expects.
    ///
    /// - Dense: `Tensor::swap_dims`.
    /// - Quantized (Q8): direct transpose on the i8 buffer — exactly lossless
    ///   because the scale is shape-invariant.
    /// - Q4: dequantize, swap_dims, requantize. This introduces an additional
    ///   Q4 round-trip noise but is only invoked when something explicitly
    ///   asks to swap a Q4 weight (Q4 fixtures are designed to be stored
    ///   pre-transposed, so the loader avoids this path).
    pub fn swap_dims(self, a: usize, b: usize) -> Self {
        match self {
            Self::Dense(t) => Self::Dense(t.swap_dims(a, b)),
            Self::Quantized(q) => {
                assert!(
                    (a == 0 && b == 1) || (a == 1 && b == 0),
                    "quantized swap_dims only supports 2-D transpose (0,1)"
                );
                Self::Quantized(q.transpose_2d())
            }
            Self::Q4(q) => {
                assert!(
                    (a == 0 && b == 1) || (a == 1 && b == 0),
                    "Q4 swap_dims only supports 2-D transpose (0,1)"
                );
                let dq = q.dequantize().swap_dims(a, b);
                Self::Q4(Q4Tensor::quantize_from(dq))
            }
            Self::Q4Gguf(_) => {
                // GGUF Q4_0 weights are always loaded in math order `[in, out]`
                // (GGUF's native shape convention). `swap_dims` is conceptually
                // wrong here — callers should use `ensure_math_order` instead,
                // which is a no-op for this variant. If we DO get called, a
                // force-dequantize + swap + re-encode would be lossy; we panic
                // to surface the misuse.
                panic!("swap_dims called on Q4Gguf — use ensure_math_order instead");
            }
        }
    }

    /// Ensure the weight is in math order `[in, out]`.
    ///
    /// Q4 and Q4Gguf fixtures store weights pre-transposed (already in
    /// `[in, out]` math order, so the loader never has to call `swap_dims`
    /// and re-quantize). Dense and Q8 fixtures follow HF's `[out, in]` layout
    /// and need the `swap_dims(0, 1)` flip at load time.
    ///
    /// Callers that previously called `<weight>.swap_dims(0, 1)` immediately
    /// after loading should use this helper instead — it dispatches correctly
    /// across all four variants.
    pub fn ensure_math_order(self) -> Self {
        match self {
            Self::Q4(_) | Self::Q4Gguf(_) => self,
            _ => self.swap_dims(0, 1),
        }
    }
}
