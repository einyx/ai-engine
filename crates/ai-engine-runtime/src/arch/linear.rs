use crate::gguf::q4_0::Q4GgufTensor;
use crate::quant::{Q4Tensor, QuantizedTensor};
use burn::tensor::{backend::Backend, Tensor, TensorData};

#[cfg(feature = "backend-cpu")]
use {
    ndarray::{ArcArray, Array, ArrayView1, ArrayViewMut1, Ix2},
    ndarray::linalg::general_mat_vec_mul,
    std::sync::{Arc, OnceLock},
};

/// A Linear's weight matrix — dense, Q8-quantized, Q4-quantized, or native
/// GGUF Q4_0-quantized.
///
/// All forms produce the same `[in, out]`-shaped weight from the caller's
/// perspective. `matmul(x: [batch, seq, in]) -> [batch, seq, out]` handles
/// the dispatch by dequantizing the quantized variants on each call.
///
/// For seq=1 decode on the CPU backend, each variant maintains a lazily-
/// populated ndarray weight cache and dispatches through BLAS `sgemv`
/// instead of `sgemm`, giving ~5× speedup per matmul layer.
pub enum LinearWeight<B: Backend> {
    Dense(DenseWeight<B>),
    Quantized(QuantizedTensor<B>),
    Q4(Q4Tensor<B>),
    Q4Gguf(Q4GgufTensor<B>),
}

/// Dense weight with an embedded GEMV cache for the seq=1 fast path.
pub struct DenseWeight<B: Backend> {
    pub tensor: Tensor<B, 2>,
    #[cfg(feature = "backend-cpu")]
    gemv: OnceLock<Arc<ArcArray<f32, Ix2>>>,
}

impl<B: Backend> DenseWeight<B> {
    pub fn new(tensor: Tensor<B, 2>) -> Self {
        Self {
            tensor,
            #[cfg(feature = "backend-cpu")]
            gemv: OnceLock::new(),
        }
    }

    pub fn dims(&self) -> [usize; 2] {
        self.tensor.dims()
    }

    pub fn swap_dims(self, a: usize, b: usize) -> Self {
        // Discard the GEMV cache when the weight is transposed.
        Self::new(self.tensor.swap_dims(a, b))
    }

    /// Return (or build) the contiguous ndarray weight for BLAS sgemv.
    #[cfg(feature = "backend-cpu")]
    fn ndarray_weight(&self) -> Arc<ArcArray<f32, Ix2>> {
        self.gemv
            .get_or_init(|| {
                let [in_dim, out_dim] = self.tensor.dims();
                let floats: Vec<f32> = self
                    .tensor
                    .to_data()
                    .to_vec()
                    .expect("Dense weight must be f32");
                let arr = Array::from_shape_vec([in_dim, out_dim], floats)
                    .expect("shape/data mismatch in Dense ndarray cache")
                    .into_shared();
                Arc::new(arr)
            })
            .clone()
    }
}

// ── LinearWeight ──────────────────────────────────────────────────────────────

impl<B: Backend> LinearWeight<B> {
    /// Convenience constructor: wraps a `Tensor<B, 2>` in the `Dense` variant.
    pub fn dense(t: Tensor<B, 2>) -> Self {
        Self::Dense(DenseWeight::new(t))
    }

    /// Eagerly populate the GEMV ndarray weight cache.
    ///
    /// Call this at model-load time (after `ensure_math_order`) to pay the
    /// one-time dequant+copy cost up front rather than on the first decode
    /// step.  On non-cpu backends this is a no-op.
    pub fn preload_gemv_cache(&self) {
        #[cfg(feature = "backend-cpu")]
        { let _ = self.get_ndarray_weight(); }
    }

    pub fn shape(&self) -> [usize; 2] {
        match self {
            Self::Dense(d) => d.dims(),
            Self::Quantized(q) => q.shape(),
            Self::Q4(q) => q.shape(),
            Self::Q4Gguf(q) => q.shape(),
        }
    }

    /// `x: [batch, seq, in]` → `[batch, seq, out]`.
    ///
    /// When `seq == 1` and the `backend-cpu` feature is active, dispatches
    /// through BLAS `sgemv` (via ndarray) for a ~5× per-layer decode speedup.
    /// For all other shapes falls back to the standard `Tensor::matmul` path.
    pub fn matmul(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_batch, seq, _in] = x.dims();
        #[cfg(feature = "backend-cpu")]
        if seq == 1 {
            return self.matmul_gemv(x);
        }
        self.matmul_standard(x)
    }

    /// Standard `Tensor::matmul` path (all backends, all seq lengths).
    fn matmul_standard(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            Self::Dense(d) => x.matmul(d.tensor.clone().unsqueeze()),
            Self::Quantized(q) => x.matmul(q.dequantize().unsqueeze()),
            Self::Q4(q) => x.matmul(q.dequantize().unsqueeze()),
            Self::Q4Gguf(q) => x.matmul(q.dequantize_cached().unsqueeze()),
        }
    }

    /// GEMV fast path: seq=1, backend-cpu only.
    ///
    /// Lazily initializes a contiguous ndarray weight (one copy per weight
    /// matrix, cached after the first call).  For `Q4Gguf` the ndarray cache
    /// is co-located with the existing burn dequant cache on the tensor.
    /// For `Dense` it is stored alongside the `DenseWeight`.
    ///
    /// Per-call: extracts `x` (2048 f32 ≈ 8 KiB), runs BLAS `sgemv`, and
    /// reconstructs a `Tensor<B, 3>` from the output (8192 f32 ≈ 32 KiB).
    #[cfg(feature = "backend-cpu")]
    fn matmul_gemv(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, _seq, in_dim] = x.dims();
        let [_in, out_dim] = self.shape();
        let dev = x.device();

        // Get the ndarray weight (lazily cached per variant).
        let w_arc = self.get_ndarray_weight();
        // w: [in, out] C-order.  w.t(): [out, in] column view — no copy.
        let w_t = w_arc.t();

        // Extract x as flat f32 from [batch, 1, in_dim] → [batch * in_dim].
        let x_data: Vec<f32> = x
            .reshape([batch, in_dim])
            .into_data()
            .to_vec()
            .expect("x must be f32");

        // Run sgemv for each batch element.
        let mut out_flat = vec![0.0_f32; batch * out_dim];
        for b in 0..batch {
            let x_row = ArrayView1::from(&x_data[b * in_dim..(b + 1) * in_dim]);
            let mut out_row =
                ArrayViewMut1::from(&mut out_flat[b * out_dim..(b + 1) * out_dim]);
            // y = 1.0 * w.t() * x_row + 0.0 * y
            // w.t(): [out, in], x_row: [in] → out_row: [out]
            general_mat_vec_mul(1.0_f32, &w_t, &x_row, 0.0_f32, &mut out_row);
        }

        // Reconstruct as Tensor<B, 3> [batch, 1, out_dim].
        Tensor::<B, 3>::from_data(
            TensorData::new(out_flat, [batch, 1usize, out_dim]),
            &dev,
        )
    }

    /// Return (or initialize) the ndarray weight for BLAS sgemv.
    ///
    /// - `Dense`: per-`DenseWeight` OnceLock.
    /// - `Q4Gguf`: per-`Q4GgufTensor` OnceLock (reuses the dequant cache).
    /// - `Q4` / `Quantized`: no persistent cache — re-dequantized each call.
    ///   These are rare in the decode hot path (GGUF models use `Q4Gguf`).
    #[cfg(feature = "backend-cpu")]
    fn get_ndarray_weight(&self) -> Arc<ArcArray<f32, Ix2>> {
        match self {
            Self::Dense(d) => d.ndarray_weight(),
            Self::Q4Gguf(q) => q.ndarray_weight(),
            Self::Quantized(q) => {
                // Not a common decode path; no persistent cache.
                let t = q.dequantize();
                let [in_dim, out_dim] = t.dims();
                let floats: Vec<f32> = t.into_data().to_vec().expect("must be f32");
                Arc::new(
                    Array::from_shape_vec([in_dim, out_dim], floats)
                        .unwrap()
                        .into_shared(),
                )
            }
            Self::Q4(q) => {
                let t = q.dequantize();
                let [in_dim, out_dim] = t.dims();
                let floats: Vec<f32> = t.into_data().to_vec().expect("must be f32");
                Arc::new(
                    Array::from_shape_vec([in_dim, out_dim], floats)
                        .unwrap()
                        .into_shared(),
                )
            }
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
            Self::Dense(d) => Self::Dense(d.swap_dims(a, b)),
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
