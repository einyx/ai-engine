use burn::tensor::{backend::Backend, Tensor, TensorData};
use std::marker::PhantomData;

/// Per-tensor symmetric Q8 quantization.
///
/// Storage:
///   - `packed`: raw i8 values, length = product of `shape`.
///   - `scale`: single f32 per tensor.
///   - `shape`: original 2-D shape.
///
/// Reconstruction: `weight_f32[i] = packed[i] as f32 * scale`.
///
/// Quantization: `packed[i] = clamp(round(weight_f32[i] / scale), -127, 127)`
/// where `scale = max(|weight_f32|) / 127`. The clamp prevents -128 (asymmetric
/// edge case) so the negation is always representable.
pub struct QuantizedTensor<B: Backend> {
    pub packed: Vec<i8>,
    pub scale: f32,
    shape: [usize; 2],
    _marker: PhantomData<B>,
    device: B::Device,
}

impl<B: Backend> QuantizedTensor<B> {
    /// Quantize an f32 tensor to Q8 with a per-tensor scale.
    pub fn quantize_from(t: Tensor<B, 2>) -> Self {
        let shape = t.dims();
        let device = t.device();
        let values: Vec<f32> = t
            .into_data()
            .to_vec()
            .expect("to_vec f32 from Tensor<B, 2>");
        let max_abs = values
            .iter()
            .copied()
            .fold(0.0_f32, |acc, x| acc.max(x.abs()));
        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
        let packed: Vec<i8> = values
            .iter()
            .map(|&v| ((v / scale).round().clamp(-127.0, 127.0)) as i8)
            .collect();
        Self {
            packed,
            scale,
            shape,
            _marker: PhantomData,
            device,
        }
    }

    /// Construct from raw packed bytes + scale (used by the loader).
    pub fn from_packed(
        packed: Vec<i8>,
        scale: f32,
        shape: [usize; 2],
        device: &B::Device,
    ) -> Self {
        assert_eq!(
            packed.len(),
            shape[0] * shape[1],
            "packed length must match shape product"
        );
        Self {
            packed,
            scale,
            shape,
            _marker: PhantomData,
            device: device.clone(),
        }
    }

    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }

    /// Dequantize to a regular f32 Tensor<B, 2>. Allocates a new f32 buffer.
    pub fn dequantize(&self) -> Tensor<B, 2> {
        let f32_values: Vec<f32> = self.packed.iter().map(|&q| (q as f32) * self.scale).collect();
        Tensor::<B, 2>::from_data(TensorData::new(f32_values, self.shape), &self.device)
    }

    /// Transpose this 2-D quantized tensor directly on the int8 buffer.
    ///
    /// Per-tensor symmetric Q8 has a single scalar `scale`, so transposition
    /// is exactly lossless: only the element ordering changes; the scale and
    /// each int8 value are preserved. This avoids the
    /// dequantize → swap_dims → requantize round-trip, which would re-round
    /// each value through Q8 and accumulate extra error.
    pub fn transpose_2d(&self) -> Self {
        let [r, c] = self.shape;
        let mut packed = vec![0_i8; r * c];
        for i in 0..r {
            for j in 0..c {
                packed[j * r + i] = self.packed[i * c + j];
            }
        }
        Self {
            packed,
            scale: self.scale,
            shape: [c, r],
            _marker: PhantomData,
            device: self.device.clone(),
        }
    }
}
