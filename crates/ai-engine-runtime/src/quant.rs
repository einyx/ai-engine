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

/// Group size along the input dim for per-group symmetric Q4 quantization.
pub const Q4_GROUP_SIZE: usize = 32;

/// Per-group symmetric Q4 (4-bit) weight quantization.
///
/// Storage layout (math order `[in, out]`):
///   - `packed`: `Vec<u8>`, length `in * out / 2`. Two nibbles per byte,
///     row-major. Within byte at `packed[i * (out/2) + j/2]`:
///       * low nibble  (`byte & 0x0F`) is the value at column `j` (even `j`)
///       * high nibble (`(byte >> 4) & 0x0F`) is the value at column `j+1`
///   - `scales`: `Vec<f32>`, length `(in / 32) * out`. One f32 per
///     (input-group, output-channel) pair, indexed as `scales[g * out + j]`.
///   - `shape`: `[in, out]`.
///
/// Reconstruction:
/// ```text
/// for i in 0..in:
///     g = i / 32
///     for j in 0..out:
///         nibble = low or high half of packed byte
///         signed = if nibble < 8 { nibble as i32 } else { nibble as i32 - 16 }
///         weight[i][j] = (signed as f32) * scales[g * out + j]
/// ```
///
/// Quantization clamps to `-7..=7` (not `-8..=7`) to keep the value range
/// symmetric around zero, matching GGUF Q4_0.
pub struct Q4Tensor<B: Backend> {
    pub packed: Vec<u8>,
    pub scales: Vec<f32>,
    shape: [usize; 2],
    _marker: PhantomData<B>,
    device: B::Device,
}

impl<B: Backend> Q4Tensor<B> {
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }

    /// Quantize a dense `[in, out]` f32 tensor with per-group symmetric Q4.
    ///
    /// Panics if `in` is not divisible by 32, or if `out` is odd (we pack two
    /// nibbles per byte along the output dim).
    pub fn quantize_from(t: Tensor<B, 2>) -> Self {
        let shape = t.dims();
        let in_dim = shape[0];
        let out_dim = shape[1];
        assert!(
            in_dim % Q4_GROUP_SIZE == 0,
            "Q4 requires in dim divisible by {Q4_GROUP_SIZE}, got in={in_dim}"
        );
        assert!(
            out_dim % 2 == 0,
            "Q4 requires out dim even (packs 2 nibbles/byte), got out={out_dim}"
        );
        let device = t.device();
        let values: Vec<f32> = t.into_data().to_vec().expect("to_vec f32");

        let num_groups = in_dim / Q4_GROUP_SIZE;
        let mut scales = vec![0.0_f32; num_groups * out_dim];
        let mut packed = vec![0u8; in_dim * (out_dim / 2)];

        // Compute per-(group, out_channel) scale = max(|block|) / 7.
        for g in 0..num_groups {
            for j in 0..out_dim {
                let mut max_abs = 0.0_f32;
                for k in 0..Q4_GROUP_SIZE {
                    let i = g * Q4_GROUP_SIZE + k;
                    let v = values[i * out_dim + j].abs();
                    if v > max_abs {
                        max_abs = v;
                    }
                }
                let s = if max_abs == 0.0 { 1.0 } else { max_abs / 7.0 };
                scales[g * out_dim + j] = s;
            }
        }

        // Pack two nibbles per byte, row-major along the out dim.
        for i in 0..in_dim {
            let g = i / Q4_GROUP_SIZE;
            for j in (0..out_dim).step_by(2) {
                let s_lo = scales[g * out_dim + j];
                let s_hi = scales[g * out_dim + j + 1];
                let v_lo = values[i * out_dim + j];
                let v_hi = values[i * out_dim + j + 1];
                let q_lo = ((v_lo / s_lo).round() as i32).clamp(-7, 7) as i8;
                let q_hi = ((v_hi / s_hi).round() as i32).clamp(-7, 7) as i8;
                let nibble_lo = (q_lo as u8) & 0x0F;
                let nibble_hi = (q_hi as u8) & 0x0F;
                let byte_idx = i * (out_dim / 2) + (j / 2);
                packed[byte_idx] = nibble_lo | (nibble_hi << 4);
            }
        }

        Self {
            packed,
            scales,
            shape,
            _marker: PhantomData,
            device,
        }
    }

    /// Construct from raw packed nibbles + per-group scales (used by the loader).
    pub fn from_packed(
        packed: Vec<u8>,
        scales: Vec<f32>,
        shape: [usize; 2],
        device: &B::Device,
    ) -> Self {
        let in_dim = shape[0];
        let out_dim = shape[1];
        assert_eq!(
            packed.len(),
            in_dim * (out_dim / 2),
            "Q4 packed length must be in * out / 2"
        );
        assert_eq!(
            scales.len(),
            (in_dim / Q4_GROUP_SIZE) * out_dim,
            "Q4 scales length must be (in / 32) * out"
        );
        Self {
            packed,
            scales,
            shape,
            _marker: PhantomData,
            device: device.clone(),
        }
    }

    /// Dequantize to a regular f32 `Tensor<B, 2>`. Allocates a fresh buffer.
    pub fn dequantize(&self) -> Tensor<B, 2> {
        let [in_dim, out_dim] = self.shape;
        let mut f32_values = vec![0.0_f32; in_dim * out_dim];
        for i in 0..in_dim {
            let g = i / Q4_GROUP_SIZE;
            for j in (0..out_dim).step_by(2) {
                let byte = self.packed[i * (out_dim / 2) + (j / 2)];
                let nibble_lo = byte & 0x0F;
                let nibble_hi = (byte >> 4) & 0x0F;
                let q_lo = if nibble_lo < 8 {
                    nibble_lo as i32
                } else {
                    (nibble_lo as i32) - 16
                };
                let q_hi = if nibble_hi < 8 {
                    nibble_hi as i32
                } else {
                    (nibble_hi as i32) - 16
                };
                let s_lo = self.scales[g * out_dim + j];
                let s_hi = self.scales[g * out_dim + j + 1];
                f32_values[i * out_dim + j] = (q_lo as f32) * s_lo;
                f32_values[i * out_dim + j + 1] = (q_hi as f32) * s_hi;
            }
        }
        Tensor::<B, 2>::from_data(TensorData::new(f32_values, [in_dim, out_dim]), &self.device)
    }
}
