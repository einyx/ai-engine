//! Per-row rotary position embedding.

use candle_core::{Device, Tensor};

/// Precomputed RoPE tables, shape (max_seq, rope_dim/2) for cos and sin.
pub struct Rope {
    pub cos: Tensor,
    pub sin: Tensor,
}

impl Rope {
    /// Mirror candle-transformers `precomput_freqs_cis`: theta_i = base^(-2i/dim),
    /// table[p, i] = p * theta_i, then cos/sin.
    pub fn new(rope_dim: usize, freq_base: f32, max_seq: usize, device: &Device) -> candle_core::Result<Self> {
        let theta: Vec<f32> = (0..rope_dim / 2)
            .map(|i| 1.0 / freq_base.powf(2.0 * i as f32 / rope_dim as f32))
            .collect();
        let theta = Tensor::new(theta.as_slice(), device)?;
        let positions: Vec<f32> = (0..max_seq).map(|p| p as f32).collect();
        let positions = Tensor::new(positions.as_slice(), device)?;
        let freqs = positions.unsqueeze(1)?.matmul(&theta.unsqueeze(0)?)?;
        Ok(Self { cos: freqs.cos()?, sin: freqs.sin()? })
    }

    /// Gather cos/sin rows for arbitrary per-token positions.
    /// `positions`: u32 tensor of shape (n_tokens,). Returns (cos, sin) each
    /// (n_tokens, rope_dim/2).
    pub fn gather(&self, positions: &Tensor) -> candle_core::Result<(Tensor, Tensor)> {
        let cos = self.cos.index_select(positions, 0)?;
        let sin = self.sin.index_select(positions, 0)?;
        Ok((cos, sin))
    }
}

/// Apply rotary embedding to `x` of shape (n_tokens, n_head, head_dim) using
/// per-token (cos, sin) of shape (n_tokens, head_dim/2). Uses the
/// rotate-half / interleaved convention candle's quantized models use.
pub fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
    candle_nn::rotary_emb::rope_i(&x.contiguous()?, cos, sin)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn gather_picks_correct_rows() {
        let dev = Device::Cpu;
        let r = Rope::new(8, 10000.0, 512, &dev).unwrap();
        let pos = Tensor::new(&[0u32, 200u32], &dev).unwrap();
        let (cos, sin) = r.gather(&pos).unwrap();
        assert_eq!(cos.dims(), &[2, 4]);
        assert_eq!(sin.dims(), &[2, 4]);
        let full0: Vec<f32> = r.cos.narrow(0, 0, 1).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let got0: Vec<f32> = cos.narrow(0, 0, 1).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(full0, got0);
        let full200: Vec<f32> = r.cos.narrow(0, 200, 1).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let got200: Vec<f32> = cos.narrow(0, 1, 1).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(full200, got200);
    }
}
