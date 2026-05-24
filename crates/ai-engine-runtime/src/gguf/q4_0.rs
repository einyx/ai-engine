//! GGUF Q4_0 tensor: 32-weight blocks with f16 scale + 16 bytes of biased nibbles.
//!
//! Block layout (18 bytes total):
//!   - bytes 0..2  : f16 scale (delta) for the block
//!   - bytes 2..18 : `qs[0..16]` packed nibbles
//!
//! Nibble convention (per ggml.c block_q4_0):
//!   low nibble  of `qs[j]` (== `qs[j] & 0x0F`)        → weight at block index `j`
//!   high nibble of `qs[j]` (== `(qs[j] >> 4) & 0x0F`) → weight at block index `j + 16`
//!   stored 0..15 minus 8 yields signed range -8..7.
//!   value = signed_nibble * scale_f32
//!
//! Tensor flat order (column-major over `[in, out]`):
//!   gguf_flat[k] = value at (in = k % in_dim, out = k / in_dim)
//! `dequantize` transposes into burn's row-major layout:
//!   burn_flat[i * out + j] = gguf_flat[j * in + i]

use burn::tensor::{backend::Backend, Tensor, TensorData};
use half::f16;
use std::marker::PhantomData;

pub const Q4_0_BLOCK_SIZE: usize = 32;
pub const Q4_0_BYTES_PER_BLOCK: usize = 18;

pub struct Q4GgufTensor<B: Backend> {
    pub blocks: Vec<u8>,
    shape: [usize; 2],
    device: B::Device,
    _marker: PhantomData<B>,
}

impl<B: Backend> Q4GgufTensor<B> {
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }

    pub fn from_blocks(
        blocks: Vec<u8>,
        shape: [usize; 2],
        device: &B::Device,
    ) -> anyhow::Result<Self> {
        let in_dim = shape[0];
        let out_dim = shape[1];
        let total = in_dim * out_dim;
        if total % Q4_0_BLOCK_SIZE != 0 {
            anyhow::bail!(
                "Q4_0 requires total elements ({total}) divisible by {Q4_0_BLOCK_SIZE}"
            );
        }
        let expected_bytes = (total / Q4_0_BLOCK_SIZE) * Q4_0_BYTES_PER_BLOCK;
        if blocks.len() != expected_bytes {
            anyhow::bail!(
                "Q4_0 block bytes len mismatch: expected {expected_bytes}, got {}",
                blocks.len()
            );
        }
        Ok(Self {
            blocks,
            shape,
            device: device.clone(),
            _marker: PhantomData,
        })
    }

    /// Reconstruct an f32 Tensor<B, 2> of shape [in, out] in burn's row-major layout.
    pub fn dequantize(&self) -> Tensor<B, 2> {
        let in_dim = self.shape[0];
        let out_dim = self.shape[1];
        let total = in_dim * out_dim;
        let num_blocks = total / Q4_0_BLOCK_SIZE;

        // First pass: reconstruct in GGUF's flat (column-major over [in, out]) order.
        let mut gguf_flat = vec![0.0_f32; total];
        for b in 0..num_blocks {
            let off = b * Q4_0_BYTES_PER_BLOCK;
            let scale_bits = u16::from_le_bytes([self.blocks[off], self.blocks[off + 1]]);
            let scale = f16::from_bits(scale_bits).to_f32();
            for j in 0..16 {
                let byte = self.blocks[off + 2 + j];
                let low = (byte & 0x0F) as i32 - 8;
                let high = ((byte >> 4) & 0x0F) as i32 - 8;
                gguf_flat[b * Q4_0_BLOCK_SIZE + j] = (low as f32) * scale;
                gguf_flat[b * Q4_0_BLOCK_SIZE + j + 16] = (high as f32) * scale;
            }
        }

        // Transpose column-major [in, out] -> burn row-major [in, out]:
        //   burn_flat[i * out + j] = gguf_flat[j * in + i]
        let mut burn_flat = vec![0.0_f32; total];
        for i in 0..in_dim {
            for j in 0..out_dim {
                burn_flat[i * out_dim + j] = gguf_flat[j * in_dim + i];
            }
        }

        Tensor::<B, 2>::from_data(TensorData::new(burn_flat, [in_dim, out_dim]), &self.device)
    }
}
