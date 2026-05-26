//! GGUF Q4_1 tensor: 32-weight blocks with f16 scale + f16 min + 16 bytes of nibbles.
//!
//! Block layout (20 bytes total):
//!   - bytes 0..2  : f16 `d`  (scale/delta)
//!   - bytes 2..4  : f16 `m`  (min/offset)
//!   - bytes 4..20 : `qs[0..16]` packed nibbles
//!
//! Nibble convention (per ggml `block_q4_1`):
//!   low nibble  of `qs[j]` (== `qs[j] & 0x0F`)        → weight at block index `j`
//!   high nibble of `qs[j]` (== `(qs[j] >> 4) & 0x0F`) → weight at block index `j + 16`
//!   nibbles are UNBIASED (range 0..15, unlike Q4_0's -8..7).
//!   value = (nibble as f32) * d + m

use half::f16;

pub const Q4_1_BLOCK_SIZE: usize = 32;
pub const Q4_1_BYTES_PER_BLOCK: usize = 20;

/// Dequantize a single Q4_1 block (20 input bytes → 32 output f32s).
pub fn dequant_q4_1_block(block: &[u8], out: &mut [f32]) -> anyhow::Result<()> {
    if block.len() != Q4_1_BYTES_PER_BLOCK {
        anyhow::bail!(
            "Q4_1 block: expected {Q4_1_BYTES_PER_BLOCK} bytes, got {}",
            block.len()
        );
    }
    if out.len() != Q4_1_BLOCK_SIZE {
        anyhow::bail!(
            "Q4_1 out: expected {Q4_1_BLOCK_SIZE} f32 slots, got {}",
            out.len()
        );
    }
    let d = f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
    let m = f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();
    for j in 0..16 {
        let byte = block[4 + j];
        let low = (byte & 0x0F) as f32;
        let high = ((byte >> 4) & 0x0F) as f32;
        out[j] = low * d + m;
        out[j + 16] = high * d + m;
    }
    Ok(())
}

/// Dequantize a whole Q4_1 tensor in GGUF's flat (column-major over [in, out]) order.
/// `bytes` must be `(total / 32) * 20` long; returns a `total`-length Vec<f32>.
pub fn dequant_q4_1_tensor(bytes: &[u8], total: usize) -> anyhow::Result<Vec<f32>> {
    if total % Q4_1_BLOCK_SIZE != 0 {
        anyhow::bail!(
            "Q4_1 requires total elements ({total}) divisible by {Q4_1_BLOCK_SIZE}"
        );
    }
    let num_blocks = total / Q4_1_BLOCK_SIZE;
    let expected_bytes = num_blocks * Q4_1_BYTES_PER_BLOCK;
    if bytes.len() != expected_bytes {
        anyhow::bail!(
            "Q4_1 bytes len mismatch: expected {expected_bytes}, got {}",
            bytes.len()
        );
    }
    let mut out = vec![0.0_f32; total];
    for b in 0..num_blocks {
        let block_off = b * Q4_1_BYTES_PER_BLOCK;
        let out_off = b * Q4_1_BLOCK_SIZE;
        dequant_q4_1_block(
            &bytes[block_off..block_off + Q4_1_BYTES_PER_BLOCK],
            &mut out[out_off..out_off + Q4_1_BLOCK_SIZE],
        )?;
    }
    Ok(out)
}
