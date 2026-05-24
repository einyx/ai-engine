//! GGUF Q6_K tensor: 256-element superblocks, 6 bits per weight, with per-16-element
//! sub-block scales packed inside each superblock.
//!
//! Superblock layout (210 bytes total, for 256 weights):
//!   bytes   0..128 : `ql[128]` low 4 bits of each weight (2 weights per byte)
//!   bytes 128..192 : `qh[64]`  high 2 bits of each weight (4 weights per byte)
//!   bytes 192..208 : `scales[16]` i8, per-16-element sub-block scale
//!   bytes 208..210 : `d` (f16) super-scale
//!
//! Reference: ggml-quants.c::dequantize_row_q6_K
//!
//! Per-element dequant: for each half (n ∈ {0, 128}) of 128 elements, for l ∈ 0..32:
//!   is = l / 16
//!   q1 = ((ql[l]    & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32   → y[l +  0]
//!   q2 = ((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32   → y[l + 32]
//!   q3 = ( ql[l]     >> 4  | (((qh[l] >> 4) & 3) << 4)) - 32   → y[l + 64]
//!   q4 = ( ql[l+32]  >> 4  | (((qh[l] >> 6) & 3) << 4)) - 32   → y[l + 96]
//!   y[l + k*32] = d * scales[is + 2*k] * qN
//! After processing the half: y += 128, ql += 64, qh += 32, scales_off += 8.

use half::f16;

pub const Q6_K_SUPERBLOCK_SIZE: usize = 256;
pub const Q6_K_BYTES_PER_SUPERBLOCK: usize = 128 + 64 + 16 + 2; // 210

/// Dequantize a single Q6_K superblock (210 input bytes → 256 output f32s).
pub fn dequant_q6_k_superblock(block: &[u8], out: &mut [f32]) -> anyhow::Result<()> {
    if block.len() != Q6_K_BYTES_PER_SUPERBLOCK {
        anyhow::bail!(
            "Q6_K superblock: expected {Q6_K_BYTES_PER_SUPERBLOCK} bytes, got {}",
            block.len()
        );
    }
    if out.len() != Q6_K_SUPERBLOCK_SIZE {
        anyhow::bail!(
            "Q6_K out: expected {Q6_K_SUPERBLOCK_SIZE} f32 slots, got {}",
            out.len()
        );
    }

    let d = f16::from_bits(u16::from_le_bytes([block[208], block[209]])).to_f32();
    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales_raw = &block[192..208];

    let mut y_off = 0usize;
    let mut ql_off = 0usize;
    let mut qh_off = 0usize;
    let mut scales_off = 0usize;

    for _half in 0..2 {
        for l in 0..32 {
            let is = l / 16;
            let ql_lo = ql[ql_off + l] as i32;
            let ql_hi = ql[ql_off + l + 32] as i32;
            let qhb = qh[qh_off + l] as i32;

            let q1 = ((ql_lo & 0x0F) | ((qhb & 0x3) << 4)) - 32;
            let q2 = ((ql_hi & 0x0F) | (((qhb >> 2) & 0x3) << 4)) - 32;
            let q3 = ((ql_lo >> 4) | (((qhb >> 4) & 0x3) << 4)) - 32;
            let q4 = ((ql_hi >> 4) | (((qhb >> 6) & 0x3) << 4)) - 32;

            let s0 = scales_raw[scales_off + is] as i8 as f32;
            let s1 = scales_raw[scales_off + is + 2] as i8 as f32;
            let s2 = scales_raw[scales_off + is + 4] as i8 as f32;
            let s3 = scales_raw[scales_off + is + 6] as i8 as f32;

            out[y_off + l] = d * s0 * (q1 as f32);
            out[y_off + l + 32] = d * s1 * (q2 as f32);
            out[y_off + l + 64] = d * s2 * (q3 as f32);
            out[y_off + l + 96] = d * s3 * (q4 as f32);
        }
        y_off += 128;
        ql_off += 64;
        qh_off += 32;
        scales_off += 8;
    }
    Ok(())
}

/// Dequantize a whole Q6_K tensor in GGUF's flat (column-major over [in, out]) order.
/// `bytes` must be `(total / 256) * 210` long; returns a `total`-length Vec<f32>.
pub fn dequant_q6_k_tensor(bytes: &[u8], total: usize) -> anyhow::Result<Vec<f32>> {
    if total % Q6_K_SUPERBLOCK_SIZE != 0 {
        anyhow::bail!(
            "Q6_K requires total elements ({total}) divisible by {Q6_K_SUPERBLOCK_SIZE}"
        );
    }
    let num_blocks = total / Q6_K_SUPERBLOCK_SIZE;
    let expected_bytes = num_blocks * Q6_K_BYTES_PER_SUPERBLOCK;
    if bytes.len() != expected_bytes {
        anyhow::bail!(
            "Q6_K bytes len mismatch: expected {expected_bytes}, got {}",
            bytes.len()
        );
    }
    let mut out = vec![0.0_f32; total];
    for b in 0..num_blocks {
        let block_off = b * Q6_K_BYTES_PER_SUPERBLOCK;
        let out_off = b * Q6_K_SUPERBLOCK_SIZE;
        dequant_q6_k_superblock(
            &bytes[block_off..block_off + Q6_K_BYTES_PER_SUPERBLOCK],
            &mut out[out_off..out_off + Q6_K_SUPERBLOCK_SIZE],
        )?;
    }
    Ok(out)
}
