use ai_engine_runtime::gguf::q6_k::{
    dequant_q6_k_superblock, Q6_K_BYTES_PER_SUPERBLOCK, Q6_K_SUPERBLOCK_SIZE,
};

#[test]
fn q6_k_dequant_one_superblock_known_values() {
    // 210 bytes = 128 (ql) + 64 (qh) + 16 (scales) + 2 (d).
    // d = 0.5 (f16).
    // scales[0] = 2 (i8), all other scales = 0.
    // All ql = 0x88 (low nibble = 8, high nibble = 8).
    // All qh = 0x00 (high-2-bit contributions = 0 for every quad).
    //
    // Pre-bias q for every position = 8 | (0 << 4) = 8.
    // Post-bias q = 8 - 32 = -24.
    // y[i] = d * scales[scale_idx] * -24 = 0.5 * scales[scale_idx] * -24
    //
    // Loop layout reminder:
    //   half 0 (n=0):
    //     l=0..15:  scales[0]=2  → y[l]    = -24
    //     l=16..31: scales[1]=0  → y[l]    =   0
    //     l=0..15:  scales[2]=0  → y[l+32] =   0
    //     ... all others scales[2..8] = 0 → 0
    //   half 1 (n=128): uses scales[8..16] = 0 → all 0
    //
    // So we expect y[0..16] = -24, everything else = 0.

    let mut block = vec![0u8; Q6_K_BYTES_PER_SUPERBLOCK];
    for b in block.iter_mut().take(128) {
        *b = 0x88;
    }
    // qh stays 0, scales stay 0 except scales[0] = 2.
    block[192] = 2;

    // d at bytes 208..210.
    let d_bits = half::f16::from_f32(0.5).to_bits();
    block[208] = (d_bits & 0xFF) as u8;
    block[209] = (d_bits >> 8) as u8;

    let mut out = vec![0.0_f32; Q6_K_SUPERBLOCK_SIZE];
    dequant_q6_k_superblock(&block, &mut out).unwrap();

    for (i, v) in out.iter().enumerate().take(16) {
        assert!((v - (-24.0)).abs() < 1e-3, "y[{i}] = {v}, expected -24");
    }
    for (i, v) in out.iter().enumerate().take(256).skip(16) {
        assert!(v.abs() < 1e-3, "y[{i}] = {v}, expected 0");
    }
}

#[test]
fn q6_k_dequant_recovers_high_bit_when_qh_set() {
    // Set qh[0] = 0b00000011, so q1's high 2 bits = 3.
    // Pre-bias q1 = (ql[0] & 0xF) | (3 << 4) = 8 | 48 = 56.
    // Post-bias q1 = 56 - 32 = 24.
    // d=1.0, scales[0]=1 → y[0] = 1*1*24 = 24.
    // Other elements still use ql=0x88, qh=0 → q = -24 (same as before).
    // But scales[1..16]=0 except we also need scales[2..8] = 0 for half-0 to be zero
    // outside y[0..16]. We set all scales 0 except scales[0]=1, so:
    //   y[0]    = +24 (qh bit override on element 0)
    //   y[1..16] = -24 (scales[0]=1, q=-24)
    //   y[16..32] = 0 (scales[1]=0)
    //   y[32..]   = 0 (scales[2..16] = 0)

    let mut block = vec![0u8; Q6_K_BYTES_PER_SUPERBLOCK];
    for b in block.iter_mut().take(128) {
        *b = 0x88;
    }
    block[128] = 0x03; // qh[0]'s low 2 bits = 3 → high 2 bits of q1
    block[192] = 1; // scales[0] = 1

    let d_bits = half::f16::from_f32(1.0).to_bits();
    block[208] = (d_bits & 0xFF) as u8;
    block[209] = (d_bits >> 8) as u8;

    let mut out = vec![0.0_f32; Q6_K_SUPERBLOCK_SIZE];
    dequant_q6_k_superblock(&block, &mut out).unwrap();

    assert!((out[0] - 24.0).abs() < 1e-3, "y[0] = {}, expected 24", out[0]);
    for (i, v) in out.iter().enumerate().take(16).skip(1) {
        assert!((v - (-24.0)).abs() < 1e-3, "y[{i}] = {v}, expected -24");
    }
    for (i, v) in out.iter().enumerate().take(256).skip(16) {
        assert!(v.abs() < 1e-3, "y[{i}] = {v}, expected 0");
    }
}
