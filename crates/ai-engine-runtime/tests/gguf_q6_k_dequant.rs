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

#[test]
fn q6_k_dequant_recovers_q2_q3_q4_high_bits() {
    // Independently exercises qh shifts at bit positions 2, 4, 6.
    // qh[0] = 0xFC = 0b11111100. So:
    //   bits 0-1 = 0  → q1's high 2 bits = 0
    //   bits 2-3 = 3  → q2's high 2 bits = 3
    //   bits 4-5 = 3  → q3's high 2 bits = 3
    //   bits 6-7 = 3  → q4's high 2 bits = 3
    //
    // All ql bytes = 0x88 (low nibble = high nibble = 8).
    //
    // For l = 0 (is = 0), with d=1.0:
    //   q1 = (8 | 0)       - 32 = -24
    //   q2 = (8 | (3<<4))  - 32 = (8 | 48) - 32 = 24
    //   q3 = (8 | (3<<4))  - 32 = 24
    //   q4 = (8 | (3<<4))  - 32 = 24
    //
    // qh[1..64] = 0, so for l > 0, the high-2-bit contributions are 0
    // (q1=q2=q3=q4=-24 there).
    //
    // We pick scales to isolate the q2/q3/q4 paths cleanly:
    //   scales[0] = 1, scales[2] = 1, scales[4] = 1, scales[6] = 1
    //   all other scales = 0
    //
    // For l = 0 (is = 0):
    //   y[0]  = d * scales[0] * q1 =  1 * 1 * -24 = -24
    //   y[32] = d * scales[2] * q2 =  1 * 1 *  24 =  24
    //   y[64] = d * scales[4] * q3 =  1 * 1 *  24 =  24
    //   y[96] = d * scales[6] * q4 =  1 * 1 *  24 =  24
    //
    // For l ∈ 1..16 (is = 0), qh[l] = 0, so all four q values = -24.
    //   y[l]    = 1 * 1 * -24 = -24
    //   y[l+32] = 1 * 1 * -24 = -24
    //   y[l+64] = 1 * 1 * -24 = -24
    //   y[l+96] = 1 * 1 * -24 = -24
    //
    // For l ∈ 16..32 (is = 1), scales[is + 0/2/4/6] = scales[1/3/5/7] = 0.
    //   All y values in this slice = 0.
    //
    // Second half (l ∈ 0..32, scales_off = 8): scales[8..16] are all 0.
    //   All y values in [128..256] = 0.

    use ai_engine_runtime::gguf::q6_k::{
        dequant_q6_k_superblock, Q6_K_BYTES_PER_SUPERBLOCK, Q6_K_SUPERBLOCK_SIZE,
    };

    let mut block = vec![0u8; Q6_K_BYTES_PER_SUPERBLOCK];
    for b in block.iter_mut().take(128) {
        *b = 0x88;
    }
    block[128] = 0xFC; // qh[0]: bits 2-3 = bits 4-5 = bits 6-7 = 3, bits 0-1 = 0
    block[192] = 1; // scales[0]
    block[194] = 1; // scales[2]
    block[196] = 1; // scales[4]
    block[198] = 1; // scales[6]

    let d_bits = half::f16::from_f32(1.0).to_bits();
    block[208] = (d_bits & 0xFF) as u8;
    block[209] = (d_bits >> 8) as u8;

    let mut out = vec![0.0_f32; Q6_K_SUPERBLOCK_SIZE];
    dequant_q6_k_superblock(&block, &mut out).unwrap();

    // l = 0: q1's high bits are 0, q2/q3/q4's high bits are 3.
    assert!((out[0] - (-24.0)).abs() < 1e-3, "y[0]  = {}, expected -24", out[0]);
    assert!((out[32] - 24.0).abs() < 1e-3, "y[32] = {}, expected  24", out[32]);
    assert!((out[64] - 24.0).abs() < 1e-3, "y[64] = {}, expected  24", out[64]);
    assert!((out[96] - 24.0).abs() < 1e-3, "y[96] = {}, expected  24", out[96]);

    // l ∈ 1..16: qh[l] = 0, so all four q values = -24 → all four y groups = -24.
    for l in 1..16 {
        for &k in &[0usize, 32, 64, 96] {
            let i = l + k;
            assert!(
                (out[i] - (-24.0)).abs() < 1e-3,
                "y[{i}] (l={l}, k={k}) = {}, expected -24",
                out[i]
            );
        }
    }

    // l ∈ 16..32 (is = 1): scales[is + 0/2/4/6] = scales[1/3/5/7] = 0 → y = 0.
    for l in 16..32 {
        for &k in &[0usize, 32, 64, 96] {
            let i = l + k;
            assert!(out[i].abs() < 1e-3, "y[{i}] (l={l}, k={k}) = {}, expected 0", out[i]);
        }
    }

    // Second half: scales[8..16] = 0 → all output 0.
    for (i, v) in out.iter().enumerate().take(256).skip(128) {
        assert!(v.abs() < 1e-3, "y[{i}] = {v}, expected 0 (second half)");
    }
}
