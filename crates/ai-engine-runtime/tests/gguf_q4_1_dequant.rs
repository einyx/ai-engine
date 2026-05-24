use ai_engine_runtime::gguf::q4_1::{dequant_q4_1_block, Q4_1_BYTES_PER_BLOCK};

#[test]
fn q4_1_dequant_one_block_known_values() {
    // 20 bytes per block: 2 (d) + 2 (m) + 16 (qs).
    // d = 0.5, m = -3.0, so value = nibble * 0.5 - 3.0.
    // qs[0] low nibble = 6 → weight[0]  = 6*0.5 - 3.0 = 0.0
    //       high nibble = 14 → weight[16] = 14*0.5 - 3.0 = 4.0
    // All other bytes 0x44: low=4 → 4*0.5-3.0 = -1.0
    //                      high=4 → 4*0.5-3.0 = -1.0
    let d_bits = half::f16::from_f32(0.5).to_bits();
    let m_bits = half::f16::from_f32(-3.0).to_bits();
    let mut block = Vec::with_capacity(Q4_1_BYTES_PER_BLOCK);
    block.extend_from_slice(&d_bits.to_le_bytes());
    block.extend_from_slice(&m_bits.to_le_bytes());
    // qs[0] = 0xE6 (high nibble 0xE=14, low nibble 0x6=6).
    block.push(0xE6);
    // qs[1..16] = 0x44.
    block.resize(Q4_1_BYTES_PER_BLOCK, 0x44);
    assert_eq!(block.len(), 20);

    let mut out = [0.0_f32; 32];
    dequant_q4_1_block(&block, &mut out).unwrap();

    assert!((out[0] - 0.0).abs() < 1e-4, "out[0] = {}", out[0]);
    assert!((out[16] - 4.0).abs() < 1e-4, "out[16] = {}", out[16]);
    for (i, v) in out.iter().enumerate().take(16).skip(1) {
        assert!((v - (-1.0)).abs() < 1e-4, "out[{i}] = {v}");
    }
    for (i, v) in out.iter().enumerate().take(32).skip(17) {
        assert!((v - (-1.0)).abs() < 1e-4, "out[{i}] = {v}");
    }
}
