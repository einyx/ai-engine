use ai_engine_runtime::gguf::q4_0::Q4GgufTensor;

type B = burn_ndarray::NdArray;

#[test]
fn q4_0_decode_one_block_known_values() {
    // Build one Q4_0 block by hand. 18 bytes per block: f16 scale + 16 bytes of nibbles.
    // Scale = 0.5 (encoded as f16).
    // Nibbles for indices 0..16 in low half; 16..32 in high half.
    // Biased: stored 0..15 -> signed -8..7.
    // We want weight[0] = +7 * 0.5 = 3.5, so we need stored = 7+8 = 15 = 0x0F.
    // weight[16] (high nibble at qs[0]) = -7 * 0.5 = -3.5, so stored = -7+8 = 1 = 0x01.
    // Combined byte: high << 4 | low = (1 << 4) | 0x0F = 0x1F.
    let scale_f16_bits: u16 = half::f16::from_f32(0.5).to_bits();
    let mut block = Vec::with_capacity(18);
    block.extend_from_slice(&scale_f16_bits.to_le_bytes());
    block.push(0x1F);
    block.resize(2 + 16, 0x88); // remaining bytes: both nibbles = 8 = signed 0, weight = 0.0
    assert_eq!(block.len(), 18);

    let dev = Default::default();
    // shape [32, 1] meaning 32 in × 1 out — one block total.
    let t = Q4GgufTensor::<B>::from_blocks(block.clone(), [32, 1], &dev).unwrap();
    assert_eq!(t.shape(), [32, 1]);

    let dq = t.dequantize();
    let v: Vec<f32> = dq.into_data().to_vec().unwrap();
    // v[0] = +7 * 0.5 = 3.5
    // v[16] = -7 * 0.5 = -3.5
    // Others = 0.0
    assert!((v[0] - 3.5).abs() < 1e-4, "v[0] = {}", v[0]);
    assert!((v[16] - (-3.5)).abs() < 1e-4, "v[16] = {}", v[16]);
    for (i, val) in v.iter().enumerate().take(16).skip(1) {
        assert!(val.abs() < 1e-4, "v[{i}] = {val}");
    }
    for (i, val) in v.iter().enumerate().take(32).skip(17) {
        assert!(val.abs() < 1e-4, "v[{i}] = {val}");
    }
}

#[test]
fn q4_0_decode_multi_block_tensor_recovers_quantized() {
    let dev = Default::default();
    // Build a [32, 2] tensor (2 columns × 1 block-along-in = 2 blocks).
    // GGUF flat order is column-major over [in, out]:
    //   gguf_flat[k] = value at (in = k % in_dim, out = k / in_dim)
    // Block 0 covers (in=0..32, out=0): all +1.0.
    // Block 1 covers (in=0..32, out=1): all -1.0.
    // Scale 1.0/7 ≈ 0.143; +7 stored → nibble 15 → byte 0xFF for both halves.
    //                       -7 stored → nibble  1 → byte 0x11.
    let scale = 1.0_f32 / 7.0;
    let scale_bits = half::f16::from_f32(scale).to_bits();

    let mut buf = Vec::with_capacity(36);
    // Block 0: all +7, both halves
    buf.extend_from_slice(&scale_bits.to_le_bytes());
    buf.resize(buf.len() + 16, 0xFF);
    // Block 1: all -7, both halves
    buf.extend_from_slice(&scale_bits.to_le_bytes());
    buf.resize(buf.len() + 16, 0x11);

    let t = Q4GgufTensor::<B>::from_blocks(buf, [32, 2], &dev).unwrap();
    let dq = t.dequantize();
    let v: Vec<f32> = dq.into_data().to_vec().unwrap();
    // burn row-major [in, out]: v[i * out + j] = value at (in=i, out=j).
    // (in=0, out=0) should be in block 0 = +1
    // (in=0, out=1) should be in block 1 = -1
    assert!((v[0] - 1.0).abs() < 0.01, "(0,0) = {}", v[0]);
    assert!((v[1] - (-1.0)).abs() < 0.01, "(0,1) = {}", v[1]);
    assert!((v[15 * 2] - 1.0).abs() < 0.01, "(15,0) = {}", v[15 * 2]);
    assert!(
        (v[15 * 2 + 1] - (-1.0)).abs() < 0.01,
        "(15,1) = {}",
        v[15 * 2 + 1]
    );
    // Also spot-check second-half (high nibbles) positions:
    assert!((v[31 * 2] - 1.0).abs() < 0.01, "(31,0) = {}", v[31 * 2]);
    assert!(
        (v[31 * 2 + 1] - (-1.0)).abs() < 0.01,
        "(31,1) = {}",
        v[31 * 2 + 1]
    );
}
