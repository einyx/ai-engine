use ai_engine_runtime::quant::Q4Tensor;
use burn::tensor::{Tensor, TensorData};

type B = burn_ndarray::NdArray;

#[test]
fn q4_quantize_then_dequantize_recovers_within_4bit_noise() {
    let dev = Default::default();
    // Build a known f32 weight, in=32 (one group), out=4.
    // Values chosen so int4 quantization has bounded error.
    let raw: Vec<f32> = (0..32 * 4).map(|i| ((i as f32) * 0.05).sin()).collect();
    let original = Tensor::<B, 2>::from_data(TensorData::new(raw.clone(), [32, 4]), &dev);

    let q = Q4Tensor::<B>::quantize_from(original.clone());
    assert_eq!(q.shape(), [32, 4]);
    assert_eq!(q.packed.len(), 32 * 2); // 32 rows × (4 cols / 2 cols/byte) = 64 bytes
    assert_eq!(q.scales.len(), 4); // 1 group × 4 cols

    let recovered = q.dequantize();
    let original_v: Vec<f32> = original.into_data().to_vec().unwrap();
    let recovered_v: Vec<f32> = recovered.into_data().to_vec().unwrap();

    // Q4 with signed -7..7 range, per-group scale s = max|block|/7.
    // Worst-case rounding error per value is s/2 ~ max|block|/14.
    // For our sin-based input, max|block| < 1, so error per value < ~0.07.
    let max_err = original_v
        .iter()
        .zip(recovered_v.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_err < 0.08,
        "Q4 quantization error {max_err} exceeded expected ~0.07"
    );
}

#[test]
fn q4_storage_size_is_quarter_of_dense() {
    // 32 × 8 = 256 weights.
    // Dense f32: 256 × 4 = 1024 bytes.
    // Q4: 256 × 0.5 = 128 bytes packed + 1 group × 8 cols × 4 = 32 bytes scales = 160 bytes total.
    let dev = Default::default();
    let raw: Vec<f32> = (0..32 * 8).map(|i| i as f32 * 0.01).collect();
    let t = Tensor::<B, 2>::from_data(TensorData::new(raw, [32, 8]), &dev);
    let q = Q4Tensor::<B>::quantize_from(t);
    assert_eq!(q.packed.len(), 128); // 32 × 4
    assert_eq!(q.scales.len(), 8); // 1 group × 8 cols
}

#[test]
fn q4_from_packed_components_roundtrips() {
    let dev = Default::default();
    // One group, two columns.
    // Col 0: alternating 7, -7 across 32 rows; scale 0.5 -> values 3.5, -3.5.
    // Col 1: zeros; scale 1.0 -> values 0.
    // Packed[i, 0]: low nibble = col0 nibble, high nibble = col1 nibble (0).
    // col0 = 7 -> 0x07; col0 = -7 -> nibble 0x09 (since -7 in i4 two's complement -> 9).
    let mut packed = Vec::with_capacity(32);
    for i in 0..32 {
        if i % 2 == 0 {
            packed.push(0x07);
        } else {
            packed.push(0x09);
        }
    }
    let scales = vec![0.5_f32, 1.0_f32];
    let q = Q4Tensor::<B>::from_packed(packed, scales, [32, 2], &dev);
    let d = q.dequantize();
    let v: Vec<f32> = d.into_data().to_vec().unwrap();
    for i in 0..32 {
        let c0 = v[i * 2];
        let c1 = v[i * 2 + 1];
        let expected_c0 = if i % 2 == 0 { 3.5 } else { -3.5 };
        assert!(
            (c0 - expected_c0).abs() < 1e-5,
            "row {i} col 0: {c0} != {expected_c0}"
        );
        assert!(c1.abs() < 1e-5, "row {i} col 1: {c1} != 0");
    }
}
