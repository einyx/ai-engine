use ai_engine_runtime::quant::QuantizedTensor;
use burn::tensor::{Tensor, TensorData};

type B = burn_ndarray::NdArray;

#[test]
fn quantize_then_dequantize_recovers_original_within_q8_noise() {
    let dev = Default::default();
    // Build a known dense matrix.
    let original_f32 = vec![
        1.0_f32, -0.5, 0.25, -0.125,
        0.6, -0.6, 0.1, -0.1,
    ];
    let original = Tensor::<B, 2>::from_data(
        TensorData::new(original_f32.clone(), [2, 4]),
        &dev,
    );

    let q = QuantizedTensor::<B>::quantize_from(original.clone());
    let recovered = q.dequantize();

    let original_v: Vec<f32> = original.into_data().to_vec().unwrap();
    let recovered_v: Vec<f32> = recovered.into_data().to_vec().unwrap();

    let max_err = original_v.iter().zip(recovered_v.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    // Q8 with per-tensor scale, signed range -128..127, max abs of inputs = 1.0
    // -> scale = 1.0 / 127 ~ 0.00787. Worst-case quantization error per value
    // is about scale/2 ~ 0.004.
    assert!(max_err < 0.005, "quantization error {max_err} exceeded Q8 bound");
}

#[test]
fn quantized_tensor_stores_int8_bytes_not_f32() {
    let dev = Default::default();
    let original = Tensor::<B, 2>::from_data(
        TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0], [2, 2]),
        &dev,
    );
    let q = QuantizedTensor::<B>::quantize_from(original);
    // Storage: 2x2 = 4 i8 values = 4 bytes, plus an f32 scale.
    assert_eq!(q.packed.len(), 4);
    assert!(q.scale > 0.0);
    assert_eq!(q.shape(), [2, 2]);
}

#[test]
fn quantized_tensor_from_raw_components_roundtrips() {
    let dev = Default::default();
    let packed = vec![127_i8, -127, 0, 64];
    let scale = 0.5_f32;
    let q = QuantizedTensor::<B>::from_packed(packed.clone(), scale, [2, 2], &dev);
    assert_eq!(q.shape(), [2, 2]);
    let d = q.dequantize();
    let v: Vec<f32> = d.into_data().to_vec().unwrap();
    // 127 * 0.5 = 63.5; -127 * 0.5 = -63.5; 0 * 0.5 = 0; 64 * 0.5 = 32
    assert!((v[0] - 63.5).abs() < 1e-4);
    assert!((v[1] - -63.5).abs() < 1e-4);
    assert!(v[2].abs() < 1e-4);
    assert!((v[3] - 32.0).abs() < 1e-4);
}
