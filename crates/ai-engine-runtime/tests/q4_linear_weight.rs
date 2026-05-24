use ai_engine_runtime::arch::linear::LinearWeight;
use ai_engine_runtime::quant::Q4Tensor;
use burn::tensor::{Tensor, TensorData};

type B = burn_ndarray::NdArray;

#[test]
fn q4_linear_matmul_approximates_dense() {
    let dev = Default::default();
    // Build [in=32, out=4] dense weight with values that quantize well.
    let raw: Vec<f32> = (0..32 * 4)
        .map(|i| ((i as f32) * 0.07).sin() * 0.5)
        .collect();
    let w = Tensor::<B, 2>::from_data(TensorData::new(raw.clone(), [32, 4]), &dev);

    // [batch=1, seq=1, in=32] activation
    let x_data: Vec<f32> = (0..32).map(|i| ((i as f32) * 0.1).cos()).collect();
    let x = Tensor::<B, 3>::from_data(TensorData::new(x_data, [1, 1, 32]), &dev);

    let dense = LinearWeight::Dense(w.clone());
    let q4 = LinearWeight::Q4(Q4Tensor::<B>::quantize_from(w));

    let out_dense: Vec<f32> = dense.matmul(x.clone()).into_data().to_vec().unwrap();
    let out_q4: Vec<f32> = q4.matmul(x).into_data().to_vec().unwrap();

    let max_diff = out_dense
        .iter()
        .zip(out_q4.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    // 32 multiply-adds per output, each with Q4 noise; cancellation across the sum
    // typically keeps the matmul error well under per-element Q4 noise * sqrt(32).
    assert!(max_diff < 0.5, "Q4 matmul diverged from dense by {max_diff}");
}

#[test]
fn q4_linear_shape_matches_dense() {
    let dev = Default::default();
    let raw: Vec<f32> = (0..64 * 8).map(|i| i as f32 * 0.001).collect();
    let w = Tensor::<B, 2>::from_data(TensorData::new(raw, [64, 8]), &dev);
    let q = LinearWeight::Q4(Q4Tensor::<B>::quantize_from(w));
    assert_eq!(q.shape(), [64, 8]);
}
