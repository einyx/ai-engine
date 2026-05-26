use ai_engine_runtime::arch::linear::LinearWeight;
use ai_engine_runtime::quant::QuantizedTensor;
use burn::tensor::{Tensor, TensorData};

type B = burn_ndarray::NdArray;

#[test]
fn dense_linear_matmul_matches_raw_matmul() {
    let dev = Default::default();
    let w = Tensor::<B, 2>::from_data(
        TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]),
        &dev,
    );
    let x = Tensor::<B, 3>::from_data(
        TensorData::new(vec![1.0_f32, 1.0], [1, 1, 2]),
        &dev,
    );
    let lw = LinearWeight::dense(w.clone());
    let out_via_lw = lw.matmul(x.clone());
    let out_direct = x.matmul(w.unsqueeze());
    let a: Vec<f32> = out_via_lw.into_data().to_vec().unwrap();
    let b: Vec<f32> = out_direct.into_data().to_vec().unwrap();
    assert_eq!(a, b);
}

#[test]
fn quantized_linear_matmul_approximates_dense() {
    let dev = Default::default();
    // Random-ish weight; rounding under Q8 introduces small error.
    let raw: Vec<f32> = (0..6).map(|i| (i as f32 - 3.0) * 0.1).collect();
    let w = Tensor::<B, 2>::from_data(TensorData::new(raw.clone(), [2, 3]), &dev);
    let x = Tensor::<B, 3>::from_data(
        TensorData::new(vec![1.0_f32, -1.0], [1, 1, 2]),
        &dev,
    );

    let dense = LinearWeight::dense(w.clone());
    let qw = QuantizedTensor::<B>::quantize_from(w);
    let quant = LinearWeight::Quantized(qw);

    let out_dense: Vec<f32> = dense.matmul(x.clone()).into_data().to_vec().unwrap();
    let out_quant: Vec<f32> = quant.matmul(x).into_data().to_vec().unwrap();

    let max_diff = out_dense
        .iter()
        .zip(out_quant.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    // 2 multiply-adds per output element, each with up to Q8 noise.
    // Empirically well under 1e-2 for tensors of this size.
    assert!(
        max_diff < 1e-2,
        "quantized matmul diverged from dense by {max_diff}"
    );
}
