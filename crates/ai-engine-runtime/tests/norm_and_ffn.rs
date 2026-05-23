use ai_engine_runtime::arch::ffn::SwiGluFfn;
use ai_engine_runtime::arch::rmsnorm::RmsNorm;
use burn::tensor::{Tensor, TensorData};

type B = burn_ndarray::NdArray;

#[test]
fn rmsnorm_unit_weights_normalizes_to_unit_rms() {
    let dev = Default::default();
    let norm = RmsNorm::<B>::with_weights(
        /*hidden=*/ 4,
        [1.0_f32, 1.0, 1.0, 1.0],
        /*eps=*/ 1e-6,
        &dev,
    );
    let x = Tensor::<B, 2>::from_data(
        TensorData::new(vec![2.0_f32, 2.0, 2.0, 2.0], [1, 4]),
        &dev,
    );
    let out = norm.forward(x);
    // RMS of [2,2,2,2] is 2. Output should be [1,1,1,1].
    let v: Vec<f32> = out.to_data().to_vec().unwrap();
    for x in &v {
        assert!((x - 1.0).abs() < 1e-5, "{x} != 1");
    }
}

#[test]
fn swiglu_ffn_runs_with_expected_output_shape() {
    let dev = Default::default();
    let ffn = SwiGluFfn::<B>::with_random_weights(/*hidden=*/ 8, /*inter=*/ 16, &dev);
    let x = Tensor::<B, 3>::ones([1, 2, 8], &dev);
    let out = ffn.forward(x);
    assert_eq!(out.dims(), [1, 2, 8]);
}
