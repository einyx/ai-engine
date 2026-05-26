use ai_engine_runtime::arch::rope::RotaryEmbedding;
use burn::tensor::Tensor;

type B = burn_ndarray::NdArray;

#[test]
fn rope_precomputes_correct_table_shape() {
    let dev = Default::default();
    let rope = RotaryEmbedding::<B>::new(
        /*head_dim=*/ 64, /*max_seq=*/ 128, /*theta=*/ 10000.0, &dev,
    );
    assert_eq!(rope.cos_table_shape(), [128, 32]);
    assert_eq!(rope.sin_table_shape(), [128, 32]);
}

#[test]
fn rope_at_position_zero_is_identity() {
    let dev = Default::default();
    let rope = RotaryEmbedding::<B>::new(64, 128, 10000.0, &dev);
    let x = Tensor::<B, 4>::random(
        [1, 4, 1, 64],
        burn::tensor::Distribution::Default,
        &dev,
    );
    let positions = vec![0_i32];
    let out = rope.apply(x.clone(), &positions);
    let diff: f32 = (out - x).abs().max().into_scalar();
    assert!(diff < 1e-5, "RoPE@0 should be identity; max diff = {diff}");
}

#[test]
fn rope_at_different_positions_differs() {
    let dev = Default::default();
    let rope = RotaryEmbedding::<B>::new(64, 128, 10000.0, &dev);
    let x = Tensor::<B, 4>::ones([1, 4, 1, 64], &dev);
    let out_a = rope.apply(x.clone(), &[5]);
    let out_b = rope.apply(x, &[37]);
    let diff: f32 = (out_a - out_b).abs().max().into_scalar();
    assert!(
        diff > 1e-3,
        "RoPE at different positions should differ; max diff = {diff}"
    );
}
