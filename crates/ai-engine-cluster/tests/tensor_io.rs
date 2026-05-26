use ai_engine_cluster::tensor_io::{tensor_to_bytes, tensor_from_bytes};
use burn::tensor::{Tensor, TensorData};

type B = burn_ndarray::NdArray;

#[test]
fn tensor_roundtrip_through_bytes() {
    let dev = Default::default();
    let original = Tensor::<B, 3>::from_data(
        TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], [1, 2, 3]),
        &dev,
    );
    let (bytes, shape) = tensor_to_bytes(original.clone()).unwrap();
    assert_eq!(shape, [1, 2, 3]);
    assert_eq!(bytes.len(), 6 * 4);   // 6 f32

    let restored: Tensor<B, 3> = tensor_from_bytes(&bytes, shape, &dev).unwrap();
    let orig_v: Vec<f32> = original.into_data().to_vec().unwrap();
    let rest_v: Vec<f32> = restored.into_data().to_vec().unwrap();
    assert_eq!(orig_v, rest_v);
}
