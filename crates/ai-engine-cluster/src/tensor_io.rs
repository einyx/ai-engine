use burn::tensor::{backend::Backend, Tensor, TensorData};

/// Convert a 3-D f32 tensor to (raw bytes, [batch, seq, hidden]) for wire transport.
///
/// The bytes are a flat little-endian f32 array — the same order as
/// `TensorData::to_vec::<f32>()` (row-major).
pub fn tensor_to_bytes<B: Backend>(t: Tensor<B, 3>) -> anyhow::Result<(Vec<u8>, [usize; 3])> {
    let shape = t.dims();
    let data: Vec<f32> = t
        .into_data()
        .to_vec()
        .map_err(|e| anyhow::anyhow!("to_vec f32: {e:?}"))?;
    let bytes = bytemuck::cast_slice::<f32, u8>(&data).to_vec();
    Ok((bytes, shape))
}

/// Reconstruct a 3-D f32 tensor from raw bytes produced by `tensor_to_bytes`.
pub fn tensor_from_bytes<B: Backend>(
    bytes: &[u8],
    shape: [usize; 3],
    device: &B::Device,
) -> anyhow::Result<Tensor<B, 3>> {
    if bytes.len() % 4 != 0 {
        anyhow::bail!("byte length {} not f32-aligned", bytes.len());
    }
    let expected = shape[0] * shape[1] * shape[2];
    let actual = bytes.len() / 4;
    if expected != actual {
        anyhow::bail!("shape {:?} expects {} f32, got {}", shape, expected, actual);
    }
    let data: Vec<f32> = bytemuck::cast_slice::<u8, f32>(bytes).to_vec();
    Ok(Tensor::<B, 3>::from_data(TensorData::new(data, shape), device))
}
