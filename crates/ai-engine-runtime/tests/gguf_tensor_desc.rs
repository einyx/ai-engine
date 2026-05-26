use ai_engine_runtime::gguf::tensor_desc::{parse_tensor_desc, GgmlType};

#[test]
fn parses_2d_q4_0_descriptor() {
    let mut bytes = Vec::new();
    // name = "blk.0.attn_q.weight" (19 chars)
    let name = "blk.0.attn_q.weight";
    bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
    bytes.extend_from_slice(name.as_bytes());
    // n_dims = 2
    bytes.extend_from_slice(&2u32.to_le_bytes());
    // shape[0] = 32, shape[1] = 4 (one block-group across in dim, 4 out columns)
    bytes.extend_from_slice(&32u64.to_le_bytes());
    bytes.extend_from_slice(&4u64.to_le_bytes());
    // ggml_type = 2 (Q4_0)
    bytes.extend_from_slice(&2u32.to_le_bytes());
    // offset = 100
    bytes.extend_from_slice(&100u64.to_le_bytes());

    let (desc, consumed) = parse_tensor_desc(&bytes).unwrap();
    assert_eq!(desc.name, "blk.0.attn_q.weight");
    assert_eq!(desc.shape, vec![32, 4]);
    assert_eq!(desc.ggml_type, GgmlType::Q4_0);
    assert_eq!(desc.offset, 100);
    assert_eq!(consumed, 8 + name.len() + 4 + 16 + 4 + 8);
}

#[test]
fn parses_1d_f32_descriptor() {
    let mut bytes = Vec::new();
    let name = "output_norm.weight";
    bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
    bytes.extend_from_slice(name.as_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&512u64.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes()); // F32
    bytes.extend_from_slice(&0u64.to_le_bytes());
    let (desc, _) = parse_tensor_desc(&bytes).unwrap();
    assert_eq!(desc.shape, vec![512]);
    assert_eq!(desc.ggml_type, GgmlType::F32);
}

#[test]
fn rejects_unsupported_ggml_type() {
    let mut bytes = Vec::new();
    let name = "x";
    bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
    bytes.extend_from_slice(name.as_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes());
    bytes.extend_from_slice(&20u32.to_le_bytes()); // type 20 = some Q5
    bytes.extend_from_slice(&0u64.to_le_bytes());
    let err = parse_tensor_desc(&bytes).unwrap_err().to_string();
    assert!(err.to_lowercase().contains("unsupported"));
}

#[test]
fn parse_tensor_desc_accepts_q4_1_type_3() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&1_u64.to_le_bytes()); // name len
    bytes.push(b'x');
    bytes.extend_from_slice(&1_u32.to_le_bytes()); // n_dims
    bytes.extend_from_slice(&32_u64.to_le_bytes()); // shape[0]
    bytes.extend_from_slice(&3_u32.to_le_bytes()); // ggml_type = Q4_1
    bytes.extend_from_slice(&0_u64.to_le_bytes()); // offset

    let (d, consumed) =
        ai_engine_runtime::gguf::tensor_desc::parse_tensor_desc(&bytes).unwrap();
    assert_eq!(d.name, "x");
    assert_eq!(d.shape, vec![32]);
    assert_eq!(d.ggml_type, ai_engine_runtime::gguf::tensor_desc::GgmlType::Q4_1);
    assert_eq!(consumed, bytes.len());
}

#[test]
fn parse_tensor_desc_accepts_q6_k_type_14() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&1_u64.to_le_bytes()); // name len
    bytes.push(b'e');
    bytes.extend_from_slice(&1_u32.to_le_bytes()); // n_dims
    bytes.extend_from_slice(&256_u64.to_le_bytes()); // shape[0]
    bytes.extend_from_slice(&14_u32.to_le_bytes()); // ggml_type = Q6_K
    bytes.extend_from_slice(&0_u64.to_le_bytes()); // offset

    let (d, consumed) =
        ai_engine_runtime::gguf::tensor_desc::parse_tensor_desc(&bytes).unwrap();
    assert_eq!(d.name, "e");
    assert_eq!(d.ggml_type, ai_engine_runtime::gguf::tensor_desc::GgmlType::Q6_K);
    assert_eq!(consumed, bytes.len());
}
