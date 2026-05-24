//! GGUF tensor descriptor: name, shape, ggml_type, offset.

use super::metadata::parse_string;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    BF16 = 30,
}

#[derive(Debug, Clone)]
pub struct TensorDesc {
    pub name: String,
    pub shape: Vec<u64>,
    pub ggml_type: GgmlType,
    pub offset: u64,
}

pub fn parse_tensor_desc(bytes: &[u8]) -> anyhow::Result<(TensorDesc, usize)> {
    let (name, name_consumed) = parse_string(bytes)?;
    let mut cursor = name_consumed;

    if bytes.len() < cursor + 4 {
        anyhow::bail!("tensor desc truncated at n_dims");
    }
    let n_dims = u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
    cursor += 4;

    if bytes.len() < cursor + n_dims * 8 {
        anyhow::bail!("tensor desc truncated at shape");
    }
    let mut shape = Vec::with_capacity(n_dims);
    for _ in 0..n_dims {
        shape.push(u64::from_le_bytes(
            bytes[cursor..cursor + 8].try_into().unwrap(),
        ));
        cursor += 8;
    }

    if bytes.len() < cursor + 4 {
        anyhow::bail!("tensor desc truncated at ggml_type");
    }
    let type_id = u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
    let ggml_type = match type_id {
        0 => GgmlType::F32,
        1 => GgmlType::F16,
        2 => GgmlType::Q4_0,
        30 => GgmlType::BF16,
        other => anyhow::bail!(
            "unsupported ggml_type {other} (only F32=0, F16=1, Q4_0=2, BF16=30 in Plan 7)"
        ),
    };
    cursor += 4;

    if bytes.len() < cursor + 8 {
        anyhow::bail!("tensor desc truncated at offset");
    }
    let offset = u64::from_le_bytes(bytes[cursor..cursor + 8].try_into().unwrap());
    cursor += 8;

    Ok((
        TensorDesc {
            name,
            shape,
            ggml_type,
            offset,
        },
        cursor,
    ))
}
