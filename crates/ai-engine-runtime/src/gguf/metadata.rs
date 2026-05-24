//! GGUF metadata: gguf_string and value-type dispatch.

#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(GgufArray),
}

#[derive(Debug, Clone)]
pub enum GgufArray {
    I32(Vec<i32>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    F32(Vec<f32>),
    String(Vec<String>),
    // Plan 7 only supports these array types; others reject.
}

/// Parse a length-prefixed UTF-8 string. Returns (value, bytes_consumed).
pub fn parse_string(bytes: &[u8]) -> anyhow::Result<(String, usize)> {
    if bytes.len() < 8 {
        anyhow::bail!("string length truncated");
    }
    let len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    if bytes.len() < 8 + len {
        anyhow::bail!("string truncated: declared len {len}, have {}", bytes.len() - 8);
    }
    let s = std::str::from_utf8(&bytes[8..8 + len])
        .map_err(|e| anyhow::anyhow!("invalid UTF-8 in gguf_string: {e}"))?
        .to_string();
    Ok((s, 8 + len))
}

/// Parse one (key, value, type) triple. Returns (key, value, bytes_consumed).
pub fn parse_kv(bytes: &[u8]) -> anyhow::Result<(String, GgufValue, usize)> {
    let (key, key_consumed) = parse_string(bytes)?;
    let rest = &bytes[key_consumed..];
    if rest.len() < 4 {
        anyhow::bail!("value type truncated");
    }
    let value_type = u32::from_le_bytes(rest[0..4].try_into().unwrap());
    let value_rest = &rest[4..];
    let (value, value_consumed) = parse_value(value_type, value_rest)?;
    Ok((key, value, key_consumed + 4 + value_consumed))
}

fn parse_value(type_id: u32, bytes: &[u8]) -> anyhow::Result<(GgufValue, usize)> {
    match type_id {
        0 => {
            let buf = take(bytes, 1)?;
            Ok((GgufValue::U8(buf[0]), 1))
        }
        1 => {
            let buf = take(bytes, 1)?;
            Ok((GgufValue::I8(buf[0] as i8), 1))
        }
        2 => {
            let buf = take(bytes, 2)?;
            Ok((GgufValue::U16(u16::from_le_bytes(buf.try_into().unwrap())), 2))
        }
        3 => {
            let buf = take(bytes, 2)?;
            Ok((GgufValue::I16(i16::from_le_bytes(buf.try_into().unwrap())), 2))
        }
        4 => {
            let buf = take(bytes, 4)?;
            Ok((GgufValue::U32(u32::from_le_bytes(buf.try_into().unwrap())), 4))
        }
        5 => {
            let buf = take(bytes, 4)?;
            Ok((GgufValue::I32(i32::from_le_bytes(buf.try_into().unwrap())), 4))
        }
        6 => {
            let buf = take(bytes, 4)?;
            Ok((GgufValue::F32(f32::from_le_bytes(buf.try_into().unwrap())), 4))
        }
        7 => {
            let buf = take(bytes, 1)?;
            Ok((GgufValue::Bool(buf[0] != 0), 1))
        }
        8 => {
            let (s, c) = parse_string(bytes)?;
            Ok((GgufValue::String(s), c))
        }
        9 => parse_array(bytes),
        10 => {
            let buf = take(bytes, 8)?;
            Ok((GgufValue::U64(u64::from_le_bytes(buf.try_into().unwrap())), 8))
        }
        11 => {
            let buf = take(bytes, 8)?;
            Ok((GgufValue::I64(i64::from_le_bytes(buf.try_into().unwrap())), 8))
        }
        12 => {
            let buf = take(bytes, 8)?;
            Ok((GgufValue::F64(f64::from_le_bytes(buf.try_into().unwrap())), 8))
        }
        other => anyhow::bail!("unknown GGUF value type {other}"),
    }
}

fn take(bytes: &[u8], n: usize) -> anyhow::Result<&[u8]> {
    if bytes.len() < n {
        anyhow::bail!("buffer truncated: need {n}, have {}", bytes.len());
    }
    Ok(&bytes[..n])
}

fn parse_array(bytes: &[u8]) -> anyhow::Result<(GgufValue, usize)> {
    if bytes.len() < 12 {
        anyhow::bail!("array header truncated");
    }
    let elem_type = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    let count = u64::from_le_bytes(bytes[4..12].try_into().unwrap()) as usize;
    let mut cursor = 12;
    match elem_type {
        4 => {
            let mut out = Vec::with_capacity(count);
            for _ in 0..count {
                let buf = take(&bytes[cursor..], 4)?;
                out.push(u32::from_le_bytes(buf.try_into().unwrap()));
                cursor += 4;
            }
            Ok((GgufValue::Array(GgufArray::U32(out)), cursor))
        }
        5 => {
            let mut out = Vec::with_capacity(count);
            for _ in 0..count {
                let buf = take(&bytes[cursor..], 4)?;
                out.push(i32::from_le_bytes(buf.try_into().unwrap()));
                cursor += 4;
            }
            Ok((GgufValue::Array(GgufArray::I32(out)), cursor))
        }
        10 => {
            let mut out = Vec::with_capacity(count);
            for _ in 0..count {
                let buf = take(&bytes[cursor..], 8)?;
                out.push(u64::from_le_bytes(buf.try_into().unwrap()));
                cursor += 8;
            }
            Ok((GgufValue::Array(GgufArray::U64(out)), cursor))
        }
        6 => {
            let mut out = Vec::with_capacity(count);
            for _ in 0..count {
                let buf = take(&bytes[cursor..], 4)?;
                out.push(f32::from_le_bytes(buf.try_into().unwrap()));
                cursor += 4;
            }
            Ok((GgufValue::Array(GgufArray::F32(out)), cursor))
        }
        8 => {
            let mut out = Vec::with_capacity(count);
            for _ in 0..count {
                let (s, c) = parse_string(&bytes[cursor..])?;
                out.push(s);
                cursor += c;
            }
            Ok((GgufValue::Array(GgufArray::String(out)), cursor))
        }
        other => anyhow::bail!("unknown GGUF array element type {other}"),
    }
}
