//! Parse the fixed-size GGUF file header: magic + version + counts.

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const SUPPORTED_VERSION: u32 = 3;

#[derive(Debug, Clone, Copy)]
pub struct GgufHeader {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_count: u64,
}

/// Returns (header, bytes_consumed). Always 24 bytes for v3.
pub fn parse_header(bytes: &[u8]) -> anyhow::Result<(GgufHeader, usize)> {
    if bytes.len() < 24 {
        anyhow::bail!("header truncated: need 24 bytes, got {}", bytes.len());
    }
    if &bytes[0..4] != GGUF_MAGIC {
        anyhow::bail!("bad magic: expected b\"GGUF\", got {:?}", &bytes[0..4]);
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if version != SUPPORTED_VERSION {
        anyhow::bail!("unsupported GGUF version {version} (expected {SUPPORTED_VERSION})");
    }
    let tensor_count = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
    let metadata_count = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
    Ok((GgufHeader { version, tensor_count, metadata_count }, 24))
}
