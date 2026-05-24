//! GGUF format reader. Supports v3 with ggml_type F32, F16, BF16, Q4_0, Q4_1, Q6_K.
//!
//! Reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

pub mod header;
pub mod metadata;
pub mod tensor_desc;
pub mod q4_0;
pub mod q4_1;

pub use header::{parse_header, GgufHeader};
pub use metadata::GgufValue;

use std::collections::HashMap;
use std::path::Path;

/// Read a GGUF file's metadata KV pairs WITHOUT loading tensor data.
/// Cheaper than `load_gguf` when you only need the metadata.
pub fn read_metadata_only(path: &Path) -> anyhow::Result<HashMap<String, GgufValue>> {
    let file = std::fs::File::open(path)
        .map_err(|e| anyhow::anyhow!("open {}: {e}", path.display()))?;
    // mmap the file for cheap random access into the header + metadata section.
    let mmap = unsafe { memmap2::Mmap::map(&file) }
        .map_err(|e| anyhow::anyhow!("mmap {}: {e}", path.display()))?;

    let (hdr, mut cursor) = parse_header(&mmap)?;
    let mut metadata = HashMap::with_capacity(hdr.metadata_count as usize);
    for _ in 0..hdr.metadata_count {
        let (k, v, consumed) = metadata::parse_kv(&mmap[cursor..])?;
        metadata.insert(k, v);
        cursor += consumed;
    }
    Ok(metadata)
}
