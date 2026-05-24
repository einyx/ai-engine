//! GGUF format reader. Supports v3 with ggml_type F32, F16, BF16, Q4_0.
//!
//! Reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

pub mod header;
pub mod metadata;
pub mod tensor_desc;
pub mod q4_0;

pub use header::{parse_header, GgufHeader};
