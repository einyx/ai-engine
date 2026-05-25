//! Architecture config for the paged engine.

use anyhow::Context;
use candle_core::quantized::gguf_file;

/// Per-architecture forward configuration derived from GGUF metadata.
#[derive(Debug, Clone)]
pub struct ArchConfig {
    pub arch: &'static str, // "llama" | "qwen2" | "qwen3"
    pub block_count: usize,
    pub embedding_length: usize,
    pub head_count: usize,
    pub head_count_kv: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub rope_freq_base: f32,
    pub rms_norm_eps: f64,
    /// qwen2: attn_q/k/v carry a bias tensor.
    pub qkv_bias: bool,
    /// qwen3: per-head q-norm / k-norm RmsNorm before RoPE.
    pub qk_norm: bool,
    /// Maximum sequence length from GGUF metadata (fallback: 4096).
    pub context_length: usize,
}

/// Validate + map the GGUF architecture string to the supported set.
pub fn supported_arch(arch: &str) -> anyhow::Result<&'static str> {
    match arch {
        "llama" => Ok("llama"),
        "qwen2" => Ok("qwen2"),
        "qwen3" => Ok("qwen3"),
        other => anyhow::bail!(
            "paged engine: unsupported architecture '{other}' (supported: llama, qwen2, qwen3)"
        ),
    }
}

impl ArchConfig {
    pub fn from_gguf(content: &gguf_file::Content) -> anyhow::Result<Self> {
        let raw_arch = content
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .map(|s| s.as_str().to_owned())
            .context("gguf missing general.architecture")?;
        let arch = supported_arch(&raw_arch)?;

        let g = |key: &str| -> anyhow::Result<&gguf_file::Value> {
            content
                .metadata
                .get(&format!("{arch}.{key}"))
                .with_context(|| format!("gguf missing {arch}.{key}"))
        };
        let head_count = g("attention.head_count")?.to_u32()? as usize;
        let head_count_kv = g("attention.head_count_kv")?.to_u32()? as usize;
        let block_count = g("block_count")?.to_u32()? as usize;
        let embedding_length = g("embedding_length")?.to_u32()? as usize;
        // qwen2 omits rope.dimension_count; fall back to head_dim (= embedding_length / head_count).
        let head_dim_fallback = content
            .metadata
            .get(&format!("{arch}.attention.key_length"))
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize)
            .unwrap_or(embedding_length / head_count);
        let rope_dim = content
            .metadata
            .get(&format!("{arch}.rope.dimension_count"))
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize)
            .unwrap_or(head_dim_fallback);
        let rms_norm_eps = g("attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_freq_base = content
            .metadata
            .get(&format!("{arch}.rope.freq_base"))
            .and_then(|v| v.to_f32().ok())
            .unwrap_or(10000.0);
        // qwen3 stores an explicit head_dim; llama/qwen2 derive it.
        let head_dim = content
            .metadata
            .get(&format!("{arch}.attention.key_length"))
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize)
            .unwrap_or(embedding_length / head_count);

        let context_length = content
            .metadata
            .get(&format!("{arch}.context_length"))
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize)
            .unwrap_or(4096);

        Ok(Self {
            arch,
            block_count,
            embedding_length,
            head_count,
            head_count_kv,
            head_dim,
            rope_dim,
            rope_freq_base,
            rms_norm_eps,
            qkv_bias: arch == "qwen2",
            qk_norm: arch == "qwen3",
            context_length,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supported_arch_allowlist() {
        assert_eq!(supported_arch("llama").unwrap(), "llama");
        assert_eq!(supported_arch("qwen2").unwrap(), "qwen2");
        assert_eq!(supported_arch("qwen3").unwrap(), "qwen3");
        assert!(supported_arch("gemma").is_err());
    }

    #[test]
    fn flags_match_family() {
        for (arch, qkv, qkn) in [("llama", false, false), ("qwen2", true, false), ("qwen3", false, true)] {
            assert_eq!(arch == "qwen2", qkv);
            assert_eq!(arch == "qwen3", qkn);
        }
    }
}
