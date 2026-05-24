use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily { Llama3, Mistral, Qwen25, DeepSeekV2 }

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub tie_word_embeddings: bool,
    pub family: ModelFamily,
}

#[derive(Deserialize)]
struct HfConfigJson {
    architectures: Vec<String>,
    hidden_size: usize,
    #[serde(default)]
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    #[serde(default)]
    num_key_value_heads: Option<usize>,
    #[serde(default)]
    head_dim: Option<usize>,
    vocab_size: usize,
    #[serde(default = "default_max_pos")]
    max_position_embeddings: usize,
    #[serde(default = "default_rope_theta")]
    rope_theta: f32,
    #[serde(default = "default_rms_eps")]
    rms_norm_eps: f32,
    #[serde(default)]
    tie_word_embeddings: bool,
}

fn default_max_pos() -> usize { 8192 }
fn default_rope_theta() -> f32 { 10000.0 }
fn default_rms_eps() -> f32 { 1e-6 }

impl ModelConfig {
    /// Parse a HuggingFace `config.json` string. Convenience wrapper over the
    /// `FromStr` impl so callers don't need to import the trait.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> anyhow::Result<Self> {
        s.parse()
    }

    /// Parse a HuggingFace `config.json` file into a `ModelConfig`.
    pub fn from_file(path: &std::path::Path) -> anyhow::Result<Self> {
        std::fs::read_to_string(path)?.parse()
    }

    /// Extract a `ModelConfig` from a GGUF file's metadata. Targets Llama-3-style
    /// `llama.*` keys. Returns an error if the file isn't a Llama-family GGUF
    /// or required metadata is missing.
    pub fn from_gguf_file(path: &std::path::Path) -> anyhow::Result<Self> {
        let m = crate::gguf::read_metadata_only(path)?;
        Self::from_gguf_metadata(&m)
    }

    /// Extract a `ModelConfig` from a pre-parsed GGUF metadata map.
    pub fn from_gguf_metadata(
        m: &std::collections::HashMap<String, crate::gguf::GgufValue>,
    ) -> anyhow::Result<Self> {
        use crate::gguf::metadata::GgufArray;
        use crate::gguf::GgufValue;

        let arch = match m.get("general.architecture") {
            Some(GgufValue::String(s)) => s.as_str(),
            Some(other) => anyhow::bail!("general.architecture wrong type: {other:?}"),
            None => anyhow::bail!("general.architecture missing in GGUF metadata"),
        };
        let family = match arch {
            "llama" => ModelFamily::Llama3,
            other => anyhow::bail!(
                "GGUF architecture `{other}` not supported in Plan 10 (only `llama`)"
            ),
        };

        let n_layers = gguf_read_u32(m, "llama.block_count")? as usize;
        let hidden_size = gguf_read_u32(m, "llama.embedding_length")? as usize;
        let n_heads = gguf_read_u32(m, "llama.attention.head_count")? as usize;
        let n_kv_heads = gguf_read_u32(m, "llama.attention.head_count_kv")? as usize;
        let intermediate_size = gguf_read_u32(m, "llama.feed_forward_length")? as usize;
        let max_position_embeddings = gguf_read_u32(m, "llama.context_length")? as usize;
        let rms_norm_eps = gguf_read_f32(m, "llama.attention.layer_norm_rms_epsilon")?;
        let rope_theta = gguf_read_f32(m, "llama.rope.freq_base")?;
        let head_dim = hidden_size / n_heads;

        // vocab_size from tokenizer.ggml.tokens array length; fall back to
        // llama.vocab_size u32 key.
        let vocab_size = match m.get("tokenizer.ggml.tokens") {
            Some(GgufValue::Array(GgufArray::String(v))) => v.len(),
            _ => match m.get("llama.vocab_size") {
                Some(GgufValue::U32(n)) => *n as usize,
                _ => anyhow::bail!(
                    "GGUF missing both tokenizer.ggml.tokens array and llama.vocab_size"
                ),
            },
        };

        // tie_word_embeddings: GGUF has no explicit key. Default to true (the
        // most common case — Llama-3 family ties its embeddings). Callers who
        // need precise tie-ness for non-default checkpoints can override via TOML.
        let tie_word_embeddings = true;

        Ok(Self {
            hidden_size,
            intermediate_size,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            vocab_size,
            max_position_embeddings,
            rope_theta,
            rms_norm_eps,
            tie_word_embeddings,
            family,
        })
    }
}

fn gguf_read_u32(
    m: &std::collections::HashMap<String, crate::gguf::GgufValue>,
    key: &str,
) -> anyhow::Result<u32> {
    use crate::gguf::GgufValue;
    match m.get(key) {
        Some(GgufValue::U32(n)) => Ok(*n),
        Some(GgufValue::U64(n)) => Ok(*n as u32),
        Some(GgufValue::I32(n)) => Ok(*n as u32),
        Some(other) => anyhow::bail!("GGUF key `{key}` wrong type for u32: {other:?}"),
        None => anyhow::bail!("GGUF key `{key}` missing"),
    }
}

fn gguf_read_f32(
    m: &std::collections::HashMap<String, crate::gguf::GgufValue>,
    key: &str,
) -> anyhow::Result<f32> {
    use crate::gguf::GgufValue;
    match m.get(key) {
        Some(GgufValue::F32(f)) => Ok(*f),
        Some(GgufValue::F64(f)) => Ok(*f as f32),
        Some(other) => anyhow::bail!("GGUF key `{key}` wrong type for f32: {other:?}"),
        None => anyhow::bail!("GGUF key `{key}` missing"),
    }
}

impl std::str::FromStr for ModelConfig {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let raw: HfConfigJson = serde_json::from_str(s)
            .map_err(|e| anyhow::anyhow!("config.json parse: {e}"))?;
        let family = detect_family(&raw.architectures)?;
        let n_kv_heads = raw.num_key_value_heads.unwrap_or(raw.num_attention_heads);
        let head_dim = raw.head_dim.unwrap_or(raw.hidden_size / raw.num_attention_heads);
        Ok(Self {
            hidden_size: raw.hidden_size,
            intermediate_size: raw.intermediate_size,
            n_layers: raw.num_hidden_layers,
            n_heads: raw.num_attention_heads,
            n_kv_heads,
            head_dim,
            vocab_size: raw.vocab_size,
            max_position_embeddings: raw.max_position_embeddings,
            rope_theta: raw.rope_theta,
            rms_norm_eps: raw.rms_norm_eps,
            tie_word_embeddings: raw.tie_word_embeddings,
            family,
        })
    }
}

fn detect_family(architectures: &[String]) -> anyhow::Result<ModelFamily> {
    for arch in architectures {
        let lc = arch.to_lowercase();
        // Check mixtral BEFORE mistral: "mixtral" does not contain "mistral" as a
        // substring (m-i-x vs m-i-s), but defensive ordering is clearer intent.
        if lc.contains("mixtral") {
            anyhow::bail!("Mixtral / MoE not supported in v0.2 (architecture: {arch})");
        }
        if lc.contains("llama") { return Ok(ModelFamily::Llama3); }
        if lc.contains("mistral") { return Ok(ModelFamily::Mistral); }
        if lc.contains("qwen") { return Ok(ModelFamily::Qwen25); }
        if lc.contains("deepseek") { return Ok(ModelFamily::DeepSeekV2); }
    }
    anyhow::bail!("unknown model architecture: {:?}", architectures)
}
