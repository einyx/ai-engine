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
