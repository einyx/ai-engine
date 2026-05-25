//! Single-replica candle model wrapper: GGUF load + autoregressive generation.

use anyhow::Context;
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use std::path::Path;
use std::sync::Arc;

use ai_engine_runtime::sample::{self, SamplingConfig};
use ai_engine_tokenizer::{HfTokenizer, Tokenizer};

#[derive(Debug, Clone)]
pub struct GenParams {
    pub max_tokens: usize,
    pub temperature: f32,
}

impl Default for GenParams {
    fn default() -> Self {
        Self { max_tokens: 256, temperature: 0.0 }
    }
}

pub fn should_stop(produced: usize, last_token: u32, eos: u32, params: &GenParams) -> bool {
    produced >= params.max_tokens || last_token == eos
}

/// Architectures whose tokenizer + quantized model candle supports here.
/// Guarded explicitly: arches outside this set (e.g. gemma/SentencePiece,
/// phi/custom-BPE) would silently produce garbage with our byte-level-BPE
/// tokenizer reconstruction, so we reject them with a clear error instead.
pub(crate) fn supported_arch(arch: &str) -> anyhow::Result<&'static str> {
    match arch {
        "llama" => Ok("llama"),
        "qwen2" => Ok("qwen2"),
        "qwen3" => Ok("qwen3"),
        other => anyhow::bail!(
            "candle-local: unsupported architecture '{other}' (supported: llama, qwen2, qwen3). \
             Other architectures (gemma, phi, mistral, ...) need tokenizer work and are not yet validated."
        ),
    }
}

fn detect_supported_arch(content: &gguf_file::Content) -> anyhow::Result<&'static str> {
    let arch = content
        .metadata
        .get("general.architecture")
        .and_then(|v| v.to_string().ok())
        .map(|s| s.as_str().to_owned())
        .context("gguf missing general.architecture")?;
    supported_arch(&arch)
}

enum CandleWeights {
    Llama(candle_transformers::models::quantized_llama::ModelWeights),
    Qwen2(candle_transformers::models::quantized_qwen2::ModelWeights),
    Qwen3(candle_transformers::models::quantized_qwen3::ModelWeights),
}

impl CandleWeights {
    fn forward(&mut self, x: &Tensor, pos: usize) -> candle_core::Result<Tensor> {
        match self {
            CandleWeights::Llama(m) => m.forward(x, pos),
            CandleWeights::Qwen2(m) => m.forward(x, pos),
            CandleWeights::Qwen3(m) => m.forward(x, pos),
        }
    }
}

pub struct CandleModel {
    weights: CandleWeights,
    tokenizer: Arc<HfTokenizer>,
    device: Device,
    eos_token_id: u32,
    /// Embedded Jinja chat template from GGUF metadata, if present.
    pub chat_template: Option<String>,
    /// BOS token string decoded from the tokenizer (empty if unavailable).
    pub bos_token: String,
    /// EOS token string decoded from the tokenizer (empty if unavailable).
    pub eos_token: String,
}

impl CandleModel {
    pub fn load(
        gguf_path: &Path,
        device: Device,
        tokenizer: Arc<HfTokenizer>,
    ) -> anyhow::Result<Self> {
        let mut file = std::fs::File::open(gguf_path)
            .with_context(|| format!("open {}", gguf_path.display()))?;
        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| anyhow::anyhow!("read gguf {}: {e}", gguf_path.display()))?;

        // Detect + validate architecture (ref) before consuming content.
        let arch = detect_supported_arch(&content)?;

        // Read eos token id (ref) before consuming content.
        let eos_token_id = content
            .metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.to_u32().ok())
            .context("gguf missing tokenizer.ggml.eos_token_id")?;

        // Read bos token id (ref) before consuming content.
        let bos_token_id = content
            .metadata
            .get("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.to_u32().ok());

        // Read embedded chat template (ref) before consuming content.
        let chat_template = content
            .metadata
            .get("tokenizer.chat_template")
            .and_then(|v| v.to_string().ok())
            .cloned();

        // Now consume content into the matching from_gguf variant.
        let weights = match arch {
            "llama" => {
                let m = candle_transformers::models::quantized_llama::ModelWeights::from_gguf(
                    content, &mut file, &device,
                )
                .map_err(|e| anyhow::anyhow!("ModelWeights::from_gguf (llama): {e}"))?;
                CandleWeights::Llama(m)
            }
            "qwen2" => {
                let m = candle_transformers::models::quantized_qwen2::ModelWeights::from_gguf(
                    content, &mut file, &device,
                )
                .map_err(|e| anyhow::anyhow!("ModelWeights::from_gguf (qwen2): {e}"))?;
                CandleWeights::Qwen2(m)
            }
            "qwen3" => {
                let m = candle_transformers::models::quantized_qwen3::ModelWeights::from_gguf(
                    content, &mut file, &device,
                )
                .map_err(|e| anyhow::anyhow!("ModelWeights::from_gguf (qwen3): {e}"))?;
                CandleWeights::Qwen3(m)
            }
            // detect_supported_arch already rejected anything else; this is unreachable.
            _ => unreachable!("detect_supported_arch should have rejected arch '{arch}'"),
        };

        // Decode bos/eos token ids to their string representations.
        let bos_token = bos_token_id
            .and_then(|id| tokenizer.decode(&[id]).ok())
            .unwrap_or_default();
        let eos_token = tokenizer.decode(&[eos_token_id]).unwrap_or_default();

        Ok(Self { weights, tokenizer, device, eos_token_id, chat_template, bos_token, eos_token })
    }

    /// Render messages using the model's embedded chat template if present,
    /// else `None` (caller falls back to a plain format).
    pub fn render_with_chat_template(
        &self,
        messages: &[crate::template::TemplateMessage],
    ) -> Option<anyhow::Result<String>> {
        self.chat_template.as_ref().map(|t| {
            crate::template::render_chat_template(t, messages, &self.bos_token, &self.eos_token)
        })
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        params: &GenParams,
        mut on_token: impl FnMut(u32),
        prompt_tokens_out: &mut usize,
    ) -> anyhow::Result<Vec<u32>> {
        let prompt_ids = self.tokenizer.encode(prompt)?;
        *prompt_tokens_out = prompt_ids.len();
        anyhow::ensure!(!prompt_ids.is_empty(), "empty prompt after tokenization");

        let sample_cfg = SamplingConfig {
            temperature: params.temperature,
            top_p: None,
            top_k: None,
            seed: 42,
        };

        let input = Tensor::new(prompt_ids.as_slice(), &self.device)?
            .reshape((1, prompt_ids.len()))?;
        let logits = self.weights.forward(&input, 0)
            .map_err(|e| anyhow::anyhow!("forward(prefill): {e}"))?;
        let logits_v: Vec<f32> = logits.squeeze(0)?.to_vec1()?;
        let mut next = sample::sample(&logits_v, &sample_cfg);

        let mut produced = 0usize;
        let mut out = Vec::new();
        let mut index_pos = prompt_ids.len();

        loop {
            if should_stop(produced, next, self.eos_token_id, params) {
                break;
            }
            on_token(next);
            out.push(next);
            produced += 1;

            let input = Tensor::new(&[next], &self.device)?.reshape((1, 1))?;
            let logits = self.weights.forward(&input, index_pos)
                .map_err(|e| anyhow::anyhow!("forward(decode): {e}"))?;
            let logits_v: Vec<f32> = logits.squeeze(0)?.to_vec1()?;
            next = sample::sample(&logits_v, &sample_cfg);
            index_pos += 1;
        }
        Ok(out)
    }

    pub fn decode(&self, ids: &[u32]) -> anyhow::Result<String> {
        self.tokenizer.decode(ids)
    }
}

/// Read GGUF metadata (eos/bos token ids, chat template) WITHOUT loading model
/// weights. Returns `(eos_token_id, chat_template, bos_token_str, eos_token_str)`.
pub fn read_gguf_meta(
    gguf_path: &Path,
    tokenizer: &HfTokenizer,
) -> anyhow::Result<(u32, Option<String>, String, String)> {
    let mut file = std::fs::File::open(gguf_path)
        .with_context(|| format!("open {}", gguf_path.display()))?;
    let content = gguf_file::Content::read(&mut file)
        .map_err(|e| anyhow::anyhow!("read gguf {}: {e}", gguf_path.display()))?;

    let eos_token_id = content
        .metadata
        .get("tokenizer.ggml.eos_token_id")
        .and_then(|v| v.to_u32().ok())
        .context("gguf missing tokenizer.ggml.eos_token_id")?;

    let bos_token_id = content
        .metadata
        .get("tokenizer.ggml.bos_token_id")
        .and_then(|v| v.to_u32().ok());

    let chat_template = content
        .metadata
        .get("tokenizer.chat_template")
        .and_then(|v| v.to_string().ok())
        .cloned();

    let bos_token = bos_token_id
        .and_then(|id| tokenizer.decode(&[id]).ok())
        .unwrap_or_default();
    let eos_token = tokenizer.decode(&[eos_token_id]).unwrap_or_default();

    Ok((eos_token_id, chat_template, bos_token, eos_token))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gen_params_defaults() {
        let p = GenParams::default();
        assert_eq!(p.max_tokens, 256);
        assert_eq!(p.temperature, 0.0);
    }

    #[test]
    fn should_stop_on_eos() {
        let p = GenParams { max_tokens: 100, temperature: 0.0 };
        assert!(should_stop(5, 42, 42, &p));      // last==eos -> stop
        assert!(!should_stop(5, 7, 42, &p));      // under budget, not eos -> continue
        assert!(should_stop(100, 7, 42, &p));     // hit max_tokens -> stop
    }

    #[test]
    fn supported_arch_allowlist() {
        assert_eq!(supported_arch("llama").unwrap(), "llama");
        assert_eq!(supported_arch("qwen2").unwrap(), "qwen2");
        assert_eq!(supported_arch("qwen3").unwrap(), "qwen3");
        assert!(supported_arch("gemma").is_err());
        assert!(supported_arch("phi3").is_err());
        assert!(supported_arch("mistral").is_err());
        assert!(supported_arch("").is_err());
    }
}
