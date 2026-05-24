//! Single-replica candle model wrapper: GGUF load + autoregressive generation.

use anyhow::Context;
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_llama::ModelWeights;
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

pub struct CandleModel {
    weights: ModelWeights,
    tokenizer: Arc<HfTokenizer>,
    device: Device,
    eos_token_id: u32,
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
        let eos_token_id = content
            .metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.to_u32().ok())
            .context("gguf missing tokenizer.ggml.eos_token_id")?;
        let weights = ModelWeights::from_gguf(content, &mut file, &device)
            .map_err(|e| anyhow::anyhow!("ModelWeights::from_gguf: {e}"))?;
        Ok(Self { weights, tokenizer, device, eos_token_id })
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
}
