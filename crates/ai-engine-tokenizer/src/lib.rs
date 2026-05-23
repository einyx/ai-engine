//! ai-engine-tokenizer

mod hf;

pub use hf::HfTokenizer;

pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>>;
    fn decode(&self, ids: &[u32]) -> anyhow::Result<String>;
    fn vocab_size(&self) -> usize;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    Llama3,
    Mistral,
    Qwen25,
    DeepSeekV2,
}

#[derive(Debug, Clone, Copy)]
pub struct SpecialTokens {
    pub bos_token_id: u32,
    pub eos_token_id: u32,
}

impl SpecialTokens {
    pub fn for_family(family: ModelFamily) -> Self {
        match family {
            ModelFamily::Llama3 => Self { bos_token_id: 128000, eos_token_id: 128001 },
            ModelFamily::Mistral => Self { bos_token_id: 1, eos_token_id: 2 },
            ModelFamily::Qwen25 => Self { bos_token_id: 151643, eos_token_id: 151645 },
            ModelFamily::DeepSeekV2 => Self { bos_token_id: 100000, eos_token_id: 100001 },
        }
    }
}
