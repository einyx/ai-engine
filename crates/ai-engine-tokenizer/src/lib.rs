//! ai-engine-tokenizer

mod hf;

pub use hf::HfTokenizer;

pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>>;
    fn decode(&self, ids: &[u32]) -> anyhow::Result<String>;
    fn vocab_size(&self) -> usize;
}
