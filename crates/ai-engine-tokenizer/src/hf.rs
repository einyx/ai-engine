use crate::Tokenizer;
use std::path::Path;

pub struct HfTokenizer {
    inner: tokenizers::Tokenizer,
}

impl HfTokenizer {
    pub fn from_path<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path.as_ref())
            .map_err(|e| anyhow::anyhow!("load tokenizer.json: {e}"))?;
        Ok(Self { inner })
    }
}

impl Tokenizer for HfTokenizer {
    fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    fn decode(&self, ids: &[u32]) -> anyhow::Result<String> {
        self.inner
            .decode(ids, false)
            .map_err(|e| anyhow::anyhow!("decode: {e}"))
    }

    fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}
