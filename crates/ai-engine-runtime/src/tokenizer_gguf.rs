//! Reconstruct an `HfTokenizer` from a GGUF file's tokenizer metadata.
//! Targets the Llama-3 byte-level BPE format (`tokenizer.ggml.model = "gpt2"`
//! or `"llama"`).
//!
//! SentencePiece-based GGUF tokenizers (Llama-2, Mistral) need a different
//! reconstruction path and are deferred to a follow-up plan.

use crate::gguf::metadata::{GgufArray, GgufValue};
use ai_engine_tokenizer::HfTokenizer;
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use tokenizers::{
    decoders::byte_level::ByteLevel as ByteLevelDecoder,
    models::bpe::BPE,
    pre_tokenizers::byte_level::ByteLevel as ByteLevelPre,
    Tokenizer,
};

/// Load a tokenizer from the GGUF metadata at `path`.
pub fn load_tokenizer_from_gguf(path: &Path) -> anyhow::Result<HfTokenizer> {
    let m = crate::gguf::read_metadata_only(path)?;
    load_tokenizer_from_gguf_metadata(&m)
}

/// Reconstruct an `HfTokenizer` from already-parsed GGUF metadata.
pub fn load_tokenizer_from_gguf_metadata(
    m: &HashMap<String, GgufValue>,
) -> anyhow::Result<HfTokenizer> {
    let tok_model_kind = match m.get("tokenizer.ggml.model") {
        Some(GgufValue::String(s)) => s.as_str(),
        _ => anyhow::bail!("GGUF missing tokenizer.ggml.model"),
    };
    if tok_model_kind != "gpt2" && tok_model_kind != "llama" {
        anyhow::bail!(
            "Plan 10 only supports byte-level BPE GGUF tokenizers (gpt2/llama); got `{tok_model_kind}`"
        );
    }

    let tokens = match m.get("tokenizer.ggml.tokens") {
        Some(GgufValue::Array(GgufArray::String(v))) => v.clone(),
        _ => anyhow::bail!("GGUF missing tokenizer.ggml.tokens"),
    };
    let merges = match m.get("tokenizer.ggml.merges") {
        Some(GgufValue::Array(GgufArray::String(v))) => v.clone(),
        _ => Vec::new(), // merges optional; BPE with empty merges = char-level
    };

    // Build vocab: token string -> u32 id (the tokens array is index-ordered).
    let vocab: HashMap<String, u32> = tokens
        .iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), i as u32))
        .collect();

    // Parse merges: each entry is "left right" (space-separated). Pairs that
    // don't split cleanly are skipped.
    let parsed_merges: Vec<(String, String)> = merges
        .iter()
        .filter_map(|m| {
            let mut it = m.splitn(2, ' ');
            let l = it.next()?.to_string();
            let r = it.next()?.to_string();
            Some((l, r))
        })
        .collect();

    let bpe = BPE::new(vocab, parsed_merges);
    let mut tok = Tokenizer::new(bpe);
    // Match the toy-llama-3 tokenizer.json config: pre_tokenizer has
    // add_prefix_space=false, decoder has add_prefix_space=true (the standard
    // GPT-2/Llama-3 byte-level setup).
    tok.with_pre_tokenizer(Some(ByteLevelPre::new(false, true, true)));
    tok.with_decoder(Some(ByteLevelDecoder::default()));

    // HfTokenizer's only constructor takes a path; serialize and round-trip
    // through a temp file. The tempfile crate handles cleanup automatically.
    let mut tempfile = tempfile::NamedTempFile::new()
        .map_err(|e| anyhow::anyhow!("tempfile: {e}"))?;
    let json = tok
        .to_string(false)
        .map_err(|e| anyhow::anyhow!("tokenizer.to_string: {e}"))?;
    tempfile
        .write_all(json.as_bytes())
        .map_err(|e| anyhow::anyhow!("write tempfile: {e}"))?;
    tempfile
        .flush()
        .map_err(|e| anyhow::anyhow!("flush tempfile: {e}"))?;
    HfTokenizer::from_path(tempfile.path())
}
