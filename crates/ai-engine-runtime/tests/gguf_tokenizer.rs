use ai_engine_runtime::load_tokenizer_from_gguf;
use ai_engine_tokenizer::Tokenizer;
use std::path::PathBuf;

fn gguf_fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/toy-llama-3-gguf")
}

#[test]
fn load_tokenizer_from_gguf_can_encode_and_decode() {
    let tok = load_tokenizer_from_gguf(&gguf_fixture().join("model.gguf")).unwrap();
    let text = "The quick brown fox";
    let ids = tok.encode(text).unwrap();
    assert!(!ids.is_empty(), "encode produced tokens");
    let back = tok.decode(&ids).unwrap();
    // BPE tokenizer with ByteLevel pre+post should roundtrip cleanly.
    assert_eq!(back.trim(), text);
}

#[test]
fn gguf_tokenizer_produces_same_ids_as_json_tokenizer() {
    use ai_engine_tokenizer::HfTokenizer;
    let from_gguf = load_tokenizer_from_gguf(&gguf_fixture().join("model.gguf")).unwrap();
    let from_json = HfTokenizer::from_path(gguf_fixture().join("tokenizer.json")).unwrap();

    for prompt in &["hello", "The quick brown fox", "ai-engine"] {
        let g = from_gguf.encode(prompt).unwrap();
        let j = from_json.encode(prompt).unwrap();
        assert_eq!(
            g, j,
            "tokenization mismatch on `{prompt}`: gguf={g:?} json={j:?}"
        );
    }
}
