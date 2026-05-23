use ai_engine_tokenizer::{HfTokenizer, Tokenizer};

#[test]
fn encode_then_decode_roundtrips_text() {
    let tok = HfTokenizer::from_path("tests/fixtures/tokenizer_tiny.json")
        .expect("load tokenizer");
    let text = "Hello, world!";
    let ids = tok.encode(text).unwrap();
    assert!(!ids.is_empty(), "encode produced tokens");
    let back = tok.decode(&ids).unwrap();
    assert_eq!(back.trim(), text);
}

#[test]
fn encode_handles_unicode() {
    let tok = HfTokenizer::from_path("tests/fixtures/tokenizer_tiny.json").unwrap();
    let _ids = tok.encode("café 日本").unwrap();
}
