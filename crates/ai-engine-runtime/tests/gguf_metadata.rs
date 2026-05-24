use ai_engine_runtime::gguf::metadata::{parse_kv, parse_string, GgufValue};

#[test]
fn parses_gguf_string() {
    // length=5, bytes="hello"
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&5u64.to_le_bytes());
    bytes.extend_from_slice(b"hello");
    let (s, consumed) = parse_string(&bytes).unwrap();
    assert_eq!(s, "hello");
    assert_eq!(consumed, 8 + 5);
}

#[test]
fn parses_u32_kv() {
    // key="bits", type=4 (U32), value=8
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&4u64.to_le_bytes());   // key length
    bytes.extend_from_slice(b"bits");
    bytes.extend_from_slice(&4u32.to_le_bytes());   // type U32
    bytes.extend_from_slice(&8u32.to_le_bytes());   // value
    let (k, v, _consumed) = parse_kv(&bytes).unwrap();
    assert_eq!(k, "bits");
    match v {
        GgufValue::U32(n) => assert_eq!(n, 8),
        _ => panic!("wrong type"),
    }
}

#[test]
fn parses_string_kv() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&12u64.to_le_bytes());  // key length
    bytes.extend_from_slice(b"general.name");
    bytes.extend_from_slice(&8u32.to_le_bytes());   // type STRING
    bytes.extend_from_slice(&7u64.to_le_bytes());   // value length
    bytes.extend_from_slice(b"toy-llm");
    let (k, v, _consumed) = parse_kv(&bytes).unwrap();
    assert_eq!(k, "general.name");
    match v {
        GgufValue::String(s) => assert_eq!(s, "toy-llm"),
        _ => panic!("wrong type"),
    }
}

#[test]
fn skips_unknown_value_type_gracefully() {
    // type=255 (unknown) should produce GgufValue::Unknown with a length-conservative skip.
    // For Plan 7 we accept Err on unknown so the loader fails fast with a clear message.
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&3u64.to_le_bytes());
    bytes.extend_from_slice(b"foo");
    bytes.extend_from_slice(&255u32.to_le_bytes());
    bytes.extend_from_slice(&[0u8; 8]);   // some payload
    let err = parse_kv(&bytes).unwrap_err().to_string();
    assert!(err.to_lowercase().contains("unknown"));
}
