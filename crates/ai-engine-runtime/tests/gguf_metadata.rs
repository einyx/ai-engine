use ai_engine_runtime::gguf::metadata::{parse_kv, parse_string, GgufArray, GgufValue};

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
fn parse_array_handles_i32_elements() {
    // key="k", type=9 (ARRAY), elem_type=5 (I32), count=3, values: -1, 0, 7
    let mut bytes = Vec::new();
    // key: length=1, "k"
    bytes.extend_from_slice(&1u64.to_le_bytes());
    bytes.extend_from_slice(b"k");
    // value type: 9 = ARRAY
    bytes.extend_from_slice(&9u32.to_le_bytes());
    // array header: elem_type=5 (i32), count=3
    bytes.extend_from_slice(&5u32.to_le_bytes());
    bytes.extend_from_slice(&3u64.to_le_bytes());
    // three i32 values: -1, 0, 7
    bytes.extend_from_slice(&(-1i32).to_le_bytes());
    bytes.extend_from_slice(&0i32.to_le_bytes());
    bytes.extend_from_slice(&7i32.to_le_bytes());

    let (k, v, consumed) = parse_kv(&bytes).unwrap();
    assert_eq!(k, "k");
    // key: 8+1=9, type: 4, array header: 12, elements: 3*4=12 → total 37
    assert_eq!(consumed, 9 + 4 + 12 + 12);
    match v {
        GgufValue::Array(GgufArray::I32(vals)) => assert_eq!(vals, vec![-1, 0, 7]),
        other => panic!("expected Array(I32(...)), got {:?}", other),
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
