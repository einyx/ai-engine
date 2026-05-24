use ai_engine_runtime::gguf::header::parse_header;

#[test]
fn parses_minimal_header() {
    // Hand-built bytes: magic "GGUF" + version=3 + tensor_count=0 + metadata_count=0.
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    let (hdr, consumed) = parse_header(&bytes).unwrap();
    assert_eq!(hdr.version, 3);
    assert_eq!(hdr.tensor_count, 0);
    assert_eq!(hdr.metadata_count, 0);
    assert_eq!(consumed, 24);    // 4 + 4 + 8 + 8
}

#[test]
fn rejects_wrong_magic() {
    let bytes = b"FAKE\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
    assert!(parse_header(bytes).is_err());
}

#[test]
fn rejects_unsupported_version() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    let err = parse_header(&bytes).unwrap_err().to_string();
    assert!(err.contains("version"), "got: {err}");
}

#[test]
fn rejects_truncated_header() {
    let bytes = b"GGUF\x03\x00\x00\x00";   // Only magic + version, missing counts.
    assert!(parse_header(bytes).is_err());
}
