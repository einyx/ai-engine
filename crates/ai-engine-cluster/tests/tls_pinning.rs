use ai_engine_cluster::tls::{generate_node_identity, fingerprint_sha256};

#[test]
fn cert_generation_produces_valid_pem_pair() {
    let id = generate_node_identity("test-node").unwrap();
    assert!(id.cert_pem.starts_with("-----BEGIN CERTIFICATE-----"));
    assert!(id.key_pem.starts_with("-----BEGIN PRIVATE KEY-----")
            || id.key_pem.starts_with("-----BEGIN PRIVATE KEY-----")
            || id.key_pem.contains("PRIVATE KEY-----"));
}

#[test]
fn fingerprint_is_64_hex_chars_prefixed() {
    let id = generate_node_identity("test-node").unwrap();
    assert!(id.fingerprint.starts_with("sha256:"));
    let hex = &id.fingerprint["sha256:".len()..];
    assert_eq!(hex.len(), 64);
    assert!(hex.chars().all(|c: char| c.is_ascii_hexdigit()));
}

#[test]
fn fingerprint_is_deterministic_for_a_given_cert() {
    let id = generate_node_identity("node-a").unwrap();
    let fp2 = fingerprint_sha256(&id.cert_der);
    assert_eq!(id.fingerprint, fp2);
}

#[test]
fn two_invocations_produce_distinct_fingerprints() {
    let a = generate_node_identity("node-a").unwrap();
    let b = generate_node_identity("node-b").unwrap();
    assert_ne!(a.fingerprint, b.fingerprint);
}
