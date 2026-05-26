use ai_engine_cluster::discovery::txt::{TxtRecords, SERVICE_TYPE};
use std::collections::HashMap;

#[test]
fn service_type_constant() {
    assert_eq!(SERVICE_TYPE, "_ai-engine._tcp.local.");
}

#[test]
fn txt_records_roundtrip() {
    let r = TxtRecords {
        cluster_id: "home-lab".into(),
        node_id: "worker-1".into(),
        role: "worker".into(),
        protocol_version: 1,
        fingerprint: "sha256:abc123".into(),
        backend: "cpu".into(),
    };
    let map = r.to_map();
    assert_eq!(map.get("cluster_id"), Some(&"home-lab".to_string()));
    assert_eq!(map.get("node_id"), Some(&"worker-1".to_string()));
    assert_eq!(map.get("role"), Some(&"worker".to_string()));
    assert_eq!(map.get("protocol_version"), Some(&"1".to_string()));
    assert_eq!(map.get("fingerprint"), Some(&"sha256:abc123".to_string()));
    assert_eq!(map.get("backend"), Some(&"cpu".to_string()));

    let back = TxtRecords::from_map(&map).unwrap();
    assert_eq!(back.cluster_id, r.cluster_id);
    assert_eq!(back.node_id, r.node_id);
    assert_eq!(back.role, r.role);
    assert_eq!(back.protocol_version, r.protocol_version);
    assert_eq!(back.fingerprint, r.fingerprint);
    assert_eq!(back.backend, r.backend);
}

#[test]
fn missing_required_field_errors() {
    let mut map: HashMap<String, String> = HashMap::new();
    map.insert("cluster_id".into(), "x".into());
    map.insert("node_id".into(), "y".into());
    // role / protocol_version / fingerprint / backend missing
    let err = TxtRecords::from_map(&map).unwrap_err().to_string();
    assert!(err.to_lowercase().contains("missing"), "got: {err}");
}

#[test]
fn malformed_protocol_version_errors() {
    let mut map: HashMap<String, String> = HashMap::new();
    map.insert("cluster_id".into(), "x".into());
    map.insert("node_id".into(), "y".into());
    map.insert("role".into(), "worker".into());
    map.insert("protocol_version".into(), "not-a-number".into());
    map.insert("fingerprint".into(), "sha256:abc".into());
    map.insert("backend".into(), "cpu".into());
    let err = TxtRecords::from_map(&map).unwrap_err().to_string();
    assert!(err.to_lowercase().contains("protocol_version"), "got: {err}");
}
