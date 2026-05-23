use ai_engine_cluster::capability::{detect_capability, BackendKind};

#[test]
fn capability_detection_populates_realistic_values() {
    let cap = detect_capability("test-node", BackendKind::Cpu, 0, None).unwrap();
    assert_eq!(cap.node_id, "test-node");
    assert!(cap.available_memory_bytes > 0, "memory > 0");
    assert!(cap.compute_score > 0, "compute_score > 0 (microbenchmark must run)");
    assert_eq!(cap.backend, BackendKind::Cpu);
}

#[test]
fn capability_respects_max_memory_override() {
    // If max_memory_mib is set, available_memory_bytes is min(detected, override*MiB).
    let cap = detect_capability("test-node", BackendKind::Cpu, 0, Some(100)).unwrap();
    assert!(cap.available_memory_bytes <= 100 * 1024 * 1024);
}

#[test]
fn cpu_compute_score_baseline_around_100() {
    // The CPU microbenchmark is normalized so a baseline CPU returns ~100.
    // Any sane test environment should land in [10, 10_000].
    let cap = detect_capability("benchmark-cpu", BackendKind::Cpu, 0, None).unwrap();
    assert!(
        cap.compute_score >= 10 && cap.compute_score <= 10_000,
        "compute_score = {} (expected 10-10000)",
        cap.compute_score
    );
}
