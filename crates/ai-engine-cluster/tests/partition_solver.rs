use ai_engine_cluster::capability::{BackendKind, Capability};
use ai_engine_cluster::partition::{auto_partition, manual_partition};

fn cap(node_id: &str, mem_gib: u64, compute: u32) -> Capability {
    Capability {
        node_id: node_id.into(),
        backend: BackendKind::Cpu,
        device_index: 0,
        available_memory_bytes: mem_gib * 1024 * 1024 * 1024,
        compute_score: compute,
        link_mbps_to_leader: 1000,
    }
}

#[test]
fn even_capability_yields_even_split() {
    let caps = vec![cap("a", 16, 100), cap("b", 16, 100), cap("c", 16, 100)];
    // 30-layer model, 256 MiB per layer + 1 GiB embed/output budget +
    // 512 MiB per-node KV overhead. Comfortably fits in 3 × 16 GiB nodes.
    let m = auto_partition(
        "model-test",
        &caps,
        30,
        256 * 1024 * 1024,
        1024 * 1024 * 1024,
        512 * 1024 * 1024,
    )
    .unwrap();
    assert_eq!(m.assignments.len(), 3);
    let total: usize = m.assignments.iter().map(|a| a.layer_range.len()).sum();
    assert_eq!(total, 30);
}

#[test]
fn assignments_are_contiguous_and_complete() {
    let caps = vec![cap("a", 16, 100), cap("b", 16, 100)];
    let m = auto_partition(
        "m",
        &caps,
        12,
        1024 * 1024 * 1024,
        1024 * 1024 * 1024,
        256 * 1024 * 1024,
    )
    .unwrap();
    // First assignment starts at 0; subsequent start where previous ended; last ends at n_layers.
    let mut expected_start = 0;
    for a in &m.assignments {
        assert_eq!(a.layer_range.start, expected_start);
        expected_start = a.layer_range.end;
    }
    assert_eq!(expected_start, 12);
}

#[test]
fn infeasible_partition_returns_error() {
    // 50 layers at 4 GiB each = 200 GiB. Two 8 GiB nodes -> infeasible.
    let caps = vec![cap("a", 8, 100), cap("b", 8, 100)];
    let r = auto_partition(
        "big",
        &caps,
        50,
        4 * 1024 * 1024 * 1024,
        1024 * 1024 * 1024,
        1024 * 1024 * 1024,
    );
    assert!(r.is_err(), "infeasible partition must fail");
    let msg = r.unwrap_err().to_string();
    assert!(msg.to_lowercase().contains("does not fit"));
}

#[test]
fn auto_partition_is_deterministic() {
    let caps = vec![cap("a", 16, 100), cap("b", 16, 100), cap("c", 16, 100)];
    let a = auto_partition(
        "m",
        &caps,
        30,
        1024 * 1024 * 1024,
        1024 * 1024 * 1024,
        256 * 1024 * 1024,
    )
    .unwrap();
    let b = auto_partition(
        "m",
        &caps,
        30,
        1024 * 1024 * 1024,
        1024 * 1024 * 1024,
        256 * 1024 * 1024,
    )
    .unwrap();
    assert_eq!(a.model_config_hash, b.model_config_hash);
    assert_eq!(a.assignments.len(), b.assignments.len());
    for (x, y) in a.assignments.iter().zip(b.assignments.iter()) {
        assert_eq!(x.node_id, y.node_id);
        assert_eq!(x.layer_range, y.layer_range);
    }
}

#[test]
fn manual_partition_validates_complete_cover() {
    let caps = vec![cap("a", 16, 100), cap("b", 16, 100)];
    // Complete cover 0..10 + 10..30 = 30 layers, fits.
    // 512 MiB per layer keeps node "b" (20 layers + 256 MiB overhead) under 16 GiB.
    let ok = manual_partition(
        "m",
        &caps,
        30,
        vec![("a".into(), 0..10), ("b".into(), 10..30)],
        512 * 1024 * 1024,
        1024 * 1024 * 1024,
        256 * 1024 * 1024,
    )
    .unwrap();
    assert_eq!(ok.assignments.len(), 2);

    // Overlapping ranges -> error.
    let err = manual_partition(
        "m",
        &caps,
        30,
        vec![("a".into(), 0..15), ("b".into(), 10..30)],
        512 * 1024 * 1024,
        1024 * 1024 * 1024,
        256 * 1024 * 1024,
    );
    assert!(err.is_err());

    // Gap -> error.
    let err = manual_partition(
        "m",
        &caps,
        30,
        vec![("a".into(), 0..10), ("b".into(), 15..30)],
        512 * 1024 * 1024,
        1024 * 1024 * 1024,
        256 * 1024 * 1024,
    );
    assert!(err.is_err());
}

#[test]
fn for_node_returns_assignment_for_known_node() {
    let caps = vec![cap("a", 16, 100), cap("b", 16, 100)];
    let m = auto_partition(
        "m",
        &caps,
        12,
        1024 * 1024 * 1024,
        1024 * 1024 * 1024,
        256 * 1024 * 1024,
    )
    .unwrap();
    let a = m.for_node("a").unwrap();
    assert_eq!(a.node_id, "a");
    let b = m.for_node("b").unwrap();
    assert_eq!(b.node_id, "b");
    assert!(m.for_node("missing").is_none());
}
