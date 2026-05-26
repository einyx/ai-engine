use ai_engine_cluster::capability::{BackendKind, Capability};
use ai_engine_cluster::partition::{agreed_manifest, auto_partition, manifests_agree, manual_partition};

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
fn agreed_manifest_is_order_independent() {
    // Two nodes discovering peers in different orders must still converge on
    // the same plan — that's what makes coordinator-free agreement possible.
    let order_a = vec![cap("a", 16, 100), cap("b", 16, 120), cap("c", 16, 80)];
    let order_b = vec![cap("c", 16, 80), cap("a", 16, 100), cap("b", 16, 120)];
    let ma = agreed_manifest("m", &order_a, 30, 1 << 30, 1 << 30, 256 << 20).unwrap();
    let mb = agreed_manifest("m", &order_b, 30, 1 << 30, 1 << 30, 256 << 20).unwrap();
    assert!(manifests_agree(&ma, &mb), "shuffled caps must agree by hash");
    assert_eq!(ma.model_config_hash, mb.model_config_hash);
    // And the canonical pipeline order is by node_id, regardless of input order.
    let ids: Vec<&str> = ma.assignments.iter().map(|x| x.node_id.as_str()).collect();
    assert_eq!(ids, vec!["a", "b", "c"]);
}

#[test]
fn agreed_manifest_disagrees_on_different_capability_sets() {
    let one = vec![cap("a", 16, 100), cap("b", 16, 100)];
    let two = vec![cap("a", 16, 100), cap("b", 16, 300)]; // b is much faster
    let ma = agreed_manifest("m", &one, 12, 1 << 30, 1 << 30, 256 << 20).unwrap();
    let mb = agreed_manifest("m", &two, 12, 1 << 30, 1 << 30, 256 << 20).unwrap();
    // A faster node shifts the DP cut (more layers to b) → different plan → no agreement.
    assert!(!manifests_agree(&ma, &mb));
}

#[test]
fn pipeline_order_and_hosts_follow_the_ring() {
    // 3 nodes, agreed (canonical) order a→b→c. Any node can derive this same
    // routing from the manifest alone — no leader needed.
    let caps = vec![cap("b", 16, 100), cap("a", 16, 100), cap("c", 16, 100)];
    let m = agreed_manifest("m", &caps, 30, 1 << 30, 1 << 30, 256 << 20).unwrap();
    assert_eq!(m.embedding_host(), Some("a"));
    assert_eq!(m.output_host(), Some("c"));
    assert_eq!(m.pipeline_order(), Some(vec!["a", "b", "c"]));
}

#[test]
fn forward_plan_flags_local_hops_per_coordinator() {
    let caps = vec![cap("a", 16, 100), cap("b", 16, 100), cap("c", 16, 100)];
    let m = agreed_manifest("m", &caps, 30, 1 << 30, 1 << 30, 256 << 20).unwrap();

    // Coordinator on "b": same ring order a→b→c, but only b's hop is local.
    let plan = m.forward_plan("b").unwrap();
    let ids: Vec<&str> = plan.iter().map(|s| s.node_id.as_str()).collect();
    assert_eq!(ids, vec!["a", "b", "c"]);
    assert_eq!(plan.iter().map(|s| s.local).collect::<Vec<_>>(), vec![false, true, false]);

    // A node not in the cluster coordinates an all-remote pass.
    let outsider = m.forward_plan("zzz").unwrap();
    assert!(outsider.iter().all(|s| !s.local));
}

#[test]
fn pipeline_order_single_node() {
    let caps = vec![cap("solo", 32, 100)];
    let m = agreed_manifest("m", &caps, 8, 1 << 30, 1 << 30, 256 << 20).unwrap();
    assert_eq!(m.embedding_host(), Some("solo"));
    assert_eq!(m.output_host(), Some("solo"));
    assert_eq!(m.pipeline_order(), Some(vec!["solo"]));
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
