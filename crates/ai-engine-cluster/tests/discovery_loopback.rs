//! Loopback round-trip between [`Announcer`] and [`discover_workers`]: each
//! test registers one or two services on 127.0.0.1, then verifies the
//! discoverer collects them.
//!
//! mDNS multicast may be unavailable in some Docker / restrictive sandbox
//! environments. If these tests fail in such an environment, they should be
//! marked `#[ignore]` (the production loopback path is exercised
//! end-to-end in Task 6).

use ai_engine_cluster::discovery::{
    discover_workers, Announcer, /* DiscoveredWorker not asserted directly */ TxtRecords,
};
use std::net::IpAddr;
use std::time::Duration;

fn txt(cluster_id: &str, node_id: &str, fp_hex_digit: char) -> TxtRecords {
    TxtRecords {
        cluster_id: cluster_id.into(),
        node_id: node_id.into(),
        role: "worker".into(),
        protocol_version: 1,
        fingerprint: format!("sha256:{}", fp_hex_digit.to_string().repeat(64)),
        backend: "cpu".into(),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn discover_two_workers_on_loopback() {
    let cluster_id = "test-loop";
    let _ann1 = Announcer::register(
        IpAddr::V4("127.0.0.1".parse().unwrap()),
        50001,
        "worker1.local.",
        txt(cluster_id, "worker-1", '1'),
    )
    .unwrap();
    let _ann2 = Announcer::register(
        IpAddr::V4("127.0.0.1".parse().unwrap()),
        50002,
        "worker2.local.",
        txt(cluster_id, "worker-2", '2'),
    )
    .unwrap();

    // Let mDNS broadcasts propagate.
    tokio::time::sleep(Duration::from_millis(500)).await;

    let found = discover_workers(cluster_id, 2, Duration::from_secs(10))
        .await
        .unwrap();

    assert_eq!(found.len(), 2, "expected 2 workers, found {}", found.len());
    let ids: Vec<&str> = found.iter().map(|w| w.node_id.as_str()).collect();
    assert!(ids.contains(&"worker-1"), "missing worker-1 in {:?}", ids);
    assert!(ids.contains(&"worker-2"), "missing worker-2 in {:?}", ids);

    // Port roundtrip.
    let w1 = found.iter().find(|w| w.node_id == "worker-1").unwrap();
    assert_eq!(w1.addr.port(), 50001);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn discover_returns_partial_after_timeout() {
    let cluster_id = "test-timeout";
    let _ann = Announcer::register(
        IpAddr::V4("127.0.0.1".parse().unwrap()),
        50003,
        "only-worker.local.",
        txt(cluster_id, "only-worker", 'a'),
    )
    .unwrap();

    tokio::time::sleep(Duration::from_millis(500)).await;

    // Ask for 2; only 1 exists. Timeout returns whatever was seen.
    let found = discover_workers(cluster_id, 2, Duration::from_millis(1500))
        .await
        .unwrap();

    assert_eq!(found.len(), 1);
    assert_eq!(found[0].node_id, "only-worker");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn discover_filters_by_cluster_id() {
    let _a = Announcer::register(
        IpAddr::V4("127.0.0.1".parse().unwrap()),
        50004,
        "n1.local.",
        txt("cluster-A", "n1", '3'),
    )
    .unwrap();
    let _b = Announcer::register(
        IpAddr::V4("127.0.0.1".parse().unwrap()),
        50005,
        "n2.local.",
        txt("cluster-B", "n2", '4'),
    )
    .unwrap();

    tokio::time::sleep(Duration::from_millis(500)).await;

    let found_a = discover_workers("cluster-A", 1, Duration::from_secs(5))
        .await
        .unwrap();
    assert_eq!(found_a.len(), 1);
    assert_eq!(found_a[0].node_id, "n1");

    let found_b = discover_workers("cluster-B", 1, Duration::from_secs(5))
        .await
        .unwrap();
    assert_eq!(found_b.len(), 1);
    assert_eq!(found_b[0].node_id, "n2");
}
