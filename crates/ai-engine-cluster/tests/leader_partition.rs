use ai_engine_cluster::capability::BackendKind;
use ai_engine_cluster::leader::{ClusterLeader, LeaderConfig, WorkerEndpoint};
use ai_engine_cluster::tls::generate_node_identity;
use ai_engine_cluster::transport::quic::server_endpoint;
use ai_engine_cluster::worker::run_worker_handshake;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn leader_connects_to_two_workers_and_computes_manifest() {
    // Spin up two in-process workers.
    let worker_a_id = generate_node_identity("worker-a").unwrap();
    let worker_a_ep =
        server_endpoint(&worker_a_id, "127.0.0.1:0".parse().unwrap()).unwrap();
    let worker_a_addr = worker_a_ep.local_addr().unwrap();
    let worker_a_fp = worker_a_id.fingerprint.clone();

    let worker_b_id = generate_node_identity("worker-b").unwrap();
    let worker_b_ep =
        server_endpoint(&worker_b_id, "127.0.0.1:0".parse().unwrap()).unwrap();
    let worker_b_addr = worker_b_ep.local_addr().unwrap();
    let worker_b_fp = worker_b_id.fingerprint.clone();

    let task_a = tokio::spawn(async move {
        run_worker_handshake(worker_a_ep, "worker-a".to_string(), BackendKind::Cpu).await
    });
    let task_b = tokio::spawn(async move {
        run_worker_handshake(worker_b_ep, "worker-b".to_string(), BackendKind::Cpu).await
    });

    // Build leader.
    let leader_id = generate_node_identity("leader").unwrap();
    let cfg = LeaderConfig {
        cluster_id: "test-cluster".into(),
        leader_node_id: "leader".into(),
        model_id: "test-model".into(),
        n_layers: 4,
        layer_bytes: 1024,
        embed_output_bytes: 1024,
        per_node_overhead: 1024,
        workers: vec![
            WorkerEndpoint {
                node_id: "worker-a".into(),
                addr: worker_a_addr,
                fingerprint: worker_a_fp,
            },
            WorkerEndpoint {
                node_id: "worker-b".into(),
                addr: worker_b_addr,
                fingerprint: worker_b_fp,
            },
        ],
    };

    let leader = ClusterLeader::start(&leader_id, cfg).await.unwrap();
    let manifest = leader.manifest();

    assert_eq!(manifest.model_id, "test-model");
    assert_eq!(manifest.assignments.len(), 2);

    // Contiguous coverage of 0..4.
    let mut expected_start = 0;
    let mut total = 0;
    for a in &manifest.assignments {
        assert_eq!(a.layer_range.start, expected_start);
        assert!(a.layer_range.end > a.layer_range.start);
        expected_start = a.layer_range.end;
        total += a.layer_range.end - a.layer_range.start;
    }
    assert_eq!(expected_start, 4);
    assert_eq!(total, 4);

    // First node hosts embedding, last hosts output.
    assert!(manifest.assignments[0].hosts_embedding);
    assert!(manifest.assignments[manifest.assignments.len() - 1].hosts_output);

    drop(leader);
    let _ = task_a.await;
    let _ = task_b.await;
}
