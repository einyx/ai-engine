use ai_engine_cluster::capability::BackendKind;
use ai_engine_cluster::leader::{ClusterLeader, LeaderConfig, WorkerEndpoint};
use ai_engine_cluster::protocol::codec::{decode, encode};
use ai_engine_cluster::protocol::control::LeaderToWorker;
use ai_engine_cluster::tls::generate_node_identity;
use ai_engine_cluster::transport::frame::{read_frame, write_frame};
use ai_engine_cluster::transport::quic::server_endpoint;
use ai_engine_cluster::worker::run_worker_handshake;
use tokio::sync::oneshot;

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
        partition_override: None,
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

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn workers_receive_correct_assignment_from_leader() {
    let w1_id = generate_node_identity("w1").unwrap();
    let w1_ep = server_endpoint(&w1_id, "127.0.0.1:0".parse().unwrap()).unwrap();
    let w1_addr = w1_ep.local_addr().unwrap();
    let w2_id = generate_node_identity("w2").unwrap();
    let w2_ep = server_endpoint(&w2_id, "127.0.0.1:0".parse().unwrap()).unwrap();
    let w2_addr = w2_ep.local_addr().unwrap();

    // For this test we replicate just the handshake-and-Assignment-receive portion
    // of run_worker_full, so we can capture the assigned range without loading
    // any model weights.
    let (w1_tx, w1_rx) = oneshot::channel();
    let (w2_tx, w2_rx) = oneshot::channel();
    tokio::spawn(handshake_and_capture_assignment(
        w1_ep,
        "w1".into(),
        w1_tx,
    ));
    tokio::spawn(handshake_and_capture_assignment(
        w2_ep,
        "w2".into(),
        w2_tx,
    ));

    let leader_id = generate_node_identity("leader").unwrap();
    let cfg = LeaderConfig {
        cluster_id: "test".into(),
        leader_node_id: "leader".into(),
        model_id: "toy".into(),
        n_layers: 4,
        layer_bytes: 1024 * 1024,
        embed_output_bytes: 1024 * 1024,
        per_node_overhead: 256 * 1024,
        workers: vec![
            WorkerEndpoint {
                node_id: "w1".into(),
                addr: w1_addr,
                fingerprint: w1_id.fingerprint.clone(),
            },
            WorkerEndpoint {
                node_id: "w2".into(),
                addr: w2_addr,
                fingerprint: w2_id.fingerprint.clone(),
            },
        ],
        partition_override: None,
    };
    let leader = ClusterLeader::start(&leader_id, cfg).await.unwrap();
    let manifest = leader.manifest();

    // Each worker should have received an Assignment whose embedded manifest
    // contains its expected range. With auto-partition on 4 layers / 2 nodes,
    // each worker gets 2 layers contiguously.
    let assn_w1 = tokio::time::timeout(std::time::Duration::from_secs(5), w1_rx)
        .await
        .unwrap()
        .unwrap();
    let assn_w2 = tokio::time::timeout(std::time::Duration::from_secs(5), w2_rx)
        .await
        .unwrap()
        .unwrap();

    let w1_range = manifest.for_node("w1").unwrap().layer_range.clone();
    let w2_range = manifest.for_node("w2").unwrap().layer_range.clone();
    assert_eq!(assn_w1, w1_range);
    assert_eq!(assn_w2, w2_range);
}

async fn handshake_and_capture_assignment(
    endpoint: quinn::Endpoint,
    node_id: String,
    out: oneshot::Sender<std::ops::Range<usize>>,
) -> anyhow::Result<()> {
    let incoming = endpoint
        .accept()
        .await
        .ok_or_else(|| anyhow::anyhow!("no incoming"))?;
    let conn = incoming.await?;
    let (mut send, mut recv) = conn.accept_bi().await?;

    // Read Join, write JoinAck + Capability.
    let _join: LeaderToWorker = decode(&read_frame(&mut recv).await?)?;
    let ack = ai_engine_cluster::protocol::control::WorkerToLeader::JoinAck {
        node_id: node_id.clone(),
        certificate_sha256: [0u8; 32],
    };
    write_frame(&mut send, &encode(&ack)?).await?;
    let cap = ai_engine_cluster::capability::detect_capability(
        &node_id,
        BackendKind::Cpu,
        0,
        None,
    )?;
    write_frame(
        &mut send,
        &encode(&ai_engine_cluster::protocol::control::WorkerToLeader::Capability(cap))?,
    )
    .await?;

    // Read Assignment, extract our range.
    let assn: LeaderToWorker = decode(&read_frame(&mut recv).await?)?;
    let range = if let LeaderToWorker::Assignment { manifest, .. } = assn {
        manifest
            .for_node(&node_id)
            .ok_or_else(|| anyhow::anyhow!("no assignment for {node_id}"))?
            .layer_range
            .clone()
    } else {
        anyhow::bail!("expected Assignment, got {:?}", assn)
    };
    let _ = out.send(range);
    Ok(())
}
