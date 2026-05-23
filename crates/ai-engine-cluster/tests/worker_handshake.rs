use ai_engine_cluster::capability::BackendKind;
use ai_engine_cluster::protocol::codec::{decode, encode};
use ai_engine_cluster::protocol::control::{LeaderToWorker, WorkerToLeader};
use ai_engine_cluster::tls::generate_node_identity;
use ai_engine_cluster::transport::quic::{client_endpoint, server_endpoint};
use ai_engine_cluster::transport::frame::{read_frame, write_frame};
use ai_engine_cluster::worker::run_worker_handshake;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn worker_replies_to_join_with_jointack_and_capability() {
    let worker_id = generate_node_identity("worker-a").unwrap();
    let worker_ep = server_endpoint(&worker_id, "127.0.0.1:0".parse().unwrap()).unwrap();
    let worker_addr = worker_ep.local_addr().unwrap();

    // Start the worker handshake handler.
    let worker_task = tokio::spawn(async move {
        run_worker_handshake(worker_ep, "worker-a".to_string(), BackendKind::Cpu).await
    });

    // Mimic the leader: connect, send Join, expect JoinAck + Capability.
    let leader_id = generate_node_identity("leader").unwrap();
    let leader_ep = client_endpoint(&leader_id, std::slice::from_ref(&worker_id.fingerprint)).unwrap();
    let conn = leader_ep.connect(worker_addr, "worker-a").unwrap().await.unwrap();
    let (mut send, mut recv) = conn.open_bi().await.unwrap();

    let join = LeaderToWorker::Join {
        cluster_id: "test".into(),
        protocol_version: 1,
        leader_node_id: "leader".into(),
    };
    write_frame(&mut send, &encode(&join).unwrap()).await.unwrap();

    let ack_bytes = read_frame(&mut recv).await.unwrap();
    let ack: WorkerToLeader = decode(&ack_bytes).unwrap();
    matches!(ack, WorkerToLeader::JoinAck { .. });

    let cap_bytes = read_frame(&mut recv).await.unwrap();
    let cap: WorkerToLeader = decode(&cap_bytes).unwrap();
    if let WorkerToLeader::Capability(c) = cap {
        assert_eq!(c.node_id, "worker-a");
    } else {
        panic!("expected Capability");
    }

    drop(conn);
    let _ = worker_task.await;
}
