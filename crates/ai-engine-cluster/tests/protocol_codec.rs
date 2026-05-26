use ai_engine_cluster::protocol::control::{EndReason, FaultKind, LeaderToWorker, WorkerToLeader};
use ai_engine_cluster::protocol::data::{ActivationHeader, Dtype};
use ai_engine_cluster::protocol::codec::{decode, encode};
use uuid::Uuid;

#[test]
fn control_message_join_roundtrips() {
    let msg = LeaderToWorker::Join {
        cluster_id: "home-lab".into(),
        protocol_version: 1,
        leader_node_id: "node-a".into(),
    };
    let bytes = encode(&msg).unwrap();
    let back: LeaderToWorker = decode(&bytes).unwrap();
    match (msg, back) {
        (LeaderToWorker::Join { cluster_id: a, protocol_version: pa, leader_node_id: na },
         LeaderToWorker::Join { cluster_id: b, protocol_version: pb, leader_node_id: nb }) => {
            assert_eq!(a, b); assert_eq!(pa, pb); assert_eq!(na, nb);
        }
        _ => panic!("variant changed"),
    }
}

#[test]
fn control_message_begin_with_uuid_roundtrips() {
    let id = Uuid::now_v7();
    let msg = LeaderToWorker::Begin { request_id: id, max_tokens: 256, prompt_len: 12 };
    let bytes = encode(&msg).unwrap();
    let back: LeaderToWorker = decode(&bytes).unwrap();
    if let LeaderToWorker::Begin { request_id, max_tokens, prompt_len } = back {
        assert_eq!(request_id, id);
        assert_eq!(max_tokens, 256);
        assert_eq!(prompt_len, 12);
    } else { panic!("variant"); }
}

#[test]
fn worker_fault_report_roundtrips() {
    let msg = WorkerToLeader::FaultReport {
        request_id: Some(Uuid::now_v7()),
        kind: FaultKind::OutOfMemory,
        detail: "VRAM exhausted at layer 17".into(),
    };
    let bytes = encode(&msg).unwrap();
    let _back: WorkerToLeader = decode(&bytes).unwrap();
}

#[test]
fn end_reason_variants_roundtrip() {
    for r in [EndReason::Completed, EndReason::ClientCancelled, EndReason::Error] {
        let msg = LeaderToWorker::End { request_id: Uuid::now_v7(), reason: r };
        let bytes = encode(&msg).unwrap();
        let _back: LeaderToWorker = decode(&bytes).unwrap();
    }
}

#[test]
fn activation_header_roundtrips() {
    let h = ActivationHeader {
        request_id: Uuid::now_v7(),
        seq_pos: 7,
        shape: [1, 1, 256],
        dtype: Dtype::Bf16,
        is_terminal: false,
    };
    let bytes = encode(&h).unwrap();
    let back: ActivationHeader = decode(&bytes).unwrap();
    assert_eq!(back.request_id, h.request_id);
    assert_eq!(back.seq_pos, h.seq_pos);
    assert_eq!(back.shape, h.shape);
    assert_eq!(back.is_terminal, h.is_terminal);
}
