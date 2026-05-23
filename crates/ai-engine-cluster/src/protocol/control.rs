use crate::capability::Capability;
use crate::partition::PartitionManifest;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LeaderToWorker {
    Join {
        cluster_id: String,
        protocol_version: u16,
        leader_node_id: String,
    },
    Assignment {
        manifest: PartitionManifest,
        model_id: String,
    },
    Begin {
        request_id: Uuid,
        max_tokens: u32,
        prompt_len: u32,
    },
    End {
        request_id: Uuid,
        reason: EndReason,
    },
    HealthPing { nonce: u64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkerToLeader {
    Capability(Capability),
    JoinAck { node_id: String, certificate_sha256: [u8; 32] },
    BeginAck { request_id: Uuid },
    Heartbeat { nonce: u64 },
    FaultReport {
        request_id: Option<Uuid>,
        kind: FaultKind,
        detail: String,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum EndReason {
    Completed,
    ClientCancelled,
    Error,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum FaultKind {
    OutOfMemory,
    BackendError,
    Internal,
}
