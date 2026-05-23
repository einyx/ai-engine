use crate::capability::Capability;
use crate::partition::{auto_partition, PartitionManifest};
use crate::protocol::codec::{decode, encode};
use crate::protocol::control::{LeaderToWorker, WorkerToLeader};
use crate::tls::NodeIdentity;
use crate::transport::frame::{read_frame, write_frame};
use crate::transport::quic::client_endpoint;
use std::net::SocketAddr;

/// A worker the leader should dial during startup.
#[derive(Debug, Clone)]
pub struct WorkerEndpoint {
    pub node_id: String,
    pub addr: SocketAddr,
    pub fingerprint: String,
}

/// Inputs to `ClusterLeader::start`.
#[derive(Debug, Clone)]
pub struct LeaderConfig {
    pub cluster_id: String,
    pub leader_node_id: String,
    pub model_id: String,
    pub n_layers: usize,
    pub layer_bytes: u64,
    pub embed_output_bytes: u64,
    pub per_node_overhead: u64,
    pub workers: Vec<WorkerEndpoint>,
}

/// Per-worker connection state owned by the leader after the join handshake.
///
/// `control_send` / `control_recv` are kept open so Task 10 can stream
/// `Assignment` and subsequent control frames on the same bidi stream.
pub struct WorkerConnection {
    pub node_id: String,
    pub conn: quinn::Connection,
    pub control_send: quinn::SendStream,
    pub control_recv: quinn::RecvStream,
}

/// Leader after startup: workers joined, capabilities collected, manifest computed.
pub struct ClusterLeader {
    manifest: PartitionManifest,
    #[allow(dead_code)] // Task 10 will use these to send Assignment frames.
    connections: Vec<WorkerConnection>,
}

impl ClusterLeader {
    /// Connect to every worker in `cfg.workers`, run the Join handshake, collect
    /// `Capability` advertisements, then compute an auto-partition manifest.
    ///
    /// Sequential per-worker dial keeps things simple — v0.2 clusters are small
    /// (≤ 8 nodes) and startup is one-shot.
    pub async fn start(
        identity: &NodeIdentity,
        cfg: LeaderConfig,
    ) -> anyhow::Result<Self> {
        let fingerprints: Vec<String> =
            cfg.workers.iter().map(|w| w.fingerprint.clone()).collect();
        let endpoint = client_endpoint(identity, &fingerprints)?;

        let mut connections: Vec<WorkerConnection> = Vec::with_capacity(cfg.workers.len());
        let mut capabilities: Vec<Capability> = Vec::with_capacity(cfg.workers.len());

        for w in &cfg.workers {
            let conn = endpoint
                .connect(w.addr, &w.node_id)
                .map_err(|e| anyhow::anyhow!("connect {}: {e}", w.node_id))?
                .await
                .map_err(|e| anyhow::anyhow!("handshake {}: {e}", w.node_id))?;

            let (mut send, mut recv) = conn.open_bi().await?;

            // 1. Send Join.
            let join = LeaderToWorker::Join {
                cluster_id: cfg.cluster_id.clone(),
                protocol_version: 1,
                leader_node_id: cfg.leader_node_id.clone(),
            };
            write_frame(&mut send, &encode(&join)?).await?;

            // 2. Read JoinAck.
            let ack_bytes = read_frame(&mut recv).await?;
            let ack: WorkerToLeader = decode(&ack_bytes)?;
            match ack {
                WorkerToLeader::JoinAck { node_id, .. } => {
                    if node_id != w.node_id {
                        anyhow::bail!(
                            "worker {} reported node_id {node_id} in JoinAck",
                            w.node_id
                        );
                    }
                }
                other => anyhow::bail!("expected JoinAck from {}, got {other:?}", w.node_id),
            }

            // 3. Read Capability.
            let cap_bytes = read_frame(&mut recv).await?;
            let cap_msg: WorkerToLeader = decode(&cap_bytes)?;
            let cap = match cap_msg {
                WorkerToLeader::Capability(c) => c,
                other => anyhow::bail!(
                    "expected Capability from {}, got {other:?}",
                    w.node_id
                ),
            };
            capabilities.push(cap);

            connections.push(WorkerConnection {
                node_id: w.node_id.clone(),
                conn,
                control_send: send,
                control_recv: recv,
            });
        }

        let manifest = auto_partition(
            &cfg.model_id,
            &capabilities,
            cfg.n_layers,
            cfg.layer_bytes,
            cfg.embed_output_bytes,
            cfg.per_node_overhead,
        )?;

        Ok(Self {
            manifest,
            connections,
        })
    }

    pub fn manifest(&self) -> &PartitionManifest {
        &self.manifest
    }
}
