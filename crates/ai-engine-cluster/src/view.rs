//! Adapts the live `ClusterLeader` into the presentation-facing
//! `ai_engine_core::cluster_view::ClusterView` trait.

use std::sync::Arc;

use ai_engine_core::cluster_view::{ClusterView, NodeTopology, TopologySnapshot};

use crate::capability::{BackendKind, Capability};
use crate::metrics::ClusterMetrics;
use crate::partition::PartitionManifest;

/// Owns a snapshot of capabilities + manifest taken at startup and a live
/// metrics handle. The manifest/capabilities are fixed for v0.2 (no
/// re-partitioning at runtime), so a snapshot is sufficient and avoids holding
/// the whole `ClusterLeader`.
pub struct ClusterViewImpl {
    capabilities: Vec<Capability>,
    manifest: PartitionManifest,
    metrics: Arc<ClusterMetrics>,
}

impl ClusterViewImpl {
    pub fn new(
        capabilities: Vec<Capability>,
        manifest: PartitionManifest,
        metrics: Arc<ClusterMetrics>,
    ) -> Self {
        Self { capabilities, manifest, metrics }
    }
}

fn backend_str(b: BackendKind) -> String {
    match b {
        BackendKind::Cpu => "Cpu",
        BackendKind::Cuda => "Cuda",
        BackendKind::Metal => "Metal",
        BackendKind::Wgpu => "Wgpu",
    }
    .to_string()
}

impl ClusterView for ClusterViewImpl {
    fn topology(&self) -> TopologySnapshot {
        let nodes = self
            .manifest
            .assignments
            .iter()
            .map(|a| {
                let cap = self.capabilities.iter().find(|c| c.node_id == a.node_id);
                NodeTopology {
                    node_id: a.node_id.clone(),
                    backend: cap.map(|c| backend_str(c.backend)).unwrap_or_default(),
                    device_index: cap.map(|c| c.device_index).unwrap_or(0),
                    available_memory_bytes: cap.map(|c| c.available_memory_bytes).unwrap_or(0),
                    compute_score: cap.map(|c| c.compute_score).unwrap_or(0),
                    link_mbps_to_leader: cap.map(|c| c.link_mbps_to_leader).unwrap_or(0),
                    layer_start: a.layer_range.start,
                    layer_end: a.layer_range.end,
                    hosts_embedding: a.hosts_embedding,
                    hosts_output: a.hosts_output,
                    previous_node: a.previous_node.clone(),
                    next_node: a.next_node.clone(),
                }
            })
            .collect();
        TopologySnapshot {
            model_id: Some(self.manifest.model_id.clone()),
            nodes,
        }
    }

    fn total_tokens(&self) -> u64 {
        self.metrics.total_tokens()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::partition::NodeAssignment;

    fn cap(id: &str) -> Capability {
        Capability {
            node_id: id.into(),
            backend: BackendKind::Cuda,
            device_index: 0,
            available_memory_bytes: 100,
            compute_score: 10,
            link_mbps_to_leader: 1000,
        }
    }

    #[test]
    fn topology_joins_caps_and_assignments() {
        let manifest = PartitionManifest {
            model_id: "m".into(),
            model_config_hash: [0u8; 32],
            assignments: vec![NodeAssignment {
                node_id: "a".into(),
                layer_range: 0..8,
                hosts_embedding: true,
                hosts_output: true,
                previous_node: None,
                next_node: None,
            }],
        };
        let view = ClusterViewImpl::new(
            vec![cap("a")],
            manifest,
            Arc::new(ClusterMetrics::new()),
        );
        let snap = view.topology();
        assert_eq!(snap.model_id.as_deref(), Some("m"));
        assert_eq!(snap.nodes.len(), 1);
        assert_eq!(snap.nodes[0].backend, "Cuda");
        assert_eq!(snap.nodes[0].layer_end, 8);
        assert!(snap.nodes[0].hosts_embedding);
    }
}
