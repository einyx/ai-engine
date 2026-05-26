//! Read-only, presentation-facing view of cluster state for the web UI.
//! Defined here (lowest crate) so `ai-engine-http` can hold a trait object
//! without depending on `ai-engine-cluster`.

use serde::Serialize;

/// One node's place in the inference pipeline.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct NodeTopology {
    pub node_id: String,
    pub backend: String,
    pub device_index: usize,
    pub available_memory_bytes: u64,
    pub compute_score: u32,
    pub link_mbps_to_leader: u32,
    pub layer_start: usize,
    pub layer_end: usize,
    pub hosts_embedding: bool,
    pub hosts_output: bool,
    pub previous_node: Option<String>,
    pub next_node: Option<String>,
}

/// Full topology snapshot. `model_id` is `None` and `nodes` empty in
/// gateway-only mode.
#[derive(Debug, Clone, Serialize, PartialEq, Default)]
pub struct TopologySnapshot {
    pub model_id: Option<String>,
    pub nodes: Vec<NodeTopology>,
}

/// Implemented by the cluster crate over the live leader. Object-safe.
pub trait ClusterView: Send + Sync {
    /// Current node assignments + capabilities.
    fn topology(&self) -> TopologySnapshot;
    /// Monotonic count of output tokens produced since startup. The metrics
    /// endpoint derives tokens/sec from deltas of this value.
    fn total_tokens(&self) -> u64;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topology_serializes_to_expected_json() {
        let snap = TopologySnapshot {
            model_id: Some("m".into()),
            nodes: vec![NodeTopology {
                node_id: "a".into(),
                backend: "Cuda".into(),
                device_index: 0,
                available_memory_bytes: 1,
                compute_score: 2,
                link_mbps_to_leader: 3,
                layer_start: 0,
                layer_end: 4,
                hosts_embedding: true,
                hosts_output: false,
                previous_node: None,
                next_node: Some("b".into()),
            }],
        };
        let v = serde_json::to_value(&snap).unwrap();
        assert_eq!(v["model_id"], "m");
        assert_eq!(v["nodes"][0]["layer_end"], 4);
        assert_eq!(v["nodes"][0]["next_node"], "b");
    }

    #[test]
    fn default_snapshot_is_empty() {
        let v = serde_json::to_value(TopologySnapshot::default()).unwrap();
        assert_eq!(v["model_id"], serde_json::Value::Null);
        assert!(v["nodes"].as_array().unwrap().is_empty());
    }
}
