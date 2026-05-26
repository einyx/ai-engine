//! TXT-record schema for mDNS service announcements.
//!
//! Each worker advertises one mDNS service instance with TXT records
//! containing these fields. `TxtRecords` provides typed encode/decode
//! over the raw `HashMap<String, String>` that `mdns-sd` exposes.

use std::collections::HashMap;

/// mDNS service type for ai-engine cluster nodes.
pub const SERVICE_TYPE: &str = "_ai-engine._tcp.local.";

/// `role` value used by Ollama endpoint advertisements (vs. `"worker"` /
/// `"leader"` for cluster nodes). Ollama ads ride the same `SERVICE_TYPE` but
/// carry a different TXT field set; consumers filter on `role` first.
pub const ROLE_OLLAMA: &str = "ollama";

/// Typed representation of the TXT records carried in an mDNS service
/// announcement.
#[derive(Debug, Clone)]
pub struct TxtRecords {
    pub cluster_id: String,
    pub node_id: String,
    /// `"worker"` or `"leader"`.
    pub role: String,
    pub protocol_version: u16,
    /// `"sha256:<64 hex chars>"`
    pub fingerprint: String,
    /// `"cpu"` | `"cuda"` | `"metal"` | `"wgpu"`
    pub backend: String,
}

impl TxtRecords {
    /// Encode into the flat string map that `mdns-sd` expects.
    pub fn to_map(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("cluster_id".into(), self.cluster_id.clone());
        m.insert("node_id".into(), self.node_id.clone());
        m.insert("role".into(), self.role.clone());
        m.insert("protocol_version".into(), self.protocol_version.to_string());
        m.insert("fingerprint".into(), self.fingerprint.clone());
        m.insert("backend".into(), self.backend.clone());
        m
    }

    /// Decode from a flat string map. Returns an error if any required field
    /// is missing or if `protocol_version` is not a valid `u16`.
    pub fn from_map(m: &HashMap<String, String>) -> anyhow::Result<Self> {
        let get = |k: &str| -> anyhow::Result<String> {
            m.get(k)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("TXT records missing required field `{k}`"))
        };
        let cluster_id = get("cluster_id")?;
        let node_id = get("node_id")?;
        let role = get("role")?;
        let protocol_version: u16 = get("protocol_version")?
            .parse()
            .map_err(|e| anyhow::anyhow!("malformed protocol_version: {e}"))?;
        let fingerprint = get("fingerprint")?;
        let backend = get("backend")?;
        Ok(Self {
            cluster_id,
            node_id,
            role,
            protocol_version,
            fingerprint,
            backend,
        })
    }
}
