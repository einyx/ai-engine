//! Worker-mode entrypoint.
//!
//! When `ai-engine` is started on a node whose `--node-id` (or hostname) matches
//! a non-leader entry in `[[cluster.node]]`, the binary skips the HTTP gateway
//! and instead spins up a QUIC listener that hosts a portion of the model's
//! layers, awaiting activations from the cluster leader.

use ai_engine_cluster::{
    capability::BackendKind,
    discovery::{Announcer, TxtRecords},
    tls::load_or_generate_node_identity,
    transport::quic::server_endpoint,
    worker::run_worker_full,
};
use ai_engine_runtime::config::ModelConfig;

/// Run the worker forever. Returns only when the QUIC endpoint stops accepting
/// connections (e.g., the process is shutting down).
pub async fn run_worker(
    cfg: &ai_engine_config::Config,
    node_id: &str,
    cluster_id: &str,
) -> anyhow::Result<()> {
    let cluster = cfg
        .clusters
        .iter()
        .find(|c| c.id == cluster_id)
        .ok_or_else(|| anyhow::anyhow!("cluster `{cluster_id}` not found in config"))?;
    let me = cluster
        .nodes
        .iter()
        .find(|n| n.id == node_id)
        .ok_or_else(|| anyhow::anyhow!("node `{node_id}` not in cluster `{cluster_id}`"))?;

    let identity_dir = dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".ai-engine");
    let identity = load_or_generate_node_identity(node_id, &identity_dir)?;
    eprintln!(
        "ai-engine worker `{}` fingerprint: {}",
        node_id, identity.fingerprint
    );

    let bind: std::net::SocketAddr = me.addr.parse()?;
    let endpoint = server_endpoint(&identity, bind)?;

    // mDNS announcement — runs alongside the QUIC server. Bound to a local
    // `_announcer` to keep the daemon alive for the lifetime of this worker.
    let txt = TxtRecords {
        cluster_id: cluster_id.into(),
        node_id: node_id.into(),
        role: "worker".into(),
        protocol_version: 1,
        fingerprint: identity.fingerprint.clone(),
        backend: me.backend.clone(),
    };
    let ann_ip: std::net::IpAddr = bind.ip();
    let ann_host = format!("{}.local.", node_id);
    let _announcer = Announcer::register(ann_ip, bind.port(), &ann_host, txt)?;
    tracing::info!(node_id = %node_id, "ai-engine worker announcing on mDNS");

    let model_path: std::path::PathBuf = (&cluster.model.weights_path).into();
    let model_cfg = match &cluster.model.config_path {
        Some(p) => ModelConfig::from_file(std::path::Path::new(p))?,
        None => ModelConfig::from_gguf_file(&model_path)?,
    };

    let backend = match me.backend.as_str() {
        "cpu" => BackendKind::Cpu,
        "cuda" => BackendKind::Cuda,
        "metal" => BackendKind::Metal,
        "wgpu" => BackendKind::Wgpu,
        other => anyhow::bail!("unknown backend `{other}` (should have been validated)"),
    };

    // v0.2.0: NdArray backend only. The worker now learns its layer range
    // from the leader's Assignment frame over QUIC, not from local config.
    run_worker_full::<burn_ndarray::NdArray>(
        endpoint,
        node_id.to_string(),
        backend,
        model_path,
        model_cfg,
    )
    .await
}
