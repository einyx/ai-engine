//! Worker-mode entrypoint.
//!
//! When `ai-engine` is started on a node whose `--node-id` (or hostname) matches
//! a non-leader entry in `[[cluster.node]]`, the binary skips the HTTP gateway
//! and instead spins up a QUIC listener that hosts a portion of the model's
//! layers, awaiting activations from the cluster leader.

use ai_engine_cluster::{
    capability::BackendKind, tls::generate_node_identity, transport::quic::server_endpoint,
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

    let identity = generate_node_identity(node_id)?;
    eprintln!(
        "ai-engine worker `{}` fingerprint: {}",
        node_id, identity.fingerprint
    );

    let bind: std::net::SocketAddr = me.addr.parse()?;
    let endpoint = server_endpoint(&identity, bind)?;
    let model_cfg = ModelConfig::from_file(std::path::Path::new(&cluster.model.config_path))?;
    let model_path: std::path::PathBuf = (&cluster.model.weights_path).into();

    // For v0.2.0, compute the layer range locally using the same even-split
    // formula the leader uses. Plan 4+ will receive the assignment over QUIC.
    let layer_range = compute_my_layer_range(cluster, &model_cfg, node_id)?;

    let backend = match me.backend.as_str() {
        "cpu" => BackendKind::Cpu,
        "cuda" => BackendKind::Cuda,
        "metal" => BackendKind::Metal,
        "wgpu" => BackendKind::Wgpu,
        other => anyhow::bail!("unknown backend `{other}` (should have been validated)"),
    };

    // v0.2.0: NdArray backend only.
    run_worker_full::<burn_ndarray::NdArray>(
        endpoint,
        node_id.to_string(),
        backend,
        model_path,
        model_cfg,
        layer_range,
    )
    .await
}

/// Even-split layer assignment across non-leader nodes in declaration order.
/// First `n_layers % n_workers` workers get one extra layer.
fn compute_my_layer_range(
    cluster: &ai_engine_config::Cluster,
    model_cfg: &ModelConfig,
    node_id: &str,
) -> anyhow::Result<std::ops::Range<usize>> {
    let workers: Vec<&ai_engine_config::ClusterNode> = cluster
        .nodes
        .iter()
        .filter(|n| n.id != cluster.leader)
        .collect();
    let n_workers = workers.len();
    if n_workers == 0 {
        anyhow::bail!("cluster `{}` has no workers", cluster.id);
    }
    let per_worker = model_cfg.n_layers / n_workers;
    let remainder = model_cfg.n_layers % n_workers;

    let my_idx = workers
        .iter()
        .position(|n| n.id == node_id)
        .ok_or_else(|| anyhow::anyhow!("node `{node_id}` is not a worker in cluster `{}`", cluster.id))?;

    let start = if my_idx < remainder {
        my_idx * (per_worker + 1)
    } else {
        remainder * (per_worker + 1) + (my_idx - remainder) * per_worker
    };
    let end = start + per_worker + if my_idx < remainder { 1 } else { 0 };
    Ok(start..end)
}
