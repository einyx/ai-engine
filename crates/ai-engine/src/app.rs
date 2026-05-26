use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use ai_engine_config::Config;
use ai_engine_http::AppState;
use ai_engine_provider::provider::{Credentials, Provider};
use ai_engine_stages::{
    ProviderRegistry, StageRegistry,
    auth::{AuthMode, AuthStage},
    content_policy::ContentPolicyStage,
    forward::ForwardStage,
    log::LogStage,
    model_route::ModelRouteStage,
};
use arc_swap::ArcSwap;

/// Role this binary instance plays in a (possibly clustered) deployment.
#[derive(Debug, Clone)]
pub enum NodeRole {
    /// This node is not in any cluster — pure gateway mode.
    Gateway,
    /// This node is the leader of one or more clusters AND/OR a gateway.
    Leader { cluster_ids: Vec<String> },
    /// This node is a worker in exactly one cluster.
    Worker {
        cluster_id: String,
        leader_addr: String,
    },
}

/// Resolve the role of this node given the config + its node id.
///
/// Worker takes precedence: if `node_id` appears as a non-leader node in any
/// cluster, this node is a worker. Otherwise, every cluster where `node_id`
/// matches the `leader` field contributes to a Leader role. If neither applies,
/// the node is a pure Gateway.
pub fn resolve_role(cfg: &Config, node_id: &str) -> NodeRole {
    let mut leader_clusters = Vec::new();
    for cluster in &cfg.clusters {
        if cluster.leader == node_id {
            leader_clusters.push(cluster.id.clone());
        } else if cluster.nodes.iter().any(|n| n.id == node_id) {
            // We're a worker in this cluster — find the leader's addr.
            let leader_addr = cluster
                .nodes
                .iter()
                .find(|n| n.id == cluster.leader)
                .map(|n| n.addr.clone())
                .unwrap_or_default();
            return NodeRole::Worker {
                cluster_id: cluster.id.clone(),
                leader_addr,
            };
        }
    }
    if leader_clusters.is_empty() {
        NodeRole::Gateway
    } else {
        NodeRole::Leader {
            cluster_ids: leader_clusters,
        }
    }
}

/// Build the `ServiceInfo` list from a config.
fn build_services(cfg: &Config) -> Vec<ai_engine_http::ServiceInfo> {
    cfg.providers
        .iter()
        .map(|p| {
            let models = cfg
                .routes
                .iter()
                .filter(|r| r.provider == p.id)
                .map(|r| r.r#match.model.clone())
                .collect();
            let endpoint = if !p.base_url.is_empty() {
                Some(p.base_url.clone())
            } else {
                p.cluster.as_ref().map(|c| format!("cluster:{c}"))
            };
            // In-process backends (candle/rustyllm) have no HTTP endpoint;
            // remote ones are "local" only when pointed at localhost.
            let in_process = matches!(p.kind.as_str(), "candle" | "rustyllm");
            let endpoint_local =
                p.base_url.contains("localhost") || p.base_url.contains("127.0.0.1");
            let weights = p.weights_path.as_deref().map(|w| {
                std::path::Path::new(w)
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or(w)
                    .to_string()
            });
            ai_engine_http::ServiceInfo {
                id: p.id.clone(),
                kind: p.kind.clone(),
                endpoint,
                models,
                device: if in_process { p.device.clone() } else { None },
                weights: if in_process { weights } else { None },
                local: in_process || endpoint_local,
            }
        })
        .collect()
}

/// Build a complete `AppState` from a validated `Config`.
///
/// This is the wiring layer: providers are instantiated, stages are constructed
/// with their config-derived parameters, the stage registry is populated, and
/// one `Pipeline` per `[pipeline."<route>"]` is built.
///
/// Routes recognized in v1: `/v1/chat/completions`, `/v1/messages`,
/// `/v1/embeddings`. Routes in `[pipeline.…]` other than these are silently
/// skipped (warned via tracing).
///
/// In cluster deployments, this also resolves the node's role: workers return
/// a stripped `AppState` (no pipelines, no providers — they only run a QUIC
/// listener via a separate worker entrypoint and respond to `/healthz`).
///
/// Async because leader-mode startup needs to await `ClusterLeader::start`
/// (QUIC join handshake against every worker).
pub async fn build_app_state(
    cfg: &Config,
    node_id: &str,
    discovered_resources: std::collections::HashMap<
        String,
        ai_engine_core::resources::NodeResources,
    >,
) -> anyhow::Result<Arc<AppState>> {
    // Leaderless p2p (Phase B): if this node belongs to a `leaderless = true`
    // cluster, take the mesh+coordinator path regardless of star "role". Every
    // such node serves its hosted stages to peers AND ingests HTTP requests via
    // a local Coordinator. This is fully additive: clusters without the flag
    // keep the legacy leader/worker star path untouched.
    let leaderless_cluster = cfg
        .clusters
        .iter()
        .find(|c| c.leaderless && c.nodes.iter().any(|n| n.id == node_id));
    if let Some(cluster_cfg) = leaderless_cluster {
        let (cluster_providers, cluster_view) =
            build_leaderless_node(cfg, cluster_cfg, node_id).await?;
        return build_gateway_app_state(
            cfg,
            cluster_providers,
            Some(cluster_view),
            discovered_resources,
        );
    }

    let role = resolve_role(cfg, node_id);
    if let NodeRole::Worker { .. } = &role {
        // Worker mode: no HTTP pipelines, just health endpoints.
        let services = build_services(cfg);
        return Ok(Arc::new(AppState {
            pipelines: HashMap::new(),
            openai_models: vec![],
            ready: AtomicBool::new(true),
            cluster: None,
            services,
            gateway_metrics: Arc::new(ai_engine_core::metrics::GatewayMetrics::default()),
            activity: Arc::new(ai_engine_core::activity::ActivityLog::default()),
            health: Arc::new(ai_engine_core::health::HealthStore::new()),
            resources: Arc::new(std::collections::HashMap::new()),
        }));
    }

    // Leader & Gateway share the same construction; the difference is which
    // [[provider]] entries we expand into ClusterProvider (leader only).
    let leader_cluster_ids: Vec<String> = match &role {
        NodeRole::Leader { cluster_ids } => cluster_ids.clone(),
        _ => Vec::new(),
    };

    // Pre-build cluster providers (async — kicks off the QUIC join handshake
    // against every worker in each cluster this node leads).
    let mut cluster_view_opt: Option<Arc<dyn ai_engine_core::cluster_view::ClusterView>> = None;
    let mut cluster_providers: HashMap<String, Arc<dyn Provider>> = HashMap::new();
    if !leader_cluster_ids.is_empty() {
        let identity = load_or_generate_node_identity(node_id)?;
        for cluster_id in &leader_cluster_ids {
            let cluster_cfg = cfg
                .clusters
                .iter()
                .find(|c| &c.id == cluster_id)
                .expect("resolve_role guarantees the cluster exists");

            let worker_endpoints: Vec<ai_engine_cluster::leader::WorkerEndpoint> =
                if let Some(disc) = &cluster_cfg.discover {
                    // mDNS discovery path: ignore the static `cert_fingerprint`
                    // fields and trust whatever fingerprints workers advertise.
                    let timeout = std::time::Duration::from_secs(disc.timeout_secs);
                    tracing::info!(
                        cluster_id = %cluster_id,
                        expected = disc.expected_workers,
                        timeout_secs = disc.timeout_secs,
                        "discovering workers via mDNS"
                    );
                    eprintln!(
                        "ai-engine leader `{}` discovering workers via mDNS (expected={}, timeout={}s)",
                        cluster_id, disc.expected_workers, disc.timeout_secs
                    );
                    let discovered = ai_engine_cluster::discovery::discover_workers(
                        cluster_id,
                        disc.expected_workers,
                        timeout,
                    )
                    .await?;
                    if discovered.is_empty() {
                        anyhow::bail!(
                            "mDNS discovery yielded zero workers for cluster `{cluster_id}` within {} sec",
                            disc.timeout_secs
                        );
                    }
                    tracing::info!(
                        cluster_id = %cluster_id,
                        found = discovered.len(),
                        "mDNS discovery complete"
                    );
                    discovered
                        .into_iter()
                        .map(ai_engine_cluster::leader::WorkerEndpoint::from_discovered)
                        .collect()
                } else {
                    cluster_cfg
                        .nodes
                        .iter()
                        .filter(|n| n.id != cluster_cfg.leader)
                        .map(|n| {
                            let addr = n.addr.parse().map_err(|e| {
                                anyhow::anyhow!(
                                    "cluster `{}` node `{}` addr `{}` invalid: {e}",
                                    cluster_id,
                                    n.id,
                                    n.addr
                                )
                            });
                            addr.map(|addr| ai_engine_cluster::leader::WorkerEndpoint {
                                node_id: n.id.clone(),
                                addr,
                                fingerprint: n.cert_fingerprint.clone(),
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?
                };

            let weights_path = std::path::PathBuf::from(&cluster_cfg.model.weights_path);
            let model_cfg = match &cluster_cfg.model.config_path {
                Some(p) => ai_engine_runtime::config::ModelConfig::from_file(
                    std::path::Path::new(p),
                )?,
                None => ai_engine_runtime::config::ModelConfig::from_gguf_file(&weights_path)?,
            };
            let n_layers = model_cfg.n_layers;
            let partition_override = if cluster_cfg.partition_override.is_empty() {
                None
            } else {
                let parsed: Result<Vec<_>, _> = cluster_cfg
                    .partition_override
                    .iter()
                    .map(|po| parse_layer_range(&po.layers).map(|r| (po.node.clone(), r)))
                    .collect();
                Some(parsed?)
            };
            let lcfg = ai_engine_cluster::leader::LeaderConfig {
                cluster_id: cluster_id.clone(),
                leader_node_id: cluster_cfg.leader.clone(),
                model_id: cluster_cfg.model.id.clone(),
                n_layers,
                // Approximate sizing; refined in v0.3 with real model metadata.
                layer_bytes: 256 * 1024,
                embed_output_bytes: 256 * 1024,
                per_node_overhead: 64 * 1024,
                workers: worker_endpoints,
                partition_override,
            };
            let leader = ai_engine_cluster::leader::ClusterLeader::start(&identity, lcfg).await?;
            let leader_arc = Arc::new(leader);
            let cluster_view: Arc<dyn ai_engine_core::cluster_view::ClusterView> =
                Arc::new(ai_engine_cluster::view::ClusterViewImpl::new(
                    leader_arc.capabilities().to_vec(),
                    leader_arc.manifest().clone(),
                    leader_arc.metrics(),
                ));
            cluster_view_opt = Some(cluster_view);
            let tokenizer = match &cluster_cfg.model.tokenizer_path {
                Some(p) => Arc::new(ai_engine_tokenizer::HfTokenizer::from_path(p)?),
                None => Arc::new(ai_engine_runtime::load_tokenizer_from_gguf(&weights_path)?),
            };
            // Plan 3 simplification: workers cover all layers; leader hosts none.
            let leader_layers = 0..0;
            let state = ai_engine_cluster::provider::LeaderState {
                leader: leader_arc,
                model_cfg,
                model_path: weights_path,
                tokenizer,
                leader_layers,
            };
            // Wire the cluster under exactly one [[provider]] entry referencing
            // it. Multiple providers pointing at the same cluster aren't
            // supported in v0.2 (would require sharing the live `ClusterLeader`
            // — that's Plan 4+).
            let matching: Vec<&ai_engine_config::Provider> = cfg
                .providers
                .iter()
                .filter(|p| {
                    p.kind == "local-cluster" && p.cluster.as_deref() == Some(cluster_id)
                })
                .collect();
            match matching.as_slice() {
                [p] => {
                    let provider_arc: Arc<dyn Provider> = Arc::new(
                        ai_engine_cluster::provider::ClusterProvider::new_leader_with_state(
                            p.id.clone(),
                            Arc::new(state),
                        ),
                    );
                    cluster_providers.insert(p.id.clone(), provider_arc);
                }
                [] => anyhow::bail!(
                    "cluster `{}` has no [[provider]] referencing it", cluster_id
                ),
                _ => anyhow::bail!(
                    "cluster `{}` is referenced by {} providers; v0.2 supports at most one",
                    cluster_id,
                    matching.len()
                ),
            }
        }
    }

    build_gateway_app_state(cfg, cluster_providers, cluster_view_opt, discovered_resources)
}

/// Build the leaderless (Phase B) node: form the full mesh across all
/// configured cluster nodes, compute the agreed manifest, load this node's
/// hosted stages, spawn a `serve_peer` loop per inbound peer connection, and
/// construct a local `Coordinator`. Returns the per-provider `ClusterProvider`
/// map (backed by the Coordinator) plus a `ClusterView` for the dashboard.
///
/// v1 scope: NdArray backend, capabilities synthesized from the static node
/// list (no QUIC capability exchange) so every node derives a byte-identical
/// `agreed_manifest`.
async fn build_leaderless_node(
    cfg: &Config,
    cluster_cfg: &ai_engine_config::Cluster,
    node_id: &str,
) -> anyhow::Result<(
    HashMap<String, Arc<dyn Provider>>,
    Arc<dyn ai_engine_core::cluster_view::ClusterView>,
)> {
    use ai_engine_cluster::capability::{BackendKind, Capability};

    let identity = load_or_generate_node_identity(node_id)?;
    eprintln!(
        "ai-engine peer `{}` fingerprint: {}",
        node_id, identity.fingerprint
    );

    // Model config + layer count.
    let weights_path = std::path::PathBuf::from(&cluster_cfg.model.weights_path);
    let model_cfg = match &cluster_cfg.model.config_path {
        Some(p) => {
            ai_engine_runtime::config::ModelConfig::from_file(std::path::Path::new(p))?
        }
        None => ai_engine_runtime::config::ModelConfig::from_gguf_file(&weights_path)?,
    };
    let n_layers = model_cfg.n_layers;

    // Synthesize a capability per node from config. Uniform compute/memory so
    // `agreed_manifest` (which canonicalises by node_id) yields the same
    // manifest on every node. Memory is set generously so the DP is feasible.
    let backend_of = |s: &str| match s {
        "cuda" => BackendKind::Cuda,
        "metal" => BackendKind::Metal,
        "wgpu" => BackendKind::Wgpu,
        _ => BackendKind::Cpu,
    };
    let caps: Vec<Capability> = cluster_cfg
        .nodes
        .iter()
        .map(|n| Capability {
            node_id: n.id.clone(),
            backend: backend_of(&n.backend),
            device_index: n.device_index,
            available_memory_bytes: 64 * 1024 * 1024 * 1024,
            compute_score: 100,
            link_mbps_to_leader: 0,
        })
        .collect();

    let layer_bytes = 256 * 1024;
    let embed_output_bytes = 256 * 1024;
    let per_node_overhead = 64 * 1024;

    let manifest = if cluster_cfg.partition_override.is_empty() {
        ai_engine_cluster::partition::agreed_manifest(
            &cluster_cfg.model.id,
            &caps,
            n_layers,
            layer_bytes,
            embed_output_bytes,
            per_node_overhead,
        )?
    } else {
        let ranges: Vec<(String, std::ops::Range<usize>)> = cluster_cfg
            .partition_override
            .iter()
            .map(|po| parse_layer_range(&po.layers).map(|r| (po.node.clone(), r)))
            .collect::<Result<_, _>>()?;
        ai_engine_cluster::partition::manual_partition(
            &cluster_cfg.model.id,
            &caps,
            n_layers,
            ranges,
            layer_bytes,
            embed_output_bytes,
            per_node_overhead,
        )?
    };

    let my_assignment = manifest.for_node(node_id).ok_or_else(|| {
        anyhow::anyhow!("node `{node_id}` absent from agreed manifest")
    })?;

    // Tokenizer (coordinator side only needs it for prompt<->ids bridging).
    let tokenizer: Arc<ai_engine_tokenizer::HfTokenizer> =
        match &cluster_cfg.model.tokenizer_path {
            Some(p) => Arc::new(ai_engine_tokenizer::HfTokenizer::from_path(p)?),
            None => Arc::new(ai_engine_runtime::load_tokenizer_from_gguf(&weights_path)?),
        };

    // Form the full mesh.
    let bind: std::net::SocketAddr = {
        let me = cluster_cfg
            .nodes
            .iter()
            .find(|n| n.id == node_id)
            .expect("node present in cluster (checked by caller)");
        me.addr.parse().map_err(|e| {
            anyhow::anyhow!("node `{node_id}` addr `{}` invalid: {e}", me.addr)
        })?
    };
    // Trust every peer fingerprint *and our own* — the coordinator forms a
    // loopback connection to itself so hops it hosts are driven over the same
    // wire protocol as remote hops.
    let peer_fingerprints: Vec<String> = cluster_cfg
        .nodes
        .iter()
        .map(|n| n.cert_fingerprint.clone())
        .collect();
    let endpoint = ai_engine_cluster::transport::mesh::mesh_endpoint(
        &identity,
        bind,
        &peer_fingerprints,
    )?;
    let peers: Vec<ai_engine_cluster::transport::mesh::Peer> = cluster_cfg
        .nodes
        .iter()
        .filter(|n| n.id != node_id)
        .map(|n| {
            n.addr
                .parse()
                .map_err(|e| anyhow::anyhow!("peer `{}` addr invalid: {e}", n.id))
                .map(|addr| ai_engine_cluster::transport::mesh::Peer {
                    node_id: n.id.clone(),
                    addr,
                    fingerprint: n.cert_fingerprint.clone(),
                })
        })
        .collect::<Result<_, _>>()?;

    tracing::info!(node_id = %node_id, peers = peers.len(), "leaderless: forming mesh");
    let mut connections =
        ai_engine_cluster::transport::mesh::connect_mesh(&endpoint, node_id, &peers).await?;
    tracing::info!(node_id = %node_id, "leaderless: mesh complete");

    // Loopback self-connection: dial our own endpoint and accept it
    // concurrently, then serve our own stages over it. This lets the local
    // Coordinator reach hops *this* node hosts via the same RPC path as remote
    // peers (the coordinator is itself a pipeline node).
    let self_conn = {
        let dial_ep = endpoint.clone();
        let dial_id = node_id.to_string();
        let dial = tokio::spawn(async move {
            ai_engine_cluster::transport::mesh::connect_self(&dial_ep, &dial_id, bind).await
        });
        let (accepted_id, accepted_conn) =
            ai_engine_cluster::transport::mesh::accept_one(&endpoint).await?;
        if accepted_id != node_id {
            anyhow::bail!(
                "leaderless self-loopback: expected hello `{node_id}`, got `{accepted_id}`"
            );
        }
        let dialed = dial
            .await
            .map_err(|e| anyhow::anyhow!("self-dial task: {e}"))??;
        // Serve our own stages to the loopback's accepted side.
        let stages = ai_engine_cluster::peer::build_peer_stages::<burn_ndarray::NdArray>(
            &weights_path,
            &model_cfg,
            my_assignment.layer_range.clone(),
            my_assignment.hosts_embedding,
            my_assignment.hosts_output,
        )?;
        tokio::spawn(async move {
            if let Err(e) = ai_engine_cluster::peer::serve_peer::<burn_ndarray::NdArray>(
                accepted_conn,
                stages,
            )
            .await
            {
                tracing::warn!(error = %e, "self serve_peer loop ended");
            }
        });
        dialed
    };

    // Spawn a `serve_peer` loop per inbound peer connection so peers can drive
    // forward passes through our hosted stages. Each loop needs its own
    // `PeerStages` (not Clone), so we build one set per connection.
    let layer_range = my_assignment.layer_range.clone();
    let hosts_embedding = my_assignment.hosts_embedding;
    let hosts_output = my_assignment.hosts_output;
    for (peer_id, conn) in connections.iter() {
        let stages = ai_engine_cluster::peer::build_peer_stages::<burn_ndarray::NdArray>(
            &weights_path,
            &model_cfg,
            layer_range.clone(),
            hosts_embedding,
            hosts_output,
        )?;
        let conn = conn.clone();
        let peer_id = peer_id.clone();
        tokio::spawn(async move {
            if let Err(e) =
                ai_engine_cluster::peer::serve_peer::<burn_ndarray::NdArray>(conn, stages).await
            {
                tracing::warn!(peer = %peer_id, error = %e, "serve_peer loop ended");
            }
        });
    }

    // Register the loopback after the per-peer serve loops so it isn't given a
    // duplicate (dialer-side) serve loop above.
    connections.insert(node_id.to_string(), self_conn);

    // Local Coordinator over the mesh.
    let coordinator = Arc::new(ai_engine_cluster::coordinator::Coordinator::new(
        node_id.to_string(),
        manifest.clone(),
        connections,
    ));
    let missing = coordinator.missing_peers();
    if !missing.is_empty() {
        anyhow::bail!("leaderless coordinator missing peers after mesh: {missing:?}");
    }

    let coord_state = Arc::new(ai_engine_cluster::provider::CoordinatorState {
        coordinator,
        tokenizer,
    });

    // Wire the cluster under the [[provider]] referencing it.
    let matching: Vec<&ai_engine_config::Provider> = cfg
        .providers
        .iter()
        .filter(|p| p.kind == "local-cluster" && p.cluster.as_deref() == Some(&cluster_cfg.id))
        .collect();
    let mut cluster_providers: HashMap<String, Arc<dyn Provider>> = HashMap::new();
    match matching.as_slice() {
        [p] => {
            let provider_arc: Arc<dyn Provider> = Arc::new(
                ai_engine_cluster::provider::ClusterProvider::new_coordinator(
                    p.id.clone(),
                    coord_state,
                ),
            );
            cluster_providers.insert(p.id.clone(), provider_arc);
        }
        [] => anyhow::bail!(
            "leaderless cluster `{}` has no [[provider]] referencing it",
            cluster_cfg.id
        ),
        _ => anyhow::bail!(
            "leaderless cluster `{}` referenced by {} providers; at most one supported",
            cluster_cfg.id,
            matching.len()
        ),
    }

    let cluster_view: Arc<dyn ai_engine_core::cluster_view::ClusterView> =
        Arc::new(ai_engine_cluster::view::ClusterViewImpl::new(
            caps,
            manifest,
            Arc::new(ai_engine_cluster::metrics::ClusterMetrics::new()),
        ));

    Ok((cluster_providers, cluster_view))
}

/// Parse a TOML `layers = "start..end"` string into a `Range<usize>`.
fn parse_layer_range(s: &str) -> anyhow::Result<std::ops::Range<usize>> {
    let (start, end) = s
        .split_once("..")
        .ok_or_else(|| anyhow::anyhow!("invalid layer range `{s}`: missing `..`"))?;
    let start: usize = start
        .trim()
        .parse()
        .map_err(|e| anyhow::anyhow!("invalid layer range `{s}` start: {e}"))?;
    let end: usize = end
        .trim()
        .parse()
        .map_err(|e| anyhow::anyhow!("invalid layer range `{s}` end: {e}"))?;
    if start >= end {
        anyhow::bail!("invalid layer range `{s}`: start must be < end");
    }
    Ok(start..end)
}

/// Load this node's TLS identity from disk, or generate + persist fresh.
///
/// Persistence is required so the node's fingerprint stays stable across
/// restarts — otherwise peers would need their `cert_fingerprint` config
/// rewritten every bounce.
fn load_or_generate_node_identity(
    node_id: &str,
) -> anyhow::Result<ai_engine_cluster::tls::NodeIdentity> {
    let dir = dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".ai-engine");
    ai_engine_cluster::tls::load_or_generate_node_identity(node_id, &dir)
}

/// Construct the gateway-only AppState. `cluster_providers` are pre-built
/// providers (leader-mode) keyed by their [[provider]] id; they shadow any
/// `kind = "local-cluster"` entry in `cfg.providers`.
fn build_gateway_app_state(
    cfg: &Config,
    cluster_providers: HashMap<String, Arc<dyn Provider>>,
    cluster_view_opt: Option<Arc<dyn ai_engine_core::cluster_view::ClusterView>>,
    discovered_resources: std::collections::HashMap<
        String,
        ai_engine_core::resources::NodeResources,
    >,
) -> anyhow::Result<Arc<AppState>> {
    // --- providers ---
    let mut providers = ProviderRegistry::new();
    for p in &cfg.providers {
        let creds = Credentials {
            api_key: p.api_key.clone(),
            raw_bearer: None,
            extra_headers: p
                .extra_headers
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        };
        let provider: Arc<dyn Provider> = match p.kind.as_str() {
            "openai" => Arc::new(ai_engine_openai::OpenAiProvider::new(
                p.id.clone(),
                p.base_url.clone(),
                p.timeout_secs,
                p.http2,
            )),
            "anthropic" => Arc::new(ai_engine_anthropic::AnthropicProvider::new(
                p.id.clone(),
                p.base_url.clone(),
                p.timeout_secs,
            )),
            "local-cluster" => {
                // Leader: provider was pre-constructed above. Worker/Gateway
                // for an unrelated cluster: fall back to a worker stub that
                // returns Unsupported on every call (the [[route]] for it is
                // unreachable on this node).
                if let Some(arc) = cluster_providers.get(&p.id) {
                    arc.clone()
                } else {
                    Arc::new(ai_engine_cluster::provider::ClusterProvider::new_worker(
                        p.id.clone(),
                    ))
                }
            }
            "candle" => {
                #[cfg(feature = "backend-candle")]
                {
                    let weights = p.weights_path.as_deref().ok_or_else(|| {
                        anyhow::anyhow!("provider '{}': candle requires weights_path", p.id)
                    })?;
                    let gguf = std::path::Path::new(weights);
                    let device = p.device.as_deref().unwrap_or("auto");
                    let cp = match p.engine.as_deref().unwrap_or("paged") {
                        "pool" => {
                            let pool_size = p.pool_size.unwrap_or(2);
                            ai_engine_candle::CandleProvider::new(&p.id, gguf, device, pool_size)?
                        }
                        _ => {
                            ai_engine_candle::CandleProvider::new_paged(
                                &p.id, gguf, device,
                                p.max_num_seqs.unwrap_or(32),
                                p.block_size.unwrap_or(16),
                                p.kv_cache_blocks.unwrap_or(4096),
                            )?
                        }
                    };
                    Arc::new(cp) as Arc<dyn Provider>
                }
                #[cfg(not(feature = "backend-candle"))]
                {
                    anyhow::bail!(
                        "provider '{}' uses kind=candle but this binary was built without the 'backend-candle' feature; rebuild with --features backend-candle",
                        p.id
                    );
                }
            }
            "rustyllm" => {
                #[cfg(feature = "backend-rustyllm")]
                {
                    let weights = p.weights_path.as_deref().ok_or_else(|| {
                        anyhow::anyhow!(
                            "provider '{}': rustyllm requires weights_path (model dir or hub id)",
                            p.id
                        )
                    })?;
                    let device = p.device.as_deref().unwrap_or("auto");
                    let rp = ai_engine_rustyllm::RustyllmProvider::new(
                        &p.id, weights, device, 2048,
                    )?;
                    Arc::new(rp) as Arc<dyn Provider>
                }
                #[cfg(not(feature = "backend-rustyllm"))]
                {
                    anyhow::bail!(
                        "provider '{}' uses kind=rustyllm but this binary was built without the 'backend-rustyllm' feature; rebuild with --features backend-rustyllm",
                        p.id
                    );
                }
            }
            other => anyhow::bail!(
                "unknown provider kind `{other}` (validated upstream — this is a bug)"
            ),
        };
        providers.insert(p.id.clone(), provider, creds);
    }
    let providers = Arc::new(providers);

    // --- route rules ---
    let rules: Vec<(String, String, Option<String>)> = cfg
        .routes
        .iter()
        .map(|r| {
            (
                r.r#match.model.clone(),
                r.provider.clone(),
                r.upstream_model.clone(),
            )
        })
        .collect();
    let model_route = ModelRouteStage::from_strings(rules)?;

    // --- stages ---
    let auth_mode = match cfg.auth.mode.as_str() {
        "passthrough" => AuthMode::Passthrough,
        "shared-key" => AuthMode::SharedKey {
            keys: cfg
                .auth
                .master_keys
                .iter()
                .map(|m| (m.key.clone(), m.name.clone()))
                .collect(),
        },
        other => {
            anyhow::bail!("unknown auth.mode `{other}` (validated upstream — this is a bug)")
        }
    };
    let auth = AuthStage { mode: auth_mode };
    let content_policy = ContentPolicyStage::new(
        cfg.content_policy.max_request_bytes,
        cfg.content_policy.prompt_injection_patterns.clone(),
    )?;
    let tracker = Arc::new(ai_engine_stages::LoadTracker::new(
        providers.ids().cloned().collect::<Vec<_>>(),
    ));
    let gateway_metrics = Arc::new(ai_engine_core::metrics::GatewayMetrics::new(
        providers.ids().cloned().collect::<Vec<_>>(),
    ));
    let forward = ForwardStage {
        providers: providers.clone(),
        tracker,
        metrics: gateway_metrics.clone(),
    };
    let activity = Arc::new(ai_engine_core::activity::ActivityLog::default());
    let log = LogStage::stdout()
        .with_activity(activity.clone())
        .with_metrics(gateway_metrics.clone());

    // Background health prober for the cluster dashboard: periodically poll each
    // HTTP provider's `/models`; in-process providers (no base_url) report up.
    let health = Arc::new(ai_engine_core::health::HealthStore::new());
    spawn_health_prober(cfg, health.clone());

    let mut stages = StageRegistry::new();
    stages.insert("auth", Arc::new(auth));
    stages.insert("content_policy", Arc::new(content_policy));
    stages.insert("model_route", Arc::new(model_route));
    stages.insert("forward", Arc::new(forward));
    stages.insert("log", Arc::new(log));

    // --- pipelines per route ---
    const ROUTES: &[&str] = &["/v1/chat/completions", "/v1/messages", "/v1/embeddings"];
    let mut pipelines: HashMap<&'static str, ArcSwap<ai_engine_core::pipeline::Pipeline>> =
        HashMap::new();
    for &route in ROUTES {
        let Some(pl_cfg) = cfg.pipeline.get(route) else {
            continue;
        };
        let pipeline = stages
            .build_pipeline(&pl_cfg.stages)
            .map_err(|e| anyhow::anyhow!("pipeline {route}: {e}"))?;
        pipelines.insert(route, ArcSwap::new(Arc::new(pipeline)));
    }
    for unknown in cfg
        .pipeline
        .keys()
        .filter(|k| !ROUTES.contains(&k.as_str()))
    {
        tracing::warn!(route = %unknown, "config defines pipeline for unknown route; skipping");
    }

    // --- /v1/models surface from route table ---
    let openai_models = cfg.routes.iter().map(|r| r.r#match.model.clone()).collect();
    let services = build_services(cfg);

    Ok(Arc::new(AppState {
        pipelines,
        openai_models,
        ready: AtomicBool::new(true),
        cluster: cluster_view_opt,
        services,
        gateway_metrics,
        activity,
        health,
        resources: Arc::new(discovered_resources),
    }))
}

/// Spawn a background task that probes each HTTP provider's `/models` endpoint
/// every few seconds and records up/down + latency in `health`. In-process
/// providers (no `base_url`) are marked up immediately.
fn spawn_health_prober(cfg: &Config, health: Arc<ai_engine_core::health::HealthStore>) {
    // (provider_id, optional http base_url) captured up front; cfg isn't 'static.
    let targets: Vec<(String, Option<String>)> = cfg
        .providers
        .iter()
        .map(|p| {
            let url = if p.base_url.is_empty() {
                None
            } else {
                Some(p.base_url.trim_end_matches('/').to_string())
            };
            (p.id.clone(), url)
        })
        .collect();

    tokio::spawn(async move {
        let client = reqwest::Client::new();
        let mut ticker = tokio::time::interval(std::time::Duration::from_secs(5));
        loop {
            ticker.tick().await;
            for (id, url) in &targets {
                match url {
                    // In-process provider: always up, zero latency.
                    None => health.set(id, true, 0),
                    Some(base) => {
                        let t0 = std::time::Instant::now();
                        let ok = client
                            .get(format!("{base}/models"))
                            .timeout(std::time::Duration::from_secs(3))
                            .send()
                            .await
                            .map(|r| r.status().is_success())
                            .unwrap_or(false);
                        health.set(id, ok, t0.elapsed().as_millis() as u64);
                    }
                }
            }
        }
    });
}
