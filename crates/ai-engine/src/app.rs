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
pub async fn build_app_state(cfg: &Config, node_id: &str) -> anyhow::Result<Arc<AppState>> {
    let role = resolve_role(cfg, node_id);
    if let NodeRole::Worker { .. } = &role {
        // Worker mode: no HTTP pipelines, just health endpoints.
        return Ok(Arc::new(AppState {
            pipelines: HashMap::new(),
            openai_models: vec![],
            ready: AtomicBool::new(true),
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

            let model_cfg = ai_engine_runtime::config::ModelConfig::from_file(
                std::path::Path::new(&cluster_cfg.model.config_path),
            )?;
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
            let tokenizer =
                ai_engine_tokenizer::HfTokenizer::from_path(&cluster_cfg.model.tokenizer_path)?;
            // Plan 3 simplification: workers cover all layers; leader hosts none.
            let leader_layers = 0..0;
            let state = ai_engine_cluster::provider::LeaderState {
                leader: Arc::new(leader),
                model_cfg,
                model_path: std::path::PathBuf::from(&cluster_cfg.model.weights_path),
                tokenizer: Arc::new(tokenizer),
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

    build_gateway_app_state(cfg, cluster_providers)
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
    let forward = ForwardStage {
        providers: providers.clone(),
    };
    let log = LogStage::stdout();

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

    Ok(Arc::new(AppState {
        pipelines,
        openai_models,
        ready: AtomicBool::new(true),
    }))
}
