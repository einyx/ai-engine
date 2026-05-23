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
pub fn build_app_state(cfg: &Config, node_id: &str) -> anyhow::Result<Arc<AppState>> {
    let role = resolve_role(cfg, node_id);
    if let NodeRole::Worker { .. } = &role {
        // Worker mode: no HTTP pipelines, just health endpoints.
        return Ok(Arc::new(AppState {
            pipelines: HashMap::new(),
            openai_models: vec![],
            ready: AtomicBool::new(true),
        }));
    }

    if let NodeRole::Leader { .. } = &role {
        // Leader: pipeline construction (real cluster startup wired in Task 7).
        todo!("leader-mode build_app_state requires async ClusterLeader::start; wired in Task 7")
    }

    build_gateway_app_state(cfg)
}

/// Construct the gateway-only AppState (no cluster providers).
fn build_gateway_app_state(cfg: &Config) -> anyhow::Result<Arc<AppState>> {
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
