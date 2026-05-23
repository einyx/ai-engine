use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use airproxy_config::Config;
use airproxy_http::AppState;
use airproxy_provider::provider::{Credentials, Provider};
use airproxy_stages::{
    ProviderRegistry, StageRegistry,
    auth::{AuthMode, AuthStage},
    content_policy::ContentPolicyStage,
    forward::ForwardStage,
    log::LogStage,
    model_route::ModelRouteStage,
};
use arc_swap::ArcSwap;

/// Build a complete `AppState` from a validated `Config`.
///
/// This is the wiring layer: providers are instantiated, stages are constructed
/// with their config-derived parameters, the stage registry is populated, and
/// one `Pipeline` per `[pipeline."<route>"]` is built.
///
/// Routes recognized in v1: `/v1/chat/completions`, `/v1/messages`,
/// `/v1/embeddings`. Routes in `[pipeline.…]` other than these are silently
/// skipped (warned via tracing).
pub fn build_app_state(cfg: &Config) -> anyhow::Result<Arc<AppState>> {
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
            "openai" => Arc::new(airproxy_openai::OpenAiProvider::new(
                p.id.clone(),
                p.base_url.clone(),
                p.timeout_secs,
                p.http2,
            )),
            "anthropic" => Arc::new(airproxy_anthropic::AnthropicProvider::new(
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
    let mut pipelines: HashMap<&'static str, ArcSwap<airproxy_core::pipeline::Pipeline>> =
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
