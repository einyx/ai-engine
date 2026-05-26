//! Startup-time ingest of mDNS-discovered upstreams into the running config.
//!
//! Discovered Ollama endpoints are merged into `cfg.providers` / `cfg.routes`
//! *before* `app::build_app_state` runs, so the rest of the wiring (forward
//! stage, `model_route`, `/cluster/services`, the UI pipeline view) treats them
//! exactly like hand-written `[[provider]]` blocks. See the design doc:
//! `docs/superpowers/specs/2026-05-26-mdns-ollama-discovery-design.md`.

use ai_engine_config::{Config, Provider, Route, RouteMatch};
use std::collections::HashSet;
use std::time::Duration;

/// If `[discovery] ollama_mdns = true`, browse mDNS and append a provider +
/// routes for each discovered Ollama. Never fatal: a discovery error or an
/// empty result leaves `cfg` untouched and the gateway serves its static
/// providers as usual.
///
/// Model-name conflicts are resolved first-wins: a model already routed (by
/// config or an earlier-sorted discovery) is not overridden.
/// Returns a map of `provider_id -> advertised host resources` for the
/// discovered (remote) providers, so the gateway can show their CPU/mem/disk.
pub async fn merge_discovered_ollamas(
    cfg: &mut Config,
) -> std::collections::HashMap<String, ai_engine_core::resources::NodeResources> {
    let mut resources = std::collections::HashMap::new();
    let Some(disc) = &cfg.discovery else {
        return resources;
    };
    if !disc.ollama_mdns {
        return resources;
    }
    let timeout = Duration::from_secs(disc.timeout_secs);
    tracing::info!(timeout_secs = disc.timeout_secs, "browsing mDNS for Ollama endpoints");
    let found = match ai_engine_cluster::discovery::discover_ollamas(timeout).await {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(error = %e, "Ollama mDNS discovery failed; continuing without it");
            return resources;
        }
    };
    if found.is_empty() {
        tracing::info!("Ollama mDNS discovery found nothing");
        return resources;
    }

    // (provider_id, model) pairs already routed — avoid exact-duplicate routes
    // while still letting multiple providers share a model (the forward stage
    // load-balances across them).
    let mut existing: HashSet<(String, String)> = cfg
        .routes
        .iter()
        .map(|r| (r.provider.clone(), r.r#match.model.clone()))
        .collect();

    for o in found {
        let provider_id = format!("ollama-{}", o.label);
        if cfg.providers.iter().any(|p| p.id == provider_id) {
            continue;
        }
        let base_url = format!("{}/v1", o.url.trim_end_matches('/'));
        cfg.providers.push(ollama_provider(&provider_id, &base_url));
        resources.insert(provider_id.clone(), o.resources.clone());

        let mut added = 0usize;
        for model in o.models {
            if !existing.insert((provider_id.clone(), model.clone())) {
                continue; // this provider already routes this model
            }
            cfg.routes.push(Route {
                r#match: RouteMatch { model },
                provider: provider_id.clone(),
                upstream_model: None,
            });
            added += 1;
        }
        tracing::info!(provider = %provider_id, url = %o.url, routes = added, "auto-registered discovered Ollama");
        eprintln!(
            "ai-engine discovered Ollama `{provider_id}` at {} ({added} model route(s))",
            o.url
        );
    }
    resources
}

/// A `[[provider]]` for an OpenAI-compatible Ollama endpoint (no auth, http/1.1,
/// generous timeout for local models). Mirrors the documented manual config.
fn ollama_provider(id: &str, base_url: &str) -> Provider {
    Provider {
        id: id.to_string(),
        kind: "openai".to_string(),
        base_url: base_url.to_string(),
        api_key: None,
        timeout_secs: 600,
        http2: false,
        extra_headers: Default::default(),
        cluster: None,
        weights_path: None,
        device: None,
        pool_size: None,
        engine: None,
        max_num_seqs: None,
        block_size: None,
        kv_cache_blocks: None,
    }
}
