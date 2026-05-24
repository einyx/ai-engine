use std::collections::HashSet;

use crate::Config;

pub fn validate(cfg: &Config) -> anyhow::Result<()> {
    // Auth mode
    if !matches!(cfg.auth.mode.as_str(), "passthrough" | "shared-key") {
        anyhow::bail!("auth.mode must be `passthrough` or `shared-key`, got `{}`", cfg.auth.mode);
    }

    // Providers: unique ids, known kinds
    let mut ids: HashSet<&String> = HashSet::new();
    for p in &cfg.providers {
        if !ids.insert(&p.id) {
            anyhow::bail!("duplicate provider id `{}`", p.id);
        }
        if !matches!(p.kind.as_str(), "openai" | "anthropic" | "local-cluster") {
            anyhow::bail!("unknown provider kind `{}` (provider `{}`); expected `openai`, `anthropic`, or `local-cluster`", p.kind, p.id);
        }
    }

    // Routes reference known providers
    for r in &cfg.routes {
        if !cfg.providers.iter().any(|p| p.id == r.provider) {
            anyhow::bail!("route `{}` references unknown provider `{}`", r.r#match.model, r.provider);
        }
    }

    // Format-pinning sanity guard
    for r in &cfg.routes {
        let Some(provider) = cfg.providers.iter().find(|p| p.id == r.provider) else { continue; };
        let m = &r.r#match.model;
        if provider.kind == "openai" && m.starts_with("claude") {
            anyhow::bail!("route `{m}` binds to openai-kind provider `{}` but model looks anthropic", provider.id);
        }
        if provider.kind == "anthropic" && (m.starts_with("gpt") || m.starts_with("text-embedding")) {
            anyhow::bail!("route `{m}` binds to anthropic-kind provider `{}` but model looks openai", provider.id);
        }
    }

    // Cluster validation
    let mut cluster_ids: HashSet<&String> = HashSet::new();
    for cluster in &cfg.clusters {
        if !cluster_ids.insert(&cluster.id) {
            anyhow::bail!("duplicate cluster id `{}`", cluster.id);
        }
        // Leader must reference an existing node.
        if !cluster.nodes.iter().any(|n| n.id == cluster.leader) {
            anyhow::bail!(
                "cluster `{}` leader `{}` does not reference any node in [[cluster.node]]",
                cluster.id,
                cluster.leader
            );
        }
        // Node ids and addrs unique within a cluster, backend kind valid.
        let mut node_ids: HashSet<&String> = HashSet::new();
        let mut node_addrs: HashSet<&String> = HashSet::new();
        for node in &cluster.nodes {
            if !node_ids.insert(&node.id) {
                anyhow::bail!(
                    "duplicate cluster node id `{}` in cluster `{}`",
                    node.id,
                    cluster.id
                );
            }
            if !node_addrs.insert(&node.addr) {
                anyhow::bail!(
                    "duplicate cluster node addr `{}` in cluster `{}`",
                    node.addr,
                    cluster.id
                );
            }
            if !matches!(node.backend.as_str(), "cpu" | "cuda" | "metal" | "wgpu") {
                anyhow::bail!(
                    "unknown backend kind `{}` for cluster node `{}` (expected cpu | cuda | metal | wgpu)",
                    node.backend,
                    node.id
                );
            }
            if !node.cert_fingerprint.starts_with("sha256:") {
                anyhow::bail!(
                    "cluster node `{}` cert_fingerprint must start with `sha256:`",
                    node.id
                );
            }
        }
        if let Some(disc) = &cluster.discover {
            if disc.expected_workers == 0 {
                anyhow::bail!(
                    "cluster `{}` discover.expected_workers must be > 0",
                    cluster.id
                );
            }
        }
    }

    // local-cluster providers must reference an existing cluster
    for p in &cfg.providers {
        if p.kind == "local-cluster" {
            let target = p.cluster.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "provider `{}` kind=local-cluster requires a `cluster` field",
                    p.id
                )
            })?;
            if !cfg.clusters.iter().any(|c| &c.id == target) {
                anyhow::bail!(
                    "provider `{}` references unknown cluster `{}`",
                    p.id,
                    target
                );
            }
        }
    }

    // Pipelines: each must contain `forward` and at least one known terminal stage.
    // The set of known terminals in v1 is just `log` — when more terminals are added,
    // this list grows. Pipelines reference stage ids only; we don't validate them against
    // a registry here because that lives in ai-engine-stages.
    for (route, pl) in &cfg.pipeline {
        if !pl.stages.iter().any(|s| s == "forward") {
            anyhow::bail!("pipeline `{route}` has no `forward` stage");
        }
        let has_terminal = pl.stages.iter().any(|s| matches!(s.as_str(), "log"));
        if !has_terminal {
            anyhow::bail!("pipeline `{route}` has no terminal stage (e.g., `log`)");
        }
    }

    Ok(())
}
