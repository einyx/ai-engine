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
        if !matches!(p.kind.as_str(), "openai" | "anthropic") {
            anyhow::bail!("unknown provider kind `{}` (provider `{}`); expected `openai` or `anthropic`", p.kind, p.id);
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

    // Pipelines: each must contain `forward` and at least one known terminal stage.
    // The set of known terminals in v1 is just `log` — when more terminals are added,
    // this list grows. Pipelines reference stage ids only; we don't validate them against
    // a registry here because that lives in airproxy-stages.
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
