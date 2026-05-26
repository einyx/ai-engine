use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use ai_engine_http::AppState;
use arc_swap::ArcSwap;
use tokio::signal::unix::{SignalKind, signal};

/// Future that resolves when SIGTERM or Ctrl-C arrives, then waits `grace`
/// before completing (giving in-flight requests time to drain).
pub async fn shutdown_signal(grace: Duration) {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    let term = async {
        if let Ok(mut s) = signal(SignalKind::terminate()) {
            s.recv().await;
        } else {
            std::future::pending::<()>().await;
        }
    };
    tokio::select! {
        _ = ctrl_c => {}
        _ = term => {}
    }
    tracing::info!(?grace, "shutdown signal received; draining");
    tokio::time::sleep(grace).await;
}

/// Listen for SIGHUP and atomically swap pipelines on each tick.
/// On any reload failure (invalid TOML, validation error, build error) the
/// old state stays in place and a warning is emitted.
pub fn spawn_reload(cfg_path: PathBuf, state: Arc<AppState>) {
    tokio::spawn(async move {
        let mut hup = match signal(SignalKind::hangup()) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(error = %e, "cannot install SIGHUP handler; reload disabled");
                return;
            }
        };
        while hup.recv().await.is_some() {
            tracing::info!(path = %cfg_path.display(), "SIGHUP: reloading config");
            match reload(&cfg_path, &state).await {
                Ok(()) => tracing::info!("config reload ok"),
                Err(e) => tracing::warn!(error = %e, "config reload failed; keeping old"),
            }
        }
    });
}

async fn reload(cfg_path: &std::path::Path, state: &Arc<AppState>) -> anyhow::Result<()> {
    let cfg = ai_engine_config::Config::load(cfg_path)?;
    let new_state = crate::app::build_app_state(&cfg, "localhost", Default::default()).await?;
    // Atomic per-pipeline swap. Routes only present in the new state are
    // ignored (would require re-binding the router). Routes removed from
    // the new config also stay live — callers see the previous pipeline.
    // This is intentional for v1: bind & route-set changes require restart.
    for (route, new_pipeline) in new_state.pipelines.iter() {
        if let Some(slot) = state.pipelines.get(route) {
            slot.store(new_pipeline.load_full());
        } else {
            tracing::warn!(route = %route, "reload: new route not present in original config; ignored until restart");
        }
    }
    for route in state.pipelines.keys() {
        if !new_state.pipelines.contains_key(route) {
            tracing::warn!(route = %route, "reload: route removed in new config; keeping old until restart");
        }
    }
    Ok(())
}

/// `ArcSwap` is used elsewhere; re-export to avoid leaking the dep boundary if needed.
pub type _PipelineSlot = ArcSwap<ai_engine_core::pipeline::Pipeline>;
