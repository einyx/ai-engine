use clap::Parser;

mod cli;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = cli::Cli::parse();

    // The advertiser doesn't need (or load) a gateway config.
    if let Some(cli::Command::AdvertiseOllama { ollama_url, label }) = cli.command {
        ai_engine::init_tracing_default();
        return ai_engine::advertise::run(ollama_url, label).await;
    }

    let mut cfg = ai_engine_config::Config::load(&cli.config)?;
    ai_engine::init_tracing(&cfg.server);

    if cli.check {
        println!("config OK: {}", cli.config.display());
        return Ok(());
    }

    let node_id = cli.node_id.clone().unwrap_or_else(|| {
        hostname::get()
            .ok()
            .and_then(|s| s.into_string().ok())
            .unwrap_or_else(|| "localhost".into())
    });

    // Leaderless clusters: every node (including ones that would be "workers"
    // in the star model) runs the HTTP server path and forms a mesh. Only the
    // legacy star path uses the dedicated worker entrypoint.
    let in_leaderless_cluster = cfg
        .clusters
        .iter()
        .any(|c| c.leaderless && c.nodes.iter().any(|n| n.id == node_id));

    match ai_engine::app::resolve_role(&cfg, &node_id) {
        ai_engine::app::NodeRole::Worker { cluster_id, .. } if !in_leaderless_cluster => {
            tracing::info!(
                node_id = %node_id,
                cluster_id = %cluster_id,
                "starting in worker mode"
            );
            ai_engine::worker_main::run_worker(&cfg, &node_id, &cluster_id).await
        }
        _ => {
            // Gateway/leader: fold any mDNS-discovered Ollamas into the config
            // before wiring providers and routes.
            let discovered_resources =
                ai_engine::discovery::merge_discovered_ollamas(&mut cfg).await;
            let state =
                ai_engine::app::build_app_state(&cfg, &node_id, discovered_resources).await?;
            let listener = tokio::net::TcpListener::bind(&cfg.server.bind).await?;
            let actual = listener.local_addr().ok();
            tracing::info!(bind = ?actual, node_id = %node_id, "ai-engine listening");

            ai_engine::signal::spawn_reload(cli.config.clone(), state.clone());

            let router = ai_engine_http::build_router(state)
                .merge(ai_engine_web::static_router());
            let grace = std::time::Duration::from_secs(cfg.server.shutdown_grace_secs);
            axum::serve(listener, router)
                .with_graceful_shutdown(ai_engine::signal::shutdown_signal(grace))
                .await?;
            Ok(())
        }
    }
}
