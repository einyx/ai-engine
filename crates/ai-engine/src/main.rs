use clap::Parser;

mod cli;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = cli::Cli::parse();
    let cfg = ai_engine_config::Config::load(&cli.config)?;
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

    match ai_engine::app::resolve_role(&cfg, &node_id) {
        ai_engine::app::NodeRole::Worker { cluster_id, .. } => {
            tracing::info!(
                node_id = %node_id,
                cluster_id = %cluster_id,
                "starting in worker mode"
            );
            ai_engine::worker_main::run_worker(&cfg, &node_id, &cluster_id).await
        }
        _ => {
            let state = ai_engine::app::build_app_state(&cfg, &node_id).await?;
            let listener = tokio::net::TcpListener::bind(&cfg.server.bind).await?;
            let actual = listener.local_addr().ok();
            tracing::info!(bind = ?actual, node_id = %node_id, "ai-engine listening");

            ai_engine::signal::spawn_reload(cli.config.clone(), state.clone());

            let router = ai_engine_http::build_router(state);
            let grace = std::time::Duration::from_secs(cfg.server.shutdown_grace_secs);
            axum::serve(listener, router)
                .with_graceful_shutdown(ai_engine::signal::shutdown_signal(grace))
                .await?;
            Ok(())
        }
    }
}
