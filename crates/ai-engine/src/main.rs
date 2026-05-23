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

    let state = ai_engine::app::build_app_state(&cfg, "localhost").await?;
    let listener = tokio::net::TcpListener::bind(&cfg.server.bind).await?;
    let actual = listener.local_addr().ok();
    tracing::info!(bind = ?actual, "ai-engine listening");

    ai_engine::signal::spawn_reload(cli.config.clone(), state.clone());

    let router = ai_engine_http::build_router(state);
    let grace = std::time::Duration::from_secs(cfg.server.shutdown_grace_secs);
    axum::serve(listener, router)
        .with_graceful_shutdown(ai_engine::signal::shutdown_signal(grace))
        .await?;
    Ok(())
}
