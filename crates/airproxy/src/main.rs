use clap::Parser;

mod cli;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = cli::Cli::parse();
    let cfg = airproxy_config::Config::load(&cli.config)?;
    airproxy::init_tracing(&cfg.server);

    if cli.check {
        println!("config OK: {}", cli.config.display());
        return Ok(());
    }

    let state = airproxy::app::build_app_state(&cfg)?;
    let listener = tokio::net::TcpListener::bind(&cfg.server.bind).await?;
    let actual = listener.local_addr().ok();
    tracing::info!(bind = ?actual, "airproxy listening");

    airproxy::signal::spawn_reload(cli.config.clone(), state.clone());

    let router = airproxy_http::build_router(state);
    let grace = std::time::Duration::from_secs(cfg.server.shutdown_grace_secs);
    axum::serve(listener, router)
        .with_graceful_shutdown(airproxy::signal::shutdown_signal(grace))
        .await?;
    Ok(())
}
