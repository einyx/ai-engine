//! ai-engine

pub mod advertise;
pub mod app;
pub mod discovery;
pub mod signal;
pub mod worker_main;

/// Initialize the global tracing subscriber. Idempotent if it's the first call;
/// subsequent calls (e.g., from tests) are silently ignored.
pub fn init_tracing(server: &ai_engine_config::Server) {
    use tracing_subscriber::{EnvFilter, fmt};
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(server.log_level.clone()));
    let result = match server.log_format.as_str() {
        "json" => fmt()
            .with_env_filter(env_filter)
            .json()
            .with_writer(std::io::stderr)
            .try_init(),
        _ => fmt()
            .with_env_filter(env_filter)
            .pretty()
            .with_writer(std::io::stderr)
            .try_init(),
    };
    if let Err(e) = result {
        eprintln!("tracing already initialized: {e}");
    }
}

/// Initialize tracing for tools that run without a gateway `Server` config
/// (e.g. the Ollama advertiser). Pretty output to stderr, `info` by default.
pub fn init_tracing_default() {
    use tracing_subscriber::{EnvFilter, fmt};
    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = fmt()
        .with_env_filter(env_filter)
        .pretty()
        .with_writer(std::io::stderr)
        .try_init();
}
