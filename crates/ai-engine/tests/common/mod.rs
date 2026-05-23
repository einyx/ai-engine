//! Shared test harness: spin up an ai-engine bound to 127.0.0.1:0 with a config
//! that points at a caller-provided upstream URL.

use std::net::SocketAddr;
use std::sync::Arc;

use ai_engine_config::Config;

/// Build the test config TOML with one provider of the given kind pointed at `upstream_url`.
pub fn config_for(kind: &str, upstream_url: &str, with_api_key: bool) -> String {
    let api_key_line = if with_api_key {
        "api_key = \"test-key\""
    } else {
        ""
    };
    let provider_id = if kind == "openai" {
        "openai-test"
    } else {
        "anthropic-test"
    };
    let model_glob = if kind == "openai" { "*" } else { "claude-*" };
    let pipeline_route = if kind == "openai" {
        "/v1/chat/completions"
    } else {
        "/v1/messages"
    };
    let extra_pipelines = if kind == "openai" {
        r#"[pipeline."/v1/embeddings"]
stages = ["auth", "model_route", "forward", "log"]
"#
    } else {
        ""
    };
    format!(
        r#"
[server]
bind = "127.0.0.1:0"
log_level = "warn"

[auth]
mode = "passthrough"

[content_policy]
max_request_bytes = 1000000

[[provider]]
id = "{provider_id}"
kind = "{kind}"
base_url = "{upstream_url}"
{api_key_line}
http2 = false

[[route]]
match = {{ model = "{model_glob}" }}
provider = "{provider_id}"

[pipeline."{pipeline_route}"]
stages = ["auth", "model_route", "forward", "log"]
{extra_pipelines}
"#
    )
}

/// Spawn ai-engine in the background, returning its base URL like "http://127.0.0.1:PORT".
pub async fn spawn(cfg_toml: &str) -> String {
    let cfg = Config::from_str(cfg_toml).expect("config parse");
    let state: Arc<ai_engine_http::AppState> =
        ai_engine::app::build_app_state(&cfg, "localhost")
            .await
            .expect("app state");
    let router = ai_engine_http::build_router(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr: SocketAddr = listener.local_addr().expect("local addr");
    tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });
    // Tiny pause to let the server start accepting connections.
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    format!("http://{addr}")
}
