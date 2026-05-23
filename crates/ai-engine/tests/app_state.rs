use ai_engine_config::Config;

const FULL_TOML: &str = r#"
[server]
bind = "127.0.0.1:0"
log_level = "info"

[auth]
mode = "shared-key"
master_keys = [{ key = "k", name = "default" }]

[content_policy]
max_request_bytes = 100000
prompt_injection_patterns = ["bad"]

[[provider]]
id = "openai"
kind = "openai"
base_url = "https://api.openai.com/v1"
api_key = "sk-test"

[[provider]]
id = "anthropic"
kind = "anthropic"
base_url = "https://api.anthropic.com"
api_key = "sk-ant"

[[provider]]
id = "ollama"
kind = "openai"
base_url = "http://localhost:11434/v1"

[[route]]
match = { model = "gpt-*" }
provider = "openai"

[[route]]
match = { model = "claude-*" }
provider = "anthropic"

[[route]]
match = { model = "llama*" }
provider = "ollama"

[pipeline."/v1/chat/completions"]
stages = ["auth", "content_policy", "model_route", "forward", "log"]

[pipeline."/v1/messages"]
stages = ["auth", "content_policy", "model_route", "forward", "log"]
"#;

#[tokio::test]
async fn build_app_state_populates_pipelines_and_models() {
    let cfg = Config::from_str(FULL_TOML).unwrap();
    let state = ai_engine::app::build_app_state(&cfg, "anywhere")
        .await
        .unwrap();
    assert!(state.pipelines.contains_key("/v1/chat/completions"));
    assert!(state.pipelines.contains_key("/v1/messages"));
    assert!(!state.pipelines.contains_key("/v1/embeddings")); // no [pipeline] block for it
    assert_eq!(state.openai_models.len(), 3);
    assert!(state.ready.load(std::sync::atomic::Ordering::Relaxed));
}

#[tokio::test]
async fn build_app_state_works_with_ollama_no_api_key() {
    let cfg = Config::from_str(FULL_TOML).unwrap();
    let _state = ai_engine::app::build_app_state(&cfg, "anywhere")
        .await
        .unwrap();
    // If we got here, the no-api-key Ollama provider didn't trip provider construction.
}
