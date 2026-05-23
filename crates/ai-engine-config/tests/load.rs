use ai_engine_config::Config;

const MINIMAL: &str = r#"
[server]
bind = "127.0.0.1:0"

[auth]
mode = "passthrough"

[[provider]]
id = "openai-prod"
kind = "openai"
base_url = "https://api.openai.com/v1"
api_key = "sk-x"

[[route]]
match = { model = "gpt-*" }
provider = "openai-prod"

[pipeline."/v1/chat/completions"]
stages = ["forward", "log"]
"#;

#[test]
fn minimal_config_parses() {
    let cfg = Config::from_str(MINIMAL).unwrap();
    assert_eq!(cfg.server.bind, "127.0.0.1:0");
    assert_eq!(cfg.providers.len(), 1);
    assert_eq!(cfg.routes.len(), 1);
}

#[test]
fn env_interpolation_substitutes_known_vars() {
    std::env::set_var("AI_ENGINE_TEST_KEY_OK", "sk-substituted");
    let toml = r#"
[server]
bind = "127.0.0.1:0"

[auth]
mode = "shared-key"
master_keys = [{ key = "${AI_ENGINE_TEST_KEY_OK}", name = "default" }]

[[provider]]
id = "p"
kind = "openai"
base_url = "https://api.openai.com/v1"
api_key = "k"

[[route]]
match = { model = "gpt-*" }
provider = "p"

[pipeline."/v1/chat/completions"]
stages = ["forward", "log"]
"#;
    let cfg = Config::from_str(toml).unwrap();
    assert_eq!(cfg.auth.master_keys[0].key, "sk-substituted");
}

#[test]
fn missing_env_var_is_fatal() {
    let toml = r#"
[server]
bind = "127.0.0.1:0"
[auth]
mode = "shared-key"
master_keys = [{ key = "${AI_ENGINE_DEFINITELY_NOT_SET_ZZZ}", name = "default" }]
[[provider]]
id = "p"
kind = "openai"
base_url = "x"
api_key = "k"
[pipeline."/v1/chat/completions"]
stages = ["forward", "log"]
"#;
    let err = Config::from_str(toml).unwrap_err();
    assert!(err.to_string().to_lowercase().contains("env"));
}

#[test]
fn validation_rejects_pipeline_without_forward() {
    let toml = r#"
[server]
bind = "127.0.0.1:0"
[auth]
mode = "passthrough"
[[provider]]
id = "p"
kind = "openai"
base_url = "x"
api_key = "k"
[pipeline."/v1/chat/completions"]
stages = ["auth", "log"]
"#;
    let err = Config::from_str(toml).unwrap_err();
    assert!(err.to_string().to_lowercase().contains("forward"));
}

#[test]
fn validation_rejects_pipeline_without_terminal() {
    let toml = r#"
[server]
bind = "127.0.0.1:0"
[auth]
mode = "passthrough"
[[provider]]
id = "p"
kind = "openai"
base_url = "x"
api_key = "k"
[pipeline."/v1/chat/completions"]
stages = ["auth", "forward"]
"#;
    let err = Config::from_str(toml).unwrap_err();
    assert!(err.to_string().to_lowercase().contains("terminal"));
}

#[test]
fn validation_rejects_route_with_unknown_provider() {
    let toml = r#"
[server]
bind = "127.0.0.1:0"
[auth]
mode = "passthrough"
[[provider]]
id = "p"
kind = "openai"
base_url = "x"
api_key = "k"
[[route]]
match = { model = "gpt-*" }
provider = "nonexistent"
[pipeline."/v1/chat/completions"]
stages = ["forward", "log"]
"#;
    let err = Config::from_str(toml).unwrap_err();
    assert!(err.to_string().to_lowercase().contains("nonexistent"));
}

#[test]
fn validation_rejects_unknown_provider_kind() {
    let toml = r#"
[server]
bind = "127.0.0.1:0"
[auth]
mode = "passthrough"
[[provider]]
id = "p"
kind = "vertex-magic"
base_url = "x"
api_key = "k"
[pipeline."/v1/chat/completions"]
stages = ["forward", "log"]
"#;
    let err = Config::from_str(toml).unwrap_err();
    assert!(err.to_string().to_lowercase().contains("kind"));
}

#[test]
fn ollama_provider_with_no_api_key_loads_successfully() {
    let toml = r#"
[server]
bind = "127.0.0.1:0"
[auth]
mode = "passthrough"
[[provider]]
id = "ollama-local"
kind = "openai"
base_url = "http://localhost:11434/v1"

[[route]]
match = { model = "llama3*" }
provider = "ollama-local"

[pipeline."/v1/chat/completions"]
stages = ["forward", "log"]
"#;
    let cfg = Config::from_str(toml).unwrap();
    assert_eq!(cfg.providers[0].id, "ollama-local");
    assert!(cfg.providers[0].api_key.is_none());
    assert_eq!(cfg.providers[0].base_url, "http://localhost:11434/v1");
}

#[test]
fn anthropic_kind_with_gpt_route_rejected() {
    // Format-pinning sanity check: route patterns that look obviously
    // wrong against the provider kind are rejected at startup.
    let toml = r#"
[server]
bind = "127.0.0.1:0"
[auth]
mode = "passthrough"
[[provider]]
id = "a"
kind = "anthropic"
base_url = "x"
api_key = "k"
[[route]]
match = { model = "gpt-*" }
provider = "a"
[pipeline."/v1/chat/completions"]
stages = ["forward", "log"]
"#;
    let err = Config::from_str(toml).unwrap_err();
    assert!(err.to_string().to_lowercase().contains("anthropic"));
}

#[test]
fn defaults_applied() {
    let cfg = Config::from_str(MINIMAL).unwrap();
    assert_eq!(cfg.server.shutdown_grace_secs, 30);
    assert_eq!(cfg.server.log_format, "json");
    assert_eq!(cfg.content_policy.max_request_bytes, 1_048_576);
    assert_eq!(cfg.providers[0].timeout_secs, 120);
    assert!(cfg.providers[0].http2);
}
