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
fn parses_cluster_config() {
    let toml = r#"
[server]
bind = "127.0.0.1:0"

[auth]
mode = "passthrough"

[[cluster]]
id = "home"
leader = "node-a"
quic_bind = "0.0.0.0:7700"

[cluster.model]
id = "llama-3-70b"
config_path = "/srv/models/llama-3-70b/config.json"
weights_path = "/srv/models/llama-3-70b"
tokenizer_path = "/srv/models/llama-3-70b/tokenizer.json"

[[cluster.node]]
id = "node-a"
addr = "192.168.1.10:7700"
cert_fingerprint = "sha256:abc123"
backend = "cuda"

[[cluster.node]]
id = "node-b"
addr = "192.168.1.11:7700"
cert_fingerprint = "sha256:def456"
backend = "metal"

[[provider]]
id = "home-cluster"
kind = "local-cluster"
cluster = "home"

[[route]]
match = { model = "llama-3-70b" }
provider = "home-cluster"

[pipeline."/v1/chat/completions"]
stages = ["forward", "log"]
"#;
    let cfg = ai_engine_config::Config::from_str(toml).unwrap();
    assert_eq!(cfg.clusters.len(), 1);
    assert_eq!(cfg.clusters[0].id, "home");
    assert_eq!(cfg.clusters[0].leader, "node-a");
    assert_eq!(cfg.clusters[0].nodes.len(), 2);
    assert_eq!(cfg.clusters[0].model.id, "llama-3-70b");
    assert!(cfg
        .providers
        .iter()
        .any(|p| p.kind == "local-cluster" && p.cluster.as_deref() == Some("home")));
}

#[test]
fn cluster_leader_must_reference_existing_node() {
    let toml = r#"
[server]
bind = "x"
[auth]
mode = "passthrough"
[[cluster]]
id = "c"
leader = "missing-node"
quic_bind = "0.0.0.0:0"
[cluster.model]
id = "m"
config_path = "x"
weights_path = "x"
tokenizer_path = "x"
[[cluster.node]]
id = "node-a"
addr = "127.0.0.1:0"
cert_fingerprint = "sha256:x"
backend = "cpu"
[[provider]]
id = "c-prov"
kind = "local-cluster"
cluster = "c"
[pipeline."/v1/chat/completions"]
stages = ["forward", "log"]
"#;
    let err = ai_engine_config::Config::from_str(toml).unwrap_err();
    assert!(err.to_string().to_lowercase().contains("leader"));
}

#[test]
fn local_cluster_provider_must_reference_existing_cluster() {
    let toml = r#"
[server]
bind = "x"
[auth]
mode = "passthrough"
[[provider]]
id = "orphan"
kind = "local-cluster"
cluster = "does-not-exist"
[pipeline."/v1/chat/completions"]
stages = ["forward", "log"]
"#;
    let err = ai_engine_config::Config::from_str(toml).unwrap_err();
    assert!(err.to_string().to_lowercase().contains("cluster"));
}

#[test]
fn duplicate_cluster_node_ids_rejected() {
    let toml = r#"
[server]
bind = "x"
[auth]
mode = "passthrough"
[[cluster]]
id = "c"
leader = "a"
quic_bind = "0.0.0.0:0"
[cluster.model]
id = "m"
config_path = "x"
weights_path = "x"
tokenizer_path = "x"
[[cluster.node]]
id = "a"
addr = "127.0.0.1:1"
cert_fingerprint = "sha256:x"
backend = "cpu"
[[cluster.node]]
id = "a"
addr = "127.0.0.1:2"
cert_fingerprint = "sha256:y"
backend = "cpu"
[[provider]]
id = "p"
kind = "local-cluster"
cluster = "c"
[pipeline."/v1/chat/completions"]
stages = ["forward", "log"]
"#;
    let err = ai_engine_config::Config::from_str(toml).unwrap_err();
    assert!(err.to_string().to_lowercase().contains("duplicate"));
}

#[test]
fn duplicate_cluster_node_addrs_rejected() {
    let toml = r#"
[server]
bind = "x"
[auth]
mode = "passthrough"
[[cluster]]
id = "c"
leader = "a"
quic_bind = "0.0.0.0:0"
[cluster.model]
id = "m"
config_path = "x"
weights_path = "x"
tokenizer_path = "x"
[[cluster.node]]
id = "a"
addr = "127.0.0.1:1"
cert_fingerprint = "sha256:x"
backend = "cpu"
[[cluster.node]]
id = "b"
addr = "127.0.0.1:1"
cert_fingerprint = "sha256:y"
backend = "cpu"
[[provider]]
id = "p"
kind = "local-cluster"
cluster = "c"
[pipeline."/v1/chat/completions"]
stages = ["forward", "log"]
"#;
    let err = ai_engine_config::Config::from_str(toml).unwrap_err();
    assert!(err.to_string().to_lowercase().contains("addr"));
}

#[test]
fn unknown_backend_kind_rejected() {
    let toml = r#"
[server]
bind = "x"
[auth]
mode = "passthrough"
[[cluster]]
id = "c"
leader = "a"
quic_bind = "0.0.0.0:0"
[cluster.model]
id = "m"
config_path = "x"
weights_path = "x"
tokenizer_path = "x"
[[cluster.node]]
id = "a"
addr = "127.0.0.1:1"
cert_fingerprint = "sha256:x"
backend = "tpu"
[[provider]]
id = "p"
kind = "local-cluster"
cluster = "c"
[pipeline."/v1/chat/completions"]
stages = ["forward", "log"]
"#;
    let err = ai_engine_config::Config::from_str(toml).unwrap_err();
    assert!(err.to_string().to_lowercase().contains("backend"));
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

#[test]
fn parses_cluster_discover_block() {
    let toml = r#"
[server]
bind = "127.0.0.1:0"

[auth]
mode = "passthrough"

[[cluster]]
id = "home"
leader = "node-a"
quic_bind = "0.0.0.0:7700"

[cluster.model]
id = "llama-3-70b"
config_path = "/srv/models/llama-3-70b/config.json"
weights_path = "/srv/models/llama-3-70b/model.safetensors"
tokenizer_path = "/srv/models/llama-3-70b/tokenizer.json"

[cluster.discover]
expected_workers = 2
timeout_secs = 30

[[cluster.node]]
id = "node-a"
addr = "127.0.0.1:7700"
cert_fingerprint = "sha256:abc123"
backend = "cuda"

[[provider]]
id = "home-cluster"
kind = "local-cluster"
cluster = "home"

[[route]]
match = { model = "llama-3-70b" }
provider = "home-cluster"

[pipeline."/v1/chat/completions"]
stages = ["auth", "model_route", "forward", "log"]
"#;
    let cfg = ai_engine_config::Config::from_str(toml).unwrap();
    let cluster = &cfg.clusters[0];
    let disc = cluster.discover.as_ref().expect("cluster.discover present");
    assert_eq!(disc.expected_workers, 2);
    assert_eq!(disc.timeout_secs, 30);
}

#[test]
fn cluster_discover_defaults_timeout() {
    let toml = r#"
[server]
bind = "127.0.0.1:0"
[auth]
mode = "passthrough"

[[cluster]]
id = "home"
leader = "node-a"
quic_bind = "0.0.0.0:7700"

[cluster.model]
id = "m"
config_path = "x"
weights_path = "x"
tokenizer_path = "x"

[cluster.discover]
expected_workers = 3

[[cluster.node]]
id = "node-a"
addr = "127.0.0.1:7700"
cert_fingerprint = "sha256:abc"
backend = "cpu"

[[provider]]
id = "p"
kind = "local-cluster"
cluster = "home"

[pipeline."/v1/chat/completions"]
stages = ["forward", "log"]
"#;
    let cfg = ai_engine_config::Config::from_str(toml).unwrap();
    let disc = cfg.clusters[0].discover.as_ref().unwrap();
    assert_eq!(disc.expected_workers, 3);
    assert_eq!(disc.timeout_secs, 30); // default
}

#[test]
fn cluster_discover_with_zero_workers_rejected() {
    let toml = r#"
[server]
bind = "127.0.0.1:0"
[auth]
mode = "passthrough"

[[cluster]]
id = "home"
leader = "node-a"
quic_bind = "0.0.0.0:7700"

[cluster.model]
id = "m"
config_path = "x"
weights_path = "x"
tokenizer_path = "x"

[cluster.discover]
expected_workers = 0

[[cluster.node]]
id = "node-a"
addr = "127.0.0.1:7700"
cert_fingerprint = "sha256:abc"
backend = "cpu"

[[provider]]
id = "p"
kind = "local-cluster"
cluster = "home"

[pipeline."/v1/chat/completions"]
stages = ["forward", "log"]
"#;
    let err = ai_engine_config::Config::from_str(toml).unwrap_err().to_string();
    assert!(err.to_lowercase().contains("expected_workers"), "got: {err}");
}
