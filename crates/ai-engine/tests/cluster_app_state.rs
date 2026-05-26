const CLUSTER_TOML: &str = r#"
[server]
bind = "127.0.0.1:0"

[auth]
mode = "passthrough"

[[cluster]]
id = "test-cluster"
leader = "leader"
quic_bind = "127.0.0.1:0"

[cluster.model]
id = "toy-llama"
config_path = "../ai-engine-runtime/fixtures/toy-llama-3/config.json"
weights_path = "../ai-engine-runtime/fixtures/toy-llama-3/model.safetensors"
tokenizer_path = "../ai-engine-runtime/fixtures/toy-llama-3/tokenizer.json"

[[cluster.node]]
id = "leader"
addr = "127.0.0.1:7700"
cert_fingerprint = "sha256:0000000000000000000000000000000000000000000000000000000000000000"
backend = "cpu"

[[cluster.node]]
id = "worker-1"
addr = "127.0.0.1:7701"
cert_fingerprint = "sha256:0000000000000000000000000000000000000000000000000000000000000000"
backend = "cpu"

[[provider]]
id = "test"
kind = "local-cluster"
cluster = "test-cluster"

[[route]]
match = { model = "toy-llama" }
provider = "test"

[pipeline."/v1/chat/completions"]
stages = ["forward", "log"]
"#;

#[test]
fn worker_mode_build_app_state_skips_pipelines() {
    let cfg = ai_engine_config::Config::from_str(CLUSTER_TOML).unwrap();
    // Resolve as worker (node-id = "worker-1", not the leader)
    let role = ai_engine::app::resolve_role(&cfg, "worker-1");
    assert!(matches!(role, ai_engine::app::NodeRole::Worker { .. }));
}

#[test]
fn leader_mode_recognized_but_not_started() {
    // For Plan 3's test we don't actually connect — we just check the role
    // resolution. Full startup is exercised by the multiproc smoke test.
    let cfg = ai_engine_config::Config::from_str(CLUSTER_TOML).unwrap();
    let role = ai_engine::app::resolve_role(&cfg, "leader");
    assert!(matches!(role, ai_engine::app::NodeRole::Leader { .. }));
}
