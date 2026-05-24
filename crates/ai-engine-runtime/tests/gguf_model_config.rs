use ai_engine_runtime::config::{ModelConfig, ModelFamily};
use ai_engine_runtime::gguf;
use std::path::PathBuf;

fn gguf_fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/toy-llama-3-gguf")
}

#[test]
fn read_model_config_from_gguf_matches_safetensors_config() {
    let from_gguf = ModelConfig::from_gguf_file(&gguf_fixture().join("model.gguf")).unwrap();
    let from_json = ModelConfig::from_file(&gguf_fixture().join("config.json")).unwrap();

    // All architectural hyperparams must match the bf16 fixture's config.json.
    assert_eq!(from_gguf.hidden_size, from_json.hidden_size);
    assert_eq!(from_gguf.n_layers, from_json.n_layers);
    assert_eq!(from_gguf.n_heads, from_json.n_heads);
    assert_eq!(from_gguf.n_kv_heads, from_json.n_kv_heads);
    assert_eq!(from_gguf.intermediate_size, from_json.intermediate_size);
    assert_eq!(from_gguf.max_position_embeddings, from_json.max_position_embeddings);
    assert_eq!(from_gguf.head_dim, from_json.head_dim);
    assert!((from_gguf.rope_theta - from_json.rope_theta).abs() < 1e-3);
    assert!((from_gguf.rms_norm_eps - from_json.rms_norm_eps).abs() < 1e-7);
    assert_eq!(from_gguf.vocab_size, from_json.vocab_size);
    assert_eq!(from_gguf.family, ModelFamily::Llama3);
}

#[test]
fn read_metadata_only_skips_tensor_data() {
    // Just verifies the function exists and parses without OOM on a real file.
    let m = gguf::read_metadata_only(&gguf_fixture().join("model.gguf")).unwrap();
    assert!(m.contains_key("general.architecture"));
    assert!(m.contains_key("llama.block_count"));
}
