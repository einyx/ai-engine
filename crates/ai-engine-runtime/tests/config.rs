use ai_engine_runtime::config::{ModelConfig, ModelFamily};

const LLAMA3_8B_CONFIG: &str = r#"{
  "architectures": ["LlamaForCausalLM"],
  "hidden_size": 4096,
  "intermediate_size": 14336,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "vocab_size": 128256,
  "max_position_embeddings": 8192,
  "rope_theta": 500000.0,
  "rms_norm_eps": 1e-5,
  "tie_word_embeddings": false
}"#;

#[test]
fn parses_llama3_hf_config() {
    let cfg = ModelConfig::from_str(LLAMA3_8B_CONFIG).unwrap();
    assert_eq!(cfg.hidden_size, 4096);
    assert_eq!(cfg.n_layers, 32);
    assert_eq!(cfg.n_heads, 32);
    assert_eq!(cfg.n_kv_heads, 8);
    assert_eq!(cfg.head_dim, 128);
    assert_eq!(cfg.family, ModelFamily::Llama3);
}

#[test]
fn rejects_mixtral_with_clear_message() {
    let mixtral = r#"{
      "architectures": ["MixtralForCausalLM"],
      "hidden_size": 4096,
      "num_hidden_layers": 32,
      "num_attention_heads": 32,
      "num_key_value_heads": 8,
      "vocab_size": 32000
    }"#;
    let err = ModelConfig::from_str(mixtral).unwrap_err();
    assert!(err.to_string().to_lowercase().contains("moe not supported"));
}

#[test]
fn computes_head_dim_when_not_in_json() {
    let cfg = ModelConfig::from_str(LLAMA3_8B_CONFIG).unwrap();
    assert_eq!(cfg.head_dim, 4096 / 32);
}
