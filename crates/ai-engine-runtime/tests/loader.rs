use ai_engine_runtime::config::ModelConfig;
use ai_engine_runtime::loader::{load_range, LoadedWeights};
use std::path::PathBuf;

type B = burn_ndarray::NdArray;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/toy-llama-3")
}

#[test]
fn load_full_model_for_single_node() {
    let cfg = ModelConfig::from_file(&fixture_path().join("config.json")).unwrap();
    let dev = Default::default();
    let weights: LoadedWeights<B> = load_range(
        &fixture_path().join("model.safetensors"),
        &cfg,
        0..cfg.n_layers,
        /*hosts_embedding=*/ true,
        /*hosts_output=*/ true,
        &dev,
    )
    .unwrap();
    assert!(weights.embedding.is_some());
    assert!(weights.final_norm.is_some());
    assert_eq!(weights.layers.len(), cfg.n_layers);
    // toy fixture has tie_word_embeddings=true, so output_proj should be None.
    assert!(weights.output_proj.is_none());
}

#[test]
fn load_layer_range_for_worker_node() {
    let cfg = ModelConfig::from_file(&fixture_path().join("config.json")).unwrap();
    let dev = Default::default();
    let weights: LoadedWeights<B> = load_range(
        &fixture_path().join("model.safetensors"),
        &cfg,
        1..3,
        false,
        false,
        &dev,
    )
    .unwrap();
    assert!(weights.embedding.is_none());
    assert!(weights.final_norm.is_none());
    assert!(weights.output_proj.is_none());
    assert_eq!(weights.layers.len(), 2);
}
