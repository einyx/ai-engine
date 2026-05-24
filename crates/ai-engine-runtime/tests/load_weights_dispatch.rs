use ai_engine_runtime::arch::linear::LinearWeight;
use ai_engine_runtime::config::ModelConfig;
use ai_engine_runtime::loader::load_weights;
use std::path::PathBuf;

type B = burn_ndarray::NdArray;

fn safetensors_fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/toy-llama-3")
}

fn gguf_fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/toy-llama-3-gguf")
}

#[test]
fn dispatches_safetensors_path_to_load_range() {
    let cfg = ModelConfig::from_file(&safetensors_fixture().join("config.json")).unwrap();
    let dev = Default::default();
    let weights = load_weights::<B>(
        &safetensors_fixture().join("model.safetensors"),
        &cfg,
        0..cfg.n_layers,
        true,
        true,
        &dev,
    )
    .unwrap();
    // Toy safetensors has dense (bf16) Linears, so expect Dense after load.
    for layer in &weights.layers {
        assert!(matches!(layer.q_proj, LinearWeight::Dense(_)));
    }
}

#[test]
fn dispatches_gguf_path_to_load_gguf() {
    let cfg = ModelConfig::from_file(&gguf_fixture().join("config.json")).unwrap();
    let dev = Default::default();
    let weights = load_weights::<B>(
        &gguf_fixture().join("model.gguf"),
        &cfg,
        0..cfg.n_layers,
        true,
        true,
        &dev,
    )
    .unwrap();
    // Toy GGUF fixture has Q4_0 Linears.
    for layer in &weights.layers {
        assert!(matches!(layer.q_proj, LinearWeight::Q4Gguf(_)));
    }
}

#[test]
fn unknown_extension_errors_clearly() {
    let cfg = ModelConfig::from_file(&safetensors_fixture().join("config.json")).unwrap();
    let dev = Default::default();
    let result = load_weights::<B>(
        std::path::Path::new("/nonexistent/model.bin"),
        &cfg,
        0..1,
        true,
        true,
        &dev,
    );
    let err = match result {
        Ok(_) => panic!("expected error for unsupported extension"),
        Err(e) => e.to_string(),
    };
    assert!(
        err.to_lowercase().contains("unsupported") || err.to_lowercase().contains(".bin"),
        "got error: {err}"
    );
}
