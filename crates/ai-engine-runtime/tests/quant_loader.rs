use ai_engine_runtime::arch::linear::LinearWeight;
use ai_engine_runtime::config::ModelConfig;
use ai_engine_runtime::loader::load_range;
use std::path::PathBuf;

type B = burn_ndarray::NdArray;

fn q8_fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/toy-llama-3-q8")
}

#[test]
fn load_q8_fixture_produces_quantized_weights() {
    let cfg = ModelConfig::from_file(&q8_fixture_path().join("config.json")).unwrap();
    let dev = Default::default();
    let weights = load_range::<B>(
        &q8_fixture_path().join("model.safetensors"),
        &cfg,
        0..cfg.n_layers,
        true,
        true,
        &dev,
    )
    .unwrap();
    // Each layer's Linear weights should be the Quantized variant.
    for layer in &weights.layers {
        assert!(matches!(layer.q_proj, LinearWeight::Quantized(_)));
        assert!(matches!(layer.k_proj, LinearWeight::Quantized(_)));
        assert!(matches!(layer.v_proj, LinearWeight::Quantized(_)));
        assert!(matches!(layer.o_proj, LinearWeight::Quantized(_)));
        assert!(matches!(layer.ffn_gate, LinearWeight::Quantized(_)));
        assert!(matches!(layer.ffn_up, LinearWeight::Quantized(_)));
        assert!(matches!(layer.ffn_down, LinearWeight::Quantized(_)));
    }
    // Embedding stays dense (we didn't quantize it).
    assert!(weights.embedding.is_some());
    // output_proj is None because the toy uses tie_word_embeddings.
    assert!(weights.output_proj.is_none());
}

#[test]
fn load_bf16_fixture_produces_dense_weights() {
    // The original bf16 fixture should still load through the same loader,
    // taking the Dense path for each Linear weight.
    let fix = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/toy-llama-3");
    let cfg = ModelConfig::from_file(&fix.join("config.json")).unwrap();
    let dev = Default::default();
    let weights = load_range::<B>(
        &fix.join("model.safetensors"),
        &cfg,
        0..cfg.n_layers,
        true,
        true,
        &dev,
    )
    .unwrap();
    for layer in &weights.layers {
        assert!(matches!(layer.q_proj, LinearWeight::Dense(_)));
        assert!(matches!(layer.ffn_down, LinearWeight::Dense(_)));
    }
}
