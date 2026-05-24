//! Verifies that the safetensors loader detects the Q4 companion-tensor format
//! (`<name>.q4_weight` packed bytes + `<name>.q4_scale` per-group scales) and
//! materializes each Linear weight as `LinearWeight::Q4(_)`.

use ai_engine_runtime::arch::linear::LinearWeight;
use ai_engine_runtime::config::ModelConfig;
use ai_engine_runtime::loader::load_range;
use std::path::PathBuf;

type B = burn_ndarray::NdArray;

fn q4_fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/toy-llama-3-q4")
}

#[test]
fn load_q4_fixture_produces_q4_weights() {
    let cfg = ModelConfig::from_file(&q4_fixture().join("config.json")).unwrap();
    let dev = Default::default();
    let weights = load_range::<B>(
        &q4_fixture().join("model.safetensors"),
        &cfg,
        0..cfg.n_layers,
        true,
        true,
        &dev,
    )
    .unwrap();
    assert_eq!(weights.layers.len(), cfg.n_layers);
    for layer in &weights.layers {
        assert!(matches!(layer.q_proj, LinearWeight::Q4(_)));
        assert!(matches!(layer.k_proj, LinearWeight::Q4(_)));
        assert!(matches!(layer.v_proj, LinearWeight::Q4(_)));
        assert!(matches!(layer.o_proj, LinearWeight::Q4(_)));
        assert!(matches!(layer.ffn_gate, LinearWeight::Q4(_)));
        assert!(matches!(layer.ffn_up, LinearWeight::Q4(_)));
        assert!(matches!(layer.ffn_down, LinearWeight::Q4(_)));
    }
    // Embedding stays dense at runtime — for tied-embedding fixtures the
    // loader dequantizes the Q4 lm_head into a Dense Tensor here.
    assert!(weights.embedding.is_some());
}
