use ai_engine_runtime::arch::model::Model;
use ai_engine_runtime::config::ModelConfig;
use ai_engine_runtime::loader::load_range;
use burn::tensor::{Tensor, Int, TensorData};
use std::path::PathBuf;

type B = burn_ndarray::NdArray;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/toy-llama-3")
}

#[test]
fn model_loads_from_toy_fixture_and_runs_forward() {
    let cfg = ModelConfig::from_file(&fixture_path().join("config.json")).unwrap();
    let dev = Default::default();
    let weights = load_range::<B>(
        &fixture_path().join("model.safetensors"),
        &cfg,
        0..cfg.n_layers,
        true, true,
        &dev,
    ).unwrap();
    let model = Model::<B>::from_loaded(&cfg, weights, &dev).unwrap();
    // Smoke: forward pass with 4 tokens.
    let ids = Tensor::<B, 2, Int>::from_data(
        TensorData::new(vec![10_i32, 20, 30, 40], [1, 4]),
        &dev,
    );
    let logits = model.forward(ids, 0);
    assert_eq!(logits.dims(), [1, 4, cfg.vocab_size]);
}
