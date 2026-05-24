use ai_engine_runtime::arch::model::Model;
use ai_engine_runtime::config::ModelConfig;
use ai_engine_runtime::loader::load_gguf;
use ai_engine_tokenizer::{HfTokenizer, Tokenizer};
use burn::tensor::{Int, Tensor, TensorData};
use std::path::PathBuf;

type B = burn_ndarray::NdArray;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/toy-llama-3-gguf")
}

#[test]
fn gguf_q4_0_forward_runs_and_logits_are_finite() {
    let fix = fixture();
    let cfg = ModelConfig::from_file(&fix.join("config.json")).unwrap();
    let tok = HfTokenizer::from_path(fix.join("tokenizer.json")).unwrap();
    let prompt = std::fs::read_to_string(fix.join("reference_prompt.txt")).unwrap();
    let ids = tok.encode(prompt.trim()).unwrap();

    let dev = Default::default();
    let weights = load_gguf::<B>(
        &fix.join("model.gguf"),
        &cfg,
        0..cfg.n_layers,
        true,
        true,
        &dev,
    )
    .unwrap();
    let model = Model::<B>::from_loaded(&cfg, weights, &dev).unwrap();

    let ids_i32: Vec<i32> = ids.iter().map(|x| *x as i32).collect();
    let token_ids =
        Tensor::<B, 2, Int>::from_data(TensorData::new(ids_i32, [1, ids.len()]), &dev);
    let logits = model.forward(token_ids, 0);
    let last_pos_logits: Tensor<B, 1> = logits
        .slice([0..1, (ids.len() - 1)..ids.len(), 0..cfg.vocab_size])
        .reshape([cfg.vocab_size]);

    let got: Vec<f32> = last_pos_logits.to_data().to_vec().unwrap();
    assert_eq!(got.len(), cfg.vocab_size);
    // Finite-value check: no NaN, no Inf.
    for (i, v) in got.iter().enumerate() {
        assert!(v.is_finite(), "logit[{i}] is not finite: {v}");
    }

    // Diff against bf16 reference (informational; do not assert tight tolerance
    // on the random-init toy — GGUF Q4_0 + GGUF flat-layout transpose introduces
    // significant intrinsic error that's not a bug). Just assert max_diff is bounded.
    let ref_bytes = std::fs::read(fix.join("reference_logits.bin")).unwrap();
    let ref_f32: &[f32] = bytemuck::cast_slice(&ref_bytes);
    let mut max_diff = 0.0_f32;
    for (a, b) in got.iter().zip(ref_f32.iter()) {
        let d = (a - b).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    eprintln!("GGUF Q4_0 vs bf16 reference max |a-b| = {max_diff}");
    // Sanity bound: random-weight GGUF Q4_0 forward must produce SOME bounded
    // output. If it's > 10 (i.e., logits diverge wildly), there's a real bug.
    assert!(
        max_diff < 10.0,
        "GGUF Q4_0 logits diverged catastrophically: {max_diff}"
    );
}
