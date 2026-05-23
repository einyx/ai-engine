use ai_engine_runtime::arch::model::Model;
use ai_engine_runtime::config::ModelConfig;
use ai_engine_runtime::loader::load_range;
use ai_engine_tokenizer::{HfTokenizer, Tokenizer};
use burn::tensor::{Tensor, Int, TensorData};
use std::path::PathBuf;

type B = burn_ndarray::NdArray;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/toy-llama-3")
}

#[test]
fn forward_matches_reference_logits_within_tolerance() {
    let fix = fixture();
    let cfg = ModelConfig::from_file(&fix.join("config.json")).unwrap();
    let tok = HfTokenizer::from_path(fix.join("tokenizer.json")).unwrap();
    let prompt = std::fs::read_to_string(fix.join("reference_prompt.txt")).unwrap();
    let ids = tok.encode(prompt.trim()).unwrap();
    eprintln!("prompt ids = {:?}", ids);

    let dev = Default::default();
    let weights = load_range::<B>(
        &fix.join("model.safetensors"), &cfg,
        0..cfg.n_layers, true, true, &dev,
    ).unwrap();
    let model = Model::<B>::from_loaded(&cfg, weights, &dev).unwrap();

    let ids_i32: Vec<i32> = ids.iter().map(|x| *x as i32).collect();
    let token_ids = Tensor::<B, 2, Int>::from_data(
        TensorData::new(ids_i32.clone(), [1, ids.len()]),
        &dev,
    );
    let logits = model.forward(token_ids, 0);
    let last_pos_logits: Tensor<B, 1> = logits
        .slice([0..1, (ids.len() - 1)..ids.len(), 0..cfg.vocab_size])
        .reshape([cfg.vocab_size]);

    let ref_bytes = std::fs::read(fix.join("reference_logits.bin")).unwrap();
    let ref_f32: &[f32] = bytemuck::cast_slice(&ref_bytes);
    assert_eq!(ref_f32.len(), cfg.vocab_size, "reference logits length matches vocab");

    let got: Vec<f32> = last_pos_logits.to_data().to_vec().unwrap();
    assert_eq!(got.len(), cfg.vocab_size);

    let mut max_abs_diff = 0.0_f32;
    let mut argmax_us = (0usize, f32::NEG_INFINITY);
    let mut argmax_ref = (0usize, f32::NEG_INFINITY);
    for (i, (a, b)) in got.iter().zip(ref_f32.iter()).enumerate() {
        let d = (a - b).abs();
        if d > max_abs_diff { max_abs_diff = d; }
        if *a > argmax_us.1 { argmax_us = (i, *a); }
        if *b > argmax_ref.1 { argmax_ref = (i, *b); }
        if d >= 1e-3 && i < 16 {
            eprintln!("logit[{i}] diff = {d}  ours={a}  ref={b}");
        }
    }
    eprintln!("max |a-b| = {max_abs_diff}");
    eprintln!("argmax  ours = {} ({})", argmax_us.0, argmax_us.1);
    eprintln!("argmax  ref  = {} ({})", argmax_ref.0, argmax_ref.1);

    // Argmax must agree: this is the strongest semantic correctness signal.
    assert_eq!(
        argmax_us.0, argmax_ref.0,
        "argmax disagreement: ours={} ref={}",
        argmax_us.0, argmax_ref.0
    );

    // The toy fixture stores weights in bf16 (matches real HF Llama checkpoints)
    // but the Python reference script upcasts to f32 before the forward pass,
    // matching what our loader does. So both sides compute in f32 and we observe
    // ~1e-5 max diff in practice — well below the 1e-3 spec gate.
    assert!(
        max_abs_diff < 1e-3,
        "bytes-tolerant gate failed: max |a-b| = {max_abs_diff}"
    );
}
