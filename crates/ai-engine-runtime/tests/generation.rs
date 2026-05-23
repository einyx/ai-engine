use ai_engine_runtime::arch::model::Model;
use ai_engine_runtime::config::ModelConfig;
use ai_engine_runtime::kv_cache::KvCacheSlot;
use ai_engine_runtime::loader::load_range;
use ai_engine_tokenizer::{HfTokenizer, Tokenizer};
use burn::tensor::{Tensor, Int, TensorData};
use std::path::PathBuf;

type B = burn_ndarray::NdArray;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/toy-llama-3")
}

#[test]
fn cached_generation_matches_fresh_full_forward() {
    let fix = fixture();
    let cfg = ModelConfig::from_file(&fix.join("config.json")).unwrap();
    let tok = HfTokenizer::from_path(fix.join("tokenizer.json")).unwrap();
    let dev = Default::default();
    let weights = load_range::<B>(
        &fix.join("model.safetensors"), &cfg, 0..cfg.n_layers, true, true, &dev,
    ).unwrap();
    let model = Model::<B>::from_loaded(&cfg, weights, &dev).unwrap();

    let prompt = "The quick brown fox";
    let prompt_ids: Vec<u32> = tok.encode(prompt).unwrap();
    let n = prompt_ids.len();

    // Path A: feed all tokens at once (prefill), read final-position logits.
    let prefill = Tensor::<B, 2, Int>::from_data(
        TensorData::new(
            prompt_ids.iter().map(|x| *x as i32).collect::<Vec<_>>(),
            [1, n],
        ),
        &dev,
    );
    let logits_a = model.forward(prefill, 0);
    let last_a: Vec<f32> = logits_a
        .slice([0..1, (n-1)..n, 0..cfg.vocab_size])
        .reshape([cfg.vocab_size])
        .to_data().to_vec().unwrap();

    // Path B: feed tokens one-at-a-time, reusing the SAME caches across steps.
    let mut caches: Vec<KvCacheSlot<B>> = (0..cfg.n_layers)
        .map(|_| KvCacheSlot::<B>::new(1, cfg.n_kv_heads, cfg.max_position_embeddings, cfg.head_dim, &dev))
        .collect();
    let mut last_b: Vec<f32> = vec![];
    for (i, tid) in prompt_ids.iter().enumerate() {
        let t = Tensor::<B, 2, Int>::from_data(
            TensorData::new(vec![*tid as i32], [1, 1]),
            &dev,
        );
        let logits = model.forward_with_caches(t, i, &mut caches);
        last_b = logits.reshape([cfg.vocab_size]).to_data().to_vec().unwrap();
    }

    let mut max_diff: f32 = 0.0;
    for (a, b) in last_a.iter().zip(last_b.iter()) {
        let d = (a - b).abs();
        if d > max_diff { max_diff = d; }
    }
    eprintln!("max |a-b| = {max_diff}");
    assert!(
        max_diff < 1e-4,
        "cached and fresh logits should match within f32 tolerance, diff = {max_diff}"
    );
}
