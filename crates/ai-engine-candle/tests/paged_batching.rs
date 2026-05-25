//! Continuous-batching determinism + throughput smoke.
use std::path::Path;
use std::time::Instant;
use ai_engine_candle::paged::engine::{Engine, EngineConfig, GenRequest};
use ai_engine_candle::device::resolve_device;
use ai_engine_candle::model::read_gguf_meta;
use ai_engine_runtime::load_tokenizer_from_gguf;
use ai_engine_tokenizer::Tokenizer;

const MODEL: &str = "/mnt/4t/cache/gguf-test/qwen2-0_5b-instruct-q4_0.gguf";
const PROMPTS: [&str; 4] = ["Hi.", "Tell me a joke.", "What is Rust?", "Count to three."];
const MAX_TOKENS: usize = 24;

// Run all prompts through ONE engine with max_num_seqs = `cap`. Returns ids per prompt (in submit order).
fn run_engine(path: &Path, prompts: &[&str], cap: usize) -> Vec<Vec<u32>> {
    let device = resolve_device("cpu").unwrap();
    let tok = load_tokenizer_from_gguf(path).unwrap();
    let (eos_id, _, _, _) = read_gguf_meta(path, &tok).unwrap();
    let engine = Engine::spawn(path, device, EngineConfig {
        max_num_seqs: cap, block_size: 16, kv_cache_blocks: 512, max_seq: 4096, eos_token_id: eos_id,
    }).unwrap();
    let mut rxs = Vec::new();
    for p in prompts {
        let ids = tok.encode(p).unwrap();
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Result<u32, String>>();
        engine.submit(GenRequest { prompt_ids: ids, max_tokens: MAX_TOKENS, temperature: 0.0, tx });
        rxs.push(rx);
    }
    let mut out = Vec::new();
    for mut rx in rxs {
        let mut ids = Vec::new();
        while let Some(item) = rx.blocking_recv() { ids.push(item.expect("engine error")); }
        out.push(ids);
    }
    out
}

#[test]
fn batched_matches_solo() {
    let path = Path::new(MODEL);
    if !path.exists() { eprintln!("SKIP: model missing"); return; }
    // Baseline: each prompt alone (cap=1, one prompt per engine).
    let solo: Vec<Vec<u32>> = PROMPTS.iter().map(|p| run_engine(path, &[p], 1).pop().unwrap()).collect();
    // Batched: all four through one engine, cap=4.
    let batched = run_engine(path, &PROMPTS, 4);
    for (i, p) in PROMPTS.iter().enumerate() {
        assert_eq!(solo[i], batched[i], "prompt {i:?} ({p}) batched ids must match solo ids");
    }
}

#[test]
fn throughput_smoke() {
    let path = Path::new(MODEL);
    if !path.exists() { eprintln!("SKIP: model missing"); return; }
    let prompts: Vec<&str> = PROMPTS.iter().cycle().take(8).cloned().collect();
    let t = Instant::now();
    let outs = run_engine(path, &prompts, 8);
    let total: usize = outs.iter().map(|o| o.len()).sum();
    let secs = t.elapsed().as_secs_f64();
    eprintln!("THROUGHPUT: {} tokens in {:.2}s = {:.1} tok/s aggregate (8 concurrent seqs)", total, secs, total as f64 / secs);
    assert!(total > 0, "engine produced no tokens");
}
