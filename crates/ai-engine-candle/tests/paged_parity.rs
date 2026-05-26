//! Greedy token-parity: paged engine == CandleModel, per arch. THE correctness gate.
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ai_engine_candle::model::{CandleModel, GenParams, read_gguf_meta};
use ai_engine_candle::paged::engine::{Engine, EngineConfig, GenRequest};
use ai_engine_candle::device::resolve_device;
use ai_engine_runtime::load_tokenizer_from_gguf;
use ai_engine_tokenizer::Tokenizer;

const PROMPT: &str = "Hello, who are you?";
const MAX_TOKENS: usize = 40;

fn run_candle_model(path: &Path) -> Vec<u32> {
    let device = resolve_device("cpu").unwrap();
    let tok = Arc::new(load_tokenizer_from_gguf(path).unwrap());
    let mut m = CandleModel::load(path, device, tok).unwrap();
    let params = GenParams { max_tokens: MAX_TOKENS, temperature: 0.0 };
    let mut pt = 0usize;
    m.generate(PROMPT, &params, |_| {}, &mut pt).unwrap()
}

fn run_paged_engine(path: &Path) -> Vec<u32> {
    let device = resolve_device("cpu").unwrap();
    let tok = load_tokenizer_from_gguf(path).unwrap();
    let prompt_ids = tok.encode(PROMPT).unwrap();
    let (eos_id, _ct, _bos, _eos) = read_gguf_meta(path, &tok).unwrap();
    let engine = Engine::spawn(
        path,
        device,
        EngineConfig {
            max_num_seqs: 1,
            block_size: 16,
            kv_cache_blocks: 256,
            max_seq: 4096,
            eos_token_id: eos_id,
        },
    )
    .unwrap();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Result<u32, String>>();
    engine.submit(GenRequest {
        prompt_ids,
        max_tokens: MAX_TOKENS,
        temperature: 0.0,
        tx,
    });
    let mut out = Vec::new();
    while let Some(item) = rx.blocking_recv() {
        out.push(item.expect("engine error"));
    }
    out
}

fn env_path(var: &str) -> Option<PathBuf> {
    std::env::var(var).ok().map(PathBuf::from).filter(|p| p.exists())
}

#[test]
fn parity_qwen2() {
    let path = Path::new("/mnt/4t/cache/gguf-test/qwen2-0_5b-instruct-q4_0.gguf");
    if !path.exists() {
        eprintln!("SKIP qwen2: model missing");
        return;
    }
    let baseline = run_candle_model(path);
    let paged = run_paged_engine(path);
    assert_eq!(baseline, paged, "qwen2 paged ids must match CandleModel ids");
}

#[test]
fn parity_qwen3() {
    let path = Path::new("/mnt/4t/cache/gguf-test/Qwen3-0.6B-Q8_0.gguf");
    if !path.exists() {
        eprintln!("SKIP qwen3: model missing");
        return;
    }
    let baseline = run_candle_model(path);
    let paged = run_paged_engine(path);
    assert_eq!(baseline, paged, "qwen3 paged ids must match CandleModel ids");
}

#[test]
#[ignore = "requires CANDLE_LLAMA_GGUF (no llama model on this box)"]
fn parity_llama() {
    let Some(path) = env_path("CANDLE_LLAMA_GGUF") else {
        eprintln!("SKIP llama");
        return;
    };
    assert_eq!(run_candle_model(&path), run_paged_engine(&path));
}
