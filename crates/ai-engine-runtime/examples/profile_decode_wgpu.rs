//! GPU profile harness: real-model GGUF decode loop on wgpu backend.
//!
//! Mirrors `profile_decode.rs` (CPU/ndarray) but uses `burn_wgpu::Wgpu`.
//! Set RUST_LOG=info,wgpu_hal=info,wgpu_core=info to see which adapter wgpu
//! selects (look for "NVIDIA" / "RTX 4070"; "llvmpipe" means software fallback).
//!
//!   AI_ENGINE_REAL_GGUF=/tmp/ai-engine-validation/model.gguf \
//!     RUST_LOG=info,wgpu_hal=info,wgpu_core=info \
//!     ./target/release/examples/profile_decode_wgpu

use ai_engine_runtime::arch::model::Model;
use ai_engine_runtime::config::ModelConfig;
use ai_engine_runtime::kv_cache::KvCacheSlot;
use ai_engine_runtime::loader::load_weights;
use ai_engine_runtime::sample::{sample, SamplingConfig};
use ai_engine_runtime::tokenizer_gguf::load_tokenizer_from_gguf;
use ai_engine_tokenizer::Tokenizer;
use burn_wgpu::WgpuDevice;
use burn::tensor::{Int, Tensor, TensorData};
use std::time::Instant;

type B = burn_wgpu::Wgpu;

const DECODE_STEPS: usize = 10;
const DEFAULT_GGUF: &str = "/tmp/ai-engine-validation/model.gguf";
const PROMPT: &str = "Hello, who are you?";

fn main() -> anyhow::Result<()> {
    let gguf_path = std::env::var("AI_ENGINE_REAL_GGUF")
        .unwrap_or_else(|_| DEFAULT_GGUF.to_string());
    let path = std::path::Path::new(&gguf_path);

    // ── Load phase ────────────────────────────────────────────────────────────
    eprintln!("loading model from {gguf_path} ...");
    let t_load = Instant::now();

    let cfg = ModelConfig::from_gguf_file(path)?;
    let tokenizer = load_tokenizer_from_gguf(path)?;

    let dev = WgpuDevice::default();
    let weights = load_weights::<B>(path, &cfg, 0..cfg.n_layers, true, true, &dev)?;
    let model = Model::<B>::from_loaded(&cfg, weights, &dev)?;

    eprintln!("model loaded in {:.1}s", t_load.elapsed().as_secs_f32());

    // ── Tokenise prompt ───────────────────────────────────────────────────────
    let prompt_ids: Vec<u32> = tokenizer.encode(PROMPT)?;
    let n_prompt = prompt_ids.len();
    eprintln!("prompt tokens: {n_prompt}");

    // ── Pre-allocate KV caches (batch=1, all layers) ──────────────────────────
    let mut caches: Vec<KvCacheSlot<B>> = (0..cfg.n_layers)
        .map(|_| {
            KvCacheSlot::<B>::new(
                1,
                cfg.n_kv_heads,
                cfg.max_position_embeddings,
                cfg.head_dim,
                &dev,
            )
        })
        .collect();

    // temperature=0.0 → greedy argmax (deterministic)
    let sampling = SamplingConfig {
        temperature: 0.0,
        top_p: None,
        top_k: None,
        seed: 42,
    };

    // ── Prefill ───────────────────────────────────────────────────────────────
    let prefill_ids_i32: Vec<i32> = prompt_ids.iter().map(|&x| x as i32).collect();
    let prefill_tensor = Tensor::<B, 2, Int>::from_data(
        TensorData::new(prefill_ids_i32, [1, n_prompt]),
        &dev,
    );
    let prefill_logits = model.forward_with_caches(prefill_tensor, 0, &mut caches);
    let last_logits: Vec<f32> = prefill_logits
        .slice([0..1, (n_prompt - 1)..n_prompt, 0..cfg.vocab_size])
        .reshape([cfg.vocab_size])
        .to_data()
        .to_vec()?;
    let mut next_token = sample(&last_logits, &sampling);
    let mut start_pos = n_prompt;

    eprintln!("prefill done. starting {DECODE_STEPS} decode steps ...");

    // ── Decode loop ───────────────────────────────────────────────────────────
    let mut step_times = Vec::with_capacity(DECODE_STEPS);
    let mut decoded_tokens = Vec::with_capacity(DECODE_STEPS);

    for step in 0..DECODE_STEPS {
        let t0 = Instant::now();

        let tok_tensor = Tensor::<B, 2, Int>::from_data(
            TensorData::new(vec![next_token as i32], [1, 1]),
            &dev,
        );
        let logits = model.forward_with_caches(tok_tensor, start_pos, &mut caches);
        let logit_vec: Vec<f32> = logits
            .reshape([cfg.vocab_size])
            .to_data()
            .to_vec()?;
        next_token = sample(&logit_vec, &sampling);

        let elapsed = t0.elapsed();
        step_times.push(elapsed);
        decoded_tokens.push(next_token);
        eprintln!(
            "step {:2}: token={:5}  {:.3}s",
            step,
            next_token,
            elapsed.as_secs_f32()
        );
        start_pos += 1;
    }

    // ── Report ────────────────────────────────────────────────────────────────
    let total_secs_all: f64 = step_times.iter().map(|d| d.as_secs_f64()).sum();
    let tok_per_sec_all = DECODE_STEPS as f64 / total_secs_all;

    // Exclude step 0 (shader-compile cold-start) for steady-state rate.
    let steady_secs: f64 = step_times[1..].iter().map(|d| d.as_secs_f64()).sum();
    let steady_steps = DECODE_STEPS - 1;
    let tok_per_sec_steady = steady_steps as f64 / steady_secs;

    eprintln!(
        "\n{DECODE_STEPS} tokens in {total_secs_all:.2}s ({tok_per_sec_all:.3} tok/s incl. cold start)"
    );
    eprintln!(
        "steps 1-{}: {steady_secs:.3}s  =>  {tok_per_sec_steady:.3} tok/s  (steady-state, cold-start excluded)",
        DECODE_STEPS - 1
    );

    // Decode the output tokens to text for coherence check.
    if let Ok(text) = tokenizer.decode(&decoded_tokens) {
        eprintln!("\ngreedy decode of \"{PROMPT}\":\n  {text:?}");
    }

    Ok(())
}
