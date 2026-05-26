//! Per-layer activation divergence harness: safetensors vs GGUF.
//!
//! Env-gated via:
//!   AI_ENGINE_REAL_GGUF           — path to the GGUF checkpoint
//!   AI_ENGINE_REAL_SAFETENSORS    — path to the safetensors checkpoint
//!
//! Config for the safetensors side:
//!   AI_ENGINE_REAL_ST_CONFIG      — path to config.json (default: same dir as .safetensors)
//!   AI_ENGINE_REAL_ST_TOKENIZER   — path to tokenizer.json (default: same dir as .safetensors)
//!
//! Run with:
//!   AI_ENGINE_REAL_GGUF=/tmp/ai-engine-validation/model.gguf \
//!   AI_ENGINE_REAL_SAFETENSORS=/tmp/ai-engine-validation/safetensors/model.safetensors \
//!     cargo test -p ai-engine-runtime --test divergence_trace -- --ignored --nocapture

use ai_engine_runtime::arch::model::Model;
use ai_engine_runtime::config::ModelConfig;
use ai_engine_runtime::kv_cache::KvCacheSlot;
use ai_engine_runtime::loader::load_weights;
use ai_engine_runtime::tokenizer_gguf::load_tokenizer_from_gguf;
use ai_engine_tokenizer::{HfTokenizer, Tokenizer};
use burn::tensor::{Int, Tensor, TensorData};
use std::path::PathBuf;

type B = burn_ndarray::NdArray;

// ---------------------------------------------------------------------------
// Comparison helpers
// ---------------------------------------------------------------------------

/// Convert a 3-D tensor to a flat f32 vec.
fn to_vec3(t: &Tensor<B, 3>) -> Vec<f32> {
    t.clone().to_data().to_vec::<f32>().unwrap()
}

/// L2 norm of a flat slice.
fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// L2 distance between two flat slices.
fn l2_dist(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// Max absolute difference.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

/// Classify relative L2 distance into MATCHES / DRIFTING / DIVERGED.
fn classify(relative: f32) -> &'static str {
    if relative < 0.05 {
        "MATCHES"
    } else if relative < 0.5 {
        "DRIFTING"
    } else {
        "DIVERGED"
    }
}

/// Core comparison: prints a labelled diagnostic line and returns the
/// relative L2 (L2_dist / L2_norm_of_st) so the caller can track drift.
fn compare(label: &str, st: &Tensor<B, 3>, gg: &Tensor<B, 3>) -> f32 {
    let sv = to_vec3(st);
    let gv = to_vec3(gg);
    let dist = l2_dist(&sv, &gv);
    let norm = l2_norm(&sv);
    let relative = if norm > 1e-9 { dist / norm } else { dist };
    let max_d = max_abs_diff(&sv, &gv);
    let tag = classify(relative);

    // Print first 4 values from each for spot-checking.
    let st_head: Vec<f32> = sv.iter().take(4).copied().collect();
    let gg_head: Vec<f32> = gv.iter().take(4).copied().collect();

    println!(
        "[{tag}] {label}: L2={dist:.4e}  norm={norm:.4e}  rel={relative:.4e}  maxdiff={max_d:.4e}\n  st[0..4]={st_head:.4?}\n  gg[0..4]={gg_head:.4?}"
    );
    relative
}

// ---------------------------------------------------------------------------
// Thin wrapper so we can pass 1-D and 2-D activations through the same helper.
// ---------------------------------------------------------------------------

/// Unsqueeze a [hidden] 1-D to [1,1,hidden] and delegate.
fn compare1(label: &str, st: &Tensor<B, 1>, gg: &Tensor<B, 1>) -> f32 {
    let st3: Tensor<B, 3> = st.clone().reshape([1, 1, st.dims()[0]]);
    let gg3: Tensor<B, 3> = gg.clone().reshape([1, 1, gg.dims()[0]]);
    compare(label, &st3, &gg3)
}

// ---------------------------------------------------------------------------
// The test
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn per_layer_activation_divergence_trace() {
    // -- Environment --------------------------------------------------------
    let gguf_path = match std::env::var("AI_ENGINE_REAL_GGUF") {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            println!("SKIP: AI_ENGINE_REAL_GGUF not set");
            return;
        }
    };
    let st_path = match std::env::var("AI_ENGINE_REAL_SAFETENSORS") {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            println!("SKIP: AI_ENGINE_REAL_SAFETENSORS not set");
            return;
        }
    };

    // Derive config / tokenizer paths relative to safetensors file unless overridden.
    let st_dir = st_path
        .parent()
        .expect("safetensors path has no parent")
        .to_owned();
    let cfg_path = std::env::var("AI_ENGINE_REAL_ST_CONFIG")
        .map(PathBuf::from)
        .unwrap_or_else(|_| st_dir.join("config.json"));
    let tok_path = std::env::var("AI_ENGINE_REAL_ST_TOKENIZER")
        .map(PathBuf::from)
        .unwrap_or_else(|_| st_dir.join("tokenizer.json"));

    // -- Config comparison --------------------------------------------------
    println!("\n=== ModelConfig comparison ===");
    let st_cfg = ModelConfig::from_file(&cfg_path)
        .expect("failed to load safetensors config.json");
    let gg_cfg = ModelConfig::from_gguf_file(&gguf_path)
        .expect("failed to load GGUF ModelConfig");

    println!("safetensors config.json:");
    println!("  n_layers={} hidden={} n_heads={} n_kv_heads={} head_dim={}",
        st_cfg.n_layers, st_cfg.hidden_size, st_cfg.n_heads, st_cfg.n_kv_heads, st_cfg.head_dim);
    println!("  vocab_size={} max_pos={} rope_theta={} rms_eps={} tie_emb={}",
        st_cfg.vocab_size, st_cfg.max_position_embeddings,
        st_cfg.rope_theta, st_cfg.rms_norm_eps, st_cfg.tie_word_embeddings);

    println!("GGUF metadata:");
    println!("  n_layers={} hidden={} n_heads={} n_kv_heads={} head_dim={}",
        gg_cfg.n_layers, gg_cfg.hidden_size, gg_cfg.n_heads, gg_cfg.n_kv_heads, gg_cfg.head_dim);
    println!("  vocab_size={} max_pos={} rope_theta={} rms_eps={} tie_emb={}",
        gg_cfg.vocab_size, gg_cfg.max_position_embeddings,
        gg_cfg.rope_theta, gg_cfg.rms_norm_eps, gg_cfg.tie_word_embeddings);

    let cfg_matches =
        st_cfg.n_layers == gg_cfg.n_layers
        && st_cfg.hidden_size == gg_cfg.hidden_size
        && st_cfg.n_heads == gg_cfg.n_heads
        && st_cfg.n_kv_heads == gg_cfg.n_kv_heads
        && st_cfg.head_dim == gg_cfg.head_dim
        && st_cfg.vocab_size == gg_cfg.vocab_size
        && (st_cfg.rope_theta - gg_cfg.rope_theta).abs() < 1.0
        && (st_cfg.rms_norm_eps - gg_cfg.rms_norm_eps).abs() < 1e-10;
    println!("Config match: {}", if cfg_matches { "YES" } else { "DIFF DETECTED" });

    // Use safetensors config as ground truth (it comes from HF config.json).
    let cfg = st_cfg.clone();
    let n_layers = cfg.n_layers;

    // -- Tokenization -------------------------------------------------------
    println!("\n=== Tokenization: \"Hello\" ===");
    let st_tok = HfTokenizer::from_path(&tok_path)
        .expect("failed to load safetensors tokenizer");
    let gg_tok = load_tokenizer_from_gguf(&gguf_path)
        .expect("failed to load GGUF tokenizer");

    let st_ids = st_tok.encode("Hello").expect("st tokenize");
    let gg_ids = gg_tok.encode("Hello").expect("gg tokenize");
    println!("safetensors ids: {:?}", st_ids);
    println!("GGUF ids:        {:?}", gg_ids);
    if st_ids != gg_ids {
        println!("WARNING: tokenizers produce DIFFERENT ids — analysis proceeds with safetensors ids");
    } else {
        println!("Tokenizers agree: OK");
    }

    // -- Load models --------------------------------------------------------
    println!("\n=== Loading weights (this may take several minutes) ===");
    let dev = Default::default();
    let layer_range = 0..n_layers;

    println!("Loading safetensors...");
    let st_weights = load_weights::<B>(
        &st_path,
        &cfg,
        layer_range.clone(),
        true,
        true,
        &dev,
    )
    .expect("failed to load safetensors weights");

    println!("Loading GGUF...");
    let gg_weights = load_weights::<B>(
        &gguf_path,
        &cfg,
        layer_range.clone(),
        true,
        true,
        &dev,
    )
    .expect("failed to load GGUF weights");

    println!("Building models...");
    let st_model = Model::<B>::from_loaded(&cfg, st_weights, &dev)
        .expect("failed to build safetensors model");
    let gg_model = Model::<B>::from_loaded(&cfg, gg_weights, &dev)
        .expect("failed to build GGUF model");

    // -- Build token_ids tensor ---------------------------------------------
    let ids_to_use = st_ids.clone();
    let ids_i32: Vec<i32> = ids_to_use.iter().map(|x| *x as i32).collect();
    let seq_len = ids_i32.len();
    let token_ids =
        Tensor::<B, 2, Int>::from_data(TensorData::new(ids_i32, [1, seq_len]), &dev);

    let positions: Vec<i32> = (0..seq_len).map(|p| p as i32).collect();

    // -- Allocate KV caches -------------------------------------------------
    let new_cache = |model: &Model<B>| -> Vec<KvCacheSlot<B>> {
        (0..n_layers)
            .map(|_| {
                KvCacheSlot::<B>::new(
                    1,
                    model.n_kv_heads,
                    model.max_seq,
                    model.head_dim,
                    &dev,
                )
            })
            .collect()
    };
    let mut st_caches = new_cache(&st_model);
    let mut gg_caches = new_cache(&gg_model);

    // -----------------------------------------------------------------------
    // Phase A: embedding lookup
    // -----------------------------------------------------------------------
    println!("\n=== Phase A: embedding ===");
    let st_x = st_model.embedding.forward(token_ids.clone());
    let gg_x = gg_model.embedding.forward(token_ids.clone());
    let emb_rel = compare("embedding", &st_x, &gg_x);

    // Embedding row for the first token (spot check).
    let first_tok_id = ids_to_use[0] as usize;
    let st_row: Tensor<B, 1> = st_model
        .embedding
        .weight
        .clone()
        .slice([first_tok_id..first_tok_id + 1, 0..cfg.hidden_size])
        .reshape([cfg.hidden_size]);
    let gg_row: Tensor<B, 1> = gg_model
        .embedding
        .weight
        .clone()
        .slice([first_tok_id..first_tok_id + 1, 0..cfg.hidden_size])
        .reshape([cfg.hidden_size]);
    compare1(&format!("embedding weight row {first_tok_id}"), &st_row, &gg_row);

    // -----------------------------------------------------------------------
    // Phase B: decoder blocks
    // -----------------------------------------------------------------------
    println!("\n=== Phase B: decoder blocks ===");
    let mut st_h = st_x.clone();
    let mut gg_h = gg_x.clone();
    let mut first_diverged: Option<String> = None;

    // Block 0 sub-step drill-down (always run, lets us see exactly where
    // within block 0 divergence first occurs).
    {
        println!("\n--- Block 0 sub-steps ---");
        let blk_st = &st_model.blocks[0];
        let blk_gg = &gg_model.blocks[0];

        // attn_norm
        let st_normed = blk_st.attn_norm.forward(st_h.clone());
        let gg_normed = blk_gg.attn_norm.forward(gg_h.clone());
        compare("block0.attn_norm", &st_normed, &gg_normed);

        // q/k/v projections (raw, before reshape/rope)
        let st_q_raw = blk_st.attn.q_proj.matmul(st_normed.clone());
        let gg_q_raw = blk_gg.attn.q_proj.matmul(gg_normed.clone());
        compare("block0.attn.q_proj(raw)", &st_q_raw, &gg_q_raw);

        let st_k_raw = blk_st.attn.k_proj.matmul(st_normed.clone());
        let gg_k_raw = blk_gg.attn.k_proj.matmul(gg_normed.clone());
        compare("block0.attn.k_proj(raw)", &st_k_raw, &gg_k_raw);

        let st_v_raw = blk_st.attn.v_proj.matmul(st_normed.clone());
        let gg_v_raw = blk_gg.attn.v_proj.matmul(gg_normed.clone());
        compare("block0.attn.v_proj(raw)", &st_v_raw, &gg_v_raw);

        // Full attention output (attn sub-block, residual NOT yet added)
        let st_attn_out = blk_st.attn.forward(st_normed.clone(), &positions, &mut st_caches[0]);
        let gg_attn_out = blk_gg.attn.forward(gg_normed.clone(), &positions, &mut gg_caches[0]);
        compare("block0.attn_out", &st_attn_out, &gg_attn_out);

        // After residual 1
        let st_after_r1 = st_h.clone().add(st_attn_out);
        let gg_after_r1 = gg_h.clone().add(gg_attn_out);
        compare("block0.after_residual1", &st_after_r1, &gg_after_r1);

        // ffn_norm
        let st_ffn_normed = blk_st.ffn_norm.forward(st_after_r1.clone());
        let gg_ffn_normed = blk_gg.ffn_norm.forward(gg_after_r1.clone());
        compare("block0.ffn_norm", &st_ffn_normed, &gg_ffn_normed);

        // ffn gate/up projections (before silu/mul)
        let st_gate_raw = blk_st.ffn.gate_proj.matmul(st_ffn_normed.clone());
        let gg_gate_raw = blk_gg.ffn.gate_proj.matmul(gg_ffn_normed.clone());
        compare("block0.ffn.gate_proj(raw)", &st_gate_raw, &gg_gate_raw);

        let st_up_raw = blk_st.ffn.up_proj.matmul(st_ffn_normed.clone());
        let gg_up_raw = blk_gg.ffn.up_proj.matmul(gg_ffn_normed.clone());
        compare("block0.ffn.up_proj(raw)", &st_up_raw, &gg_up_raw);

        // Full ffn output
        let st_ffn_out = blk_st.ffn.forward(st_ffn_normed.clone());
        let gg_ffn_out = blk_gg.ffn.forward(gg_ffn_normed.clone());
        compare("block0.ffn_out", &st_ffn_out, &gg_ffn_out);

        // Block 0 full output (residual 2 applied)
        // NOTE: caches for block 0 were already consumed by attn.forward above,
        // so we reuse the half-computed values rather than calling block.forward
        // (which would double-write the cache). We reconstruct from saved tensors.
        let st_blk0_out = st_after_r1.clone().add(st_ffn_out);
        let gg_blk0_out = gg_after_r1.clone().add(gg_ffn_out);
        let b0_rel = compare("block0.output", &st_blk0_out, &gg_blk0_out);
        if first_diverged.is_none() && classify(b0_rel) == "DIVERGED" {
            first_diverged = Some("block0.output".to_string());
        }

        // Advance h pointers for the remaining block loop (start from block 1).
        st_h = st_blk0_out;
        gg_h = gg_blk0_out;
    }

    // Blocks 1..n_layers: full block.forward (caches already pre-allocated; block 0 cache
    // was filled in the sub-step drill above so we skip it here).
    println!("\n--- Blocks 1..{n_layers} summary ---");
    for i in 1..n_layers {
        st_h = st_model.blocks[i].forward(st_h, &positions, &mut st_caches[i]);
        gg_h = gg_model.blocks[i].forward(gg_h, &positions, &mut gg_caches[i]);
        let rel = compare(&format!("block{i}.output"), &st_h, &gg_h);
        if first_diverged.is_none() && classify(rel) == "DIVERGED" {
            first_diverged = Some(format!("block{i}.output"));
        }
    }

    // -----------------------------------------------------------------------
    // Phase C: final_norm + output projection
    // -----------------------------------------------------------------------
    println!("\n=== Phase C: final_norm + logits ===");
    let st_normed_final = st_model.final_norm.forward(st_h);
    let gg_normed_final = gg_model.final_norm.forward(gg_h);
    let fn_rel = compare("final_norm", &st_normed_final, &gg_normed_final);
    if first_diverged.is_none() && classify(fn_rel) == "DIVERGED" {
        first_diverged = Some("final_norm".to_string());
    }

    let st_logits = st_model.output.forward(st_normed_final);
    let gg_logits = gg_model.output.forward(gg_normed_final);
    let logits_rel = compare("logits", &st_logits, &gg_logits);
    if first_diverged.is_none() && classify(logits_rel) == "DIVERGED" {
        first_diverged = Some("logits".to_string());
    }

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    println!("\n=== SUMMARY ===");
    println!("embedding relative L2: {emb_rel:.4e}  {}", classify(emb_rel));
    println!("first DIVERGED step: {}", first_diverged.as_deref().unwrap_or("none — all MATCHES/DRIFTING"));
    println!("logits relative L2:    {logits_rel:.4e}  {}", classify(logits_rel));
}
