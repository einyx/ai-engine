//! Cross-backend activation divergence harness: NdArray (CPU, known-good) vs Wgpu.
//!
//! Loads the toy-llama-3-gguf fixture on both backends and compares activations
//! stage by stage to find the first divergence point.
//!
//! Override fixture: AI_ENGINE_DIVERGENCE_GGUF=/path/to/model.gguf
//!
//! Run:
//!   cargo test -p ai-engine-runtime --test backend_divergence \
//!     --features backend-wgpu -- --ignored --nocapture

#[cfg(all(feature = "backend-cpu", feature = "backend-wgpu"))]
mod divergence {
    use ai_engine_runtime::arch::model::Model;
    use ai_engine_runtime::config::ModelConfig;
    use ai_engine_runtime::kv_cache::KvCacheSlot;
    use ai_engine_runtime::loader::load_weights;
    use burn::tensor::{Int, Tensor, TensorData, activation::softmax};
    use std::path::PathBuf;

    type Cpu = burn_ndarray::NdArray;
    type Gpu = burn_wgpu::Wgpu;

    // -----------------------------------------------------------------------
    // Comparison helpers — work on raw Vec<f32>
    // -----------------------------------------------------------------------

    fn l2_norm(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    fn l2_dist(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt()
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max)
    }

    fn classify(rel: f32) -> &'static str {
        if rel < 0.05 { "MATCHES" } else if rel < 0.5 { "DRIFTING" } else { "DIVERGED" }
    }

    fn compare_vecs(label: &str, cv: &[f32], gv: &[f32]) -> f32 {
        assert_eq!(cv.len(), gv.len(), "{label}: element count mismatch");
        let dist = l2_dist(cv, gv);
        let norm = l2_norm(cv);
        let rel = if norm > 1e-9 { dist / norm } else { dist };
        let max_d = max_abs_diff(cv, gv);
        let tag = classify(rel);
        let ch: Vec<f32> = cv.iter().take(4).copied().collect();
        let gh: Vec<f32> = gv.iter().take(4).copied().collect();
        println!("[{tag}] {label}: L2={dist:.4e} norm={norm:.4e} rel={rel:.4e} maxdiff={max_d:.4e}");
        println!("  cpu[0..4]={ch:.4?}");
        println!("  gpu[0..4]={gh:.4?}");
        rel
    }

    fn cpu3_to_vec(t: &Tensor<Cpu, 3>) -> Vec<f32> {
        t.clone().into_data().to_vec::<f32>().unwrap()
    }

    fn gpu3_to_vec(t: &Tensor<Gpu, 3>) -> Vec<f32> {
        t.clone().into_data().to_vec::<f32>().unwrap()
    }

    fn cmp3(label: &str, c: &Tensor<Cpu, 3>, g: &Tensor<Gpu, 3>) -> f32 {
        compare_vecs(label, &cpu3_to_vec(c), &gpu3_to_vec(g))
    }

    // -----------------------------------------------------------------------
    // Probe: causal mask -inf behaviour in softmax
    // -----------------------------------------------------------------------
    fn probe_causal_mask_softmax() {
        println!("\n=== PROBE: softmax(-inf) on both backends ===");
        let cpu_dev = burn_ndarray::NdArrayDevice::default();
        let gpu_dev = burn_wgpu::WgpuDevice::default();

        // 2x2 scores: upper triangle masked with -inf
        let scores: Vec<f32> = vec![1.0, f32::NEG_INFINITY, f32::NEG_INFINITY, 1.0];

        let cpu_probs: Vec<f32> = softmax(
            Tensor::<Cpu, 2>::from_data(TensorData::new(scores.clone(), [2, 2]), &cpu_dev),
            1,
        ).into_data().to_vec().unwrap();

        let gpu_probs: Vec<f32> = softmax(
            Tensor::<Gpu, 2>::from_data(TensorData::new(scores.clone(), [2, 2]), &gpu_dev),
            1,
        ).into_data().to_vec().unwrap();

        println!("  Input: {:?}", &scores);
        println!("  CPU  softmax: {:?}", &cpu_probs);
        println!("  GPU  softmax: {:?}", &gpu_probs);
        println!("  CPU NaN: {}  GPU NaN: {}", cpu_probs.iter().any(|x| x.is_nan()), gpu_probs.iter().any(|x| x.is_nan()));

        // 4x4 causal mask
        let inf = f32::NEG_INFINITY;
        let s4: Vec<f32> = vec![
            1.0, inf, inf, inf,
            1.0, 1.0, inf, inf,
            1.0, 1.0, 1.0, inf,
            1.0, 1.0, 1.0, 1.0,
        ];
        let cpu_p4: Vec<f32> = softmax(
            Tensor::<Cpu, 2>::from_data(TensorData::new(s4.clone(), [4, 4]), &cpu_dev), 1,
        ).into_data().to_vec().unwrap();
        let gpu_p4: Vec<f32> = softmax(
            Tensor::<Gpu, 2>::from_data(TensorData::new(s4.clone(), [4, 4]), &gpu_dev), 1,
        ).into_data().to_vec().unwrap();

        println!("\n  4-token causal mask:");
        println!("  CPU  probs: {:?}", &cpu_p4);
        println!("  GPU  probs: {:?}", &gpu_p4);
        let nan_cpu = cpu_p4.iter().any(|x| x.is_nan());
        let nan_gpu = gpu_p4.iter().any(|x| x.is_nan());
        println!("  CPU NaN: {nan_cpu}  GPU NaN: {nan_gpu}");
        if nan_gpu && !nan_cpu {
            println!("  *** CONFIRMED: GPU softmax(-inf) -> NaN. This is the divergence root cause. ***");
            println!("  *** Fix: replace f32::NEG_INFINITY with a large finite negative (e.g. -1e9) ***");
        }
        let max_diff: f32 = cpu_p4.iter().zip(gpu_p4.iter()).map(|(a,b)|(a-b).abs()).fold(0.0_f32, f32::max);
        println!("  Max abs diff: {max_diff:.4e}  {}", if max_diff < 1e-4 { "MATCHES" } else { "DIVERGED" });
    }

    // -----------------------------------------------------------------------
    // Main test
    // -----------------------------------------------------------------------

    #[test]
    #[ignore = "requires Vulkan/wgpu device; run with --ignored --nocapture"]
    fn ndarray_vs_wgpu_activation_divergence() {
        let gguf_path = std::env::var("AI_ENGINE_DIVERGENCE_GGUF")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("fixtures/toy-llama-3-gguf/model.gguf")
            });

        println!("\n=== Backend Divergence Trace: NdArray vs Wgpu ===");
        println!("GGUF: {}", gguf_path.display());

        let cpu_dev = burn_ndarray::NdArrayDevice::default();
        let gpu_dev = burn_wgpu::WgpuDevice::default();

        // -- Probe causal mask softmax FIRST (no model weights needed) --
        probe_causal_mask_softmax();

        // -- Config --
        let cfg_path = gguf_path.parent().unwrap().join("config.json");
        let cfg = ModelConfig::from_gguf_file(&gguf_path)
            .or_else(|_| ModelConfig::from_file(&cfg_path))
            .expect("failed to load model config");
        let n_layers = cfg.n_layers;
        println!("\nConfig: n_layers={n_layers} hidden={} n_heads={} n_kv_heads={} head_dim={}",
            cfg.hidden_size, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);

        // -- Load weights --
        println!("\n=== Loading weights ===");
        let cpu_w = load_weights::<Cpu>(&gguf_path, &cfg, 0..n_layers, true, true, &cpu_dev)
            .expect("CPU load_weights");
        let gpu_w = load_weights::<Gpu>(&gguf_path, &cfg, 0..n_layers, true, true, &gpu_dev)
            .expect("GPU load_weights");
        let cpu_model = Model::<Cpu>::from_loaded(&cfg, cpu_w, &cpu_dev).expect("CPU from_loaded");
        let gpu_model = Model::<Gpu>::from_loaded(&cfg, gpu_w, &gpu_dev).expect("GPU from_loaded");

        // -- Fixed token input --
        // Use vocab-safe ids; toy model has vocab_size=512
        let max_id = (cfg.vocab_size as i32) - 1;
        let ids: Vec<i32> = vec![1, max_id / 3, max_id / 2];
        let seq_len = ids.len();
        let positions: Vec<i32> = (0..seq_len as i32).collect();

        let cpu_input = Tensor::<Cpu, 2, Int>::from_data(TensorData::new(ids.clone(), [1, seq_len]), &cpu_dev);
        let gpu_input = Tensor::<Gpu, 2, Int>::from_data(TensorData::new(ids.clone(), [1, seq_len]), &gpu_dev);

        // -- KV caches --
        let new_cpu_cache = || -> Vec<KvCacheSlot<Cpu>> {
            (0..n_layers).map(|_| KvCacheSlot::<Cpu>::new(1, cpu_model.n_kv_heads, cpu_model.max_seq, cpu_model.head_dim, &cpu_dev)).collect()
        };
        let new_gpu_cache = || -> Vec<KvCacheSlot<Gpu>> {
            (0..n_layers).map(|_| KvCacheSlot::<Gpu>::new(1, gpu_model.n_kv_heads, gpu_model.max_seq, gpu_model.head_dim, &gpu_dev)).collect()
        };
        let mut cpu_caches = new_cpu_cache();
        let mut gpu_caches = new_gpu_cache();

        // -----------------------------------------------------------------------
        // Phase A: embedding
        // -----------------------------------------------------------------------
        println!("\n=== Phase A: embedding ===");
        let cpu_x = cpu_model.embedding.forward(cpu_input);
        let gpu_x = gpu_model.embedding.forward(gpu_input);
        let emb_rel = cmp3("embedding", &cpu_x, &gpu_x);

        // -----------------------------------------------------------------------
        // Phase B: block 0 sub-steps
        // -----------------------------------------------------------------------
        println!("\n=== Phase B: block 0 sub-steps ===");
        let blk_cpu = &cpu_model.blocks[0];
        let blk_gpu = &gpu_model.blocks[0];

        let cpu_normed = blk_cpu.attn_norm.forward(cpu_x.clone());
        let gpu_normed = blk_gpu.attn_norm.forward(gpu_x.clone());
        cmp3("block0.attn_norm", &cpu_normed, &gpu_normed);

        let cpu_q_raw = blk_cpu.attn.q_proj.matmul(cpu_normed.clone());
        let gpu_q_raw = blk_gpu.attn.q_proj.matmul(gpu_normed.clone());
        cmp3("block0.attn.q_proj(raw)", &cpu_q_raw, &gpu_q_raw);

        let cpu_k_raw = blk_cpu.attn.k_proj.matmul(cpu_normed.clone());
        let gpu_k_raw = blk_gpu.attn.k_proj.matmul(gpu_normed.clone());
        cmp3("block0.attn.k_proj(raw)", &cpu_k_raw, &gpu_k_raw);

        let cpu_v_raw = blk_cpu.attn.v_proj.matmul(cpu_normed.clone());
        let gpu_v_raw = blk_gpu.attn.v_proj.matmul(gpu_normed.clone());
        cmp3("block0.attn.v_proj(raw)", &cpu_v_raw, &gpu_v_raw);

        let cpu_attn_out = blk_cpu.attn.forward(cpu_normed.clone(), &positions, &mut cpu_caches[0]);
        let gpu_attn_out = blk_gpu.attn.forward(gpu_normed.clone(), &positions, &mut gpu_caches[0]);
        let attn_rel = cmp3("block0.attn_out", &cpu_attn_out, &gpu_attn_out);

        let cpu_after_r1 = cpu_x.clone().add(cpu_attn_out);
        let gpu_after_r1 = gpu_x.clone().add(gpu_attn_out);
        cmp3("block0.after_residual1", &cpu_after_r1, &gpu_after_r1);

        let cpu_ffn_normed = blk_cpu.ffn_norm.forward(cpu_after_r1.clone());
        let gpu_ffn_normed = blk_gpu.ffn_norm.forward(gpu_after_r1.clone());
        cmp3("block0.ffn_norm", &cpu_ffn_normed, &gpu_ffn_normed);

        let cpu_gate_raw = blk_cpu.ffn.gate_proj.matmul(cpu_ffn_normed.clone());
        let gpu_gate_raw = blk_gpu.ffn.gate_proj.matmul(gpu_ffn_normed.clone());
        cmp3("block0.ffn.gate_proj(raw)", &cpu_gate_raw, &gpu_gate_raw);

        let cpu_up_raw = blk_cpu.ffn.up_proj.matmul(cpu_ffn_normed.clone());
        let gpu_up_raw = blk_gpu.ffn.up_proj.matmul(gpu_ffn_normed.clone());
        cmp3("block0.ffn.up_proj(raw)", &cpu_up_raw, &gpu_up_raw);

        let cpu_ffn_out = blk_cpu.ffn.forward(cpu_ffn_normed.clone());
        let gpu_ffn_out = blk_gpu.ffn.forward(gpu_ffn_normed.clone());
        cmp3("block0.ffn_out", &cpu_ffn_out, &gpu_ffn_out);

        let cpu_blk0 = cpu_after_r1.add(cpu_ffn_out);
        let gpu_blk0 = gpu_after_r1.add(gpu_ffn_out);
        let b0_rel = cmp3("block0.output", &cpu_blk0, &gpu_blk0);

        if classify(attn_rel) == "DIVERGED" {
            println!("\n  *** block0.attn_out DIVERGED — see causal mask probe above ***");
        }
        println!("  -> block0 status: {}", classify(b0_rel));

        // -----------------------------------------------------------------------
        // Block 1 sub-steps (drill in if block1.output diverges)
        // -----------------------------------------------------------------------
        // -----------------------------------------------------------------------
        // Weight sanity check: compare block0 and block1 attn_norm weights
        // -----------------------------------------------------------------------
        println!("\n=== Weight check: block0 vs block1 attn_norm weights ===");
        {
            let w0_cpu: Vec<f32> = cpu_model.blocks[0].attn_norm.weight.clone().into_data().to_vec().unwrap();
            let w0_gpu: Vec<f32> = gpu_model.blocks[0].attn_norm.weight.clone().into_data().to_vec().unwrap();
            let w1_cpu: Vec<f32> = cpu_model.blocks[1].attn_norm.weight.clone().into_data().to_vec().unwrap();
            let w1_gpu: Vec<f32> = gpu_model.blocks[1].attn_norm.weight.clone().into_data().to_vec().unwrap();
            let d0 = l2_dist(&w0_cpu, &w0_gpu);
            let d1 = l2_dist(&w1_cpu, &w1_gpu);
            let n0 = l2_norm(&w0_cpu);
            let n1 = l2_norm(&w1_cpu);
            println!("  block0.attn_norm weight: CPU-GPU L2={d0:.4e} norm={n0:.4e} rel={:.4e} -> {}", if n0>1e-9{d0/n0}else{d0}, classify(if n0>1e-9{d0/n0}else{d0}));
            println!("  block1.attn_norm weight: CPU-GPU L2={d1:.4e} norm={n1:.4e} rel={:.4e} -> {}", if n1>1e-9{d1/n1}else{d1}, classify(if n1>1e-9{d1/n1}else{d1}));
            println!("  block0 cpu w[0..4]: {:?}", &w0_cpu[..4.min(w0_cpu.len())]);
            println!("  block0 gpu w[0..4]: {:?}", &w0_gpu[..4.min(w0_gpu.len())]);
            println!("  block1 cpu w[0..4]: {:?}", &w1_cpu[..4.min(w1_cpu.len())]);
            println!("  block1 gpu w[0..4]: {:?}", &w1_gpu[..4.min(w1_gpu.len())]);
        }

        println!("\n=== Phase B2: block 1 sub-steps ===");
        {
            // Run block0 on fresh caches to get input for block1
            let mut cc = new_cpu_cache();
            let mut gc = new_gpu_cache();
            let cpu_h0 = cpu_model.blocks[0].forward(cpu_x.clone(), &positions, &mut cc[0]);
            let gpu_h0 = gpu_model.blocks[0].forward(gpu_x.clone(), &positions, &mut gc[0]);

            // Verify that block0 output is truly identical before passing to block1
            cmp3("block1_input(=block0_output)", &cpu_h0, &gpu_h0);

            let blk1_cpu = &cpu_model.blocks[1];
            let blk1_gpu = &gpu_model.blocks[1];

            let cpu_n1 = blk1_cpu.attn_norm.forward(cpu_h0.clone());
            let gpu_n1 = blk1_gpu.attn_norm.forward(gpu_h0.clone());
            cmp3("block1.attn_norm", &cpu_n1, &gpu_n1);

            let cpu_q1 = blk1_cpu.attn.q_proj.matmul(cpu_n1.clone());
            let gpu_q1 = blk1_gpu.attn.q_proj.matmul(gpu_n1.clone());
            cmp3("block1.attn.q_proj(raw)", &cpu_q1, &gpu_q1);

            let cpu_k1 = blk1_cpu.attn.k_proj.matmul(cpu_n1.clone());
            let gpu_k1 = blk1_gpu.attn.k_proj.matmul(gpu_n1.clone());
            cmp3("block1.attn.k_proj(raw)", &cpu_k1, &gpu_k1);

            let cpu_v1 = blk1_cpu.attn.v_proj.matmul(cpu_n1.clone());
            let gpu_v1 = blk1_gpu.attn.v_proj.matmul(gpu_n1.clone());
            cmp3("block1.attn.v_proj(raw)", &cpu_v1, &gpu_v1);

            let cpu_attn1 = blk1_cpu.attn.forward(cpu_n1.clone(), &positions, &mut cc[1]);
            let gpu_attn1 = blk1_gpu.attn.forward(gpu_n1.clone(), &positions, &mut gc[1]);
            let a1_rel = cmp3("block1.attn_out", &cpu_attn1, &gpu_attn1);

            let cpu_r1 = cpu_h0.clone().add(cpu_attn1);
            let gpu_r1 = gpu_h0.clone().add(gpu_attn1);
            cmp3("block1.after_residual1", &cpu_r1, &gpu_r1);

            let cpu_fn1 = blk1_cpu.ffn_norm.forward(cpu_r1.clone());
            let gpu_fn1 = blk1_gpu.ffn_norm.forward(gpu_r1.clone());
            cmp3("block1.ffn_norm", &cpu_fn1, &gpu_fn1);

            let cpu_gate1 = blk1_cpu.ffn.gate_proj.matmul(cpu_fn1.clone());
            let gpu_gate1 = blk1_gpu.ffn.gate_proj.matmul(gpu_fn1.clone());
            cmp3("block1.ffn.gate_proj(raw)", &cpu_gate1, &gpu_gate1);

            let cpu_up1 = blk1_cpu.ffn.up_proj.matmul(cpu_fn1.clone());
            let gpu_up1 = blk1_gpu.ffn.up_proj.matmul(gpu_fn1.clone());
            cmp3("block1.ffn.up_proj(raw)", &cpu_up1, &gpu_up1);

            let cpu_ffn1 = blk1_cpu.ffn.forward(cpu_fn1.clone());
            let gpu_ffn1 = blk1_gpu.ffn.forward(gpu_fn1.clone());
            cmp3("block1.ffn_out", &cpu_ffn1, &gpu_ffn1);

            let cpu_b1 = cpu_r1.add(cpu_ffn1);
            let gpu_b1 = gpu_r1.add(gpu_ffn1);
            let b1_rel = cmp3("block1.output", &cpu_b1, &gpu_b1);
            println!("  -> block1 attn_out: {}  block1 output: {}", classify(a1_rel), classify(b1_rel));
        }

        // -----------------------------------------------------------------------
        // Phase C: remaining blocks (fresh caches; block0 already tracked above)
        // -----------------------------------------------------------------------
        println!("\n=== Phase C: remaining blocks ===");
        // Run block0 again on fresh caches to get h for sequential pass
        let mut cpu_caches2 = new_cpu_cache();
        let mut gpu_caches2 = new_gpu_cache();
        let mut cpu_h = cpu_model.blocks[0].forward(cpu_x.clone(), &positions, &mut cpu_caches2[0]);
        let mut gpu_h = gpu_model.blocks[0].forward(gpu_x.clone(), &positions, &mut gpu_caches2[0]);

        let mut first_diverged: Option<String> = None;

        for i in 1..n_layers {
            cpu_h = cpu_model.blocks[i].forward(cpu_h, &positions, &mut cpu_caches2[i]);
            gpu_h = gpu_model.blocks[i].forward(gpu_h, &positions, &mut gpu_caches2[i]);
            let rel = cmp3(&format!("block{i}.output"), &cpu_h, &gpu_h);
            if first_diverged.is_none() && classify(rel) == "DIVERGED" {
                first_diverged = Some(format!("block{i}.output"));
            }
        }

        // -----------------------------------------------------------------------
        // Phase D: final_norm + logits
        // -----------------------------------------------------------------------
        println!("\n=== Phase D: final_norm + logits ===");
        let cpu_nf = cpu_model.final_norm.forward(cpu_h);
        let gpu_nf = gpu_model.final_norm.forward(gpu_h);
        let fn_rel = cmp3("final_norm", &cpu_nf, &gpu_nf);
        if first_diverged.is_none() && classify(fn_rel) == "DIVERGED" {
            first_diverged = Some("final_norm".to_string());
        }

        let cpu_logits = cpu_model.output.forward(cpu_nf);
        let gpu_logits = gpu_model.output.forward(gpu_nf);
        let logits_rel = cmp3("logits", &cpu_logits, &gpu_logits);
        if first_diverged.is_none() && classify(logits_rel) == "DIVERGED" {
            first_diverged = Some("logits".to_string());
        }

        println!("\n=== SUMMARY ===");
        println!("embedding  rel: {emb_rel:.4e}  {}", classify(emb_rel));
        println!("block0     rel: {b0_rel:.4e}  {}", classify(b0_rel));
        println!("first DIVERGED step: {}", first_diverged.as_deref().unwrap_or("none"));
        println!("logits     rel: {logits_rel:.4e}  {}", classify(logits_rel));
    }
}
