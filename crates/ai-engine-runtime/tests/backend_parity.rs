#[cfg(all(feature = "backend-cpu", feature = "backend-wgpu"))]
mod parity {
    use ai_engine_runtime::arch::model::Model;
    use ai_engine_runtime::config::ModelConfig;
    use ai_engine_runtime::loader::load_range;
    use ai_engine_tokenizer::{HfTokenizer, Tokenizer};
    use burn::tensor::{Tensor, Int, TensorData};
    use std::path::PathBuf;

    fn fixture() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/toy-llama-3")
    }

    #[test]
    #[ignore = "requires a Vulkan/Metal-capable device; run with --ignored after confirming GPU availability"]
    fn cpu_and_wgpu_produce_matching_logits() {
        let fix = fixture();
        let cfg = ModelConfig::from_file(&fix.join("config.json")).unwrap();
        let tok = HfTokenizer::from_path(fix.join("tokenizer.json")).unwrap();
        let prompt = "The quick brown fox";
        let ids: Vec<i32> = tok.encode(prompt).unwrap().iter().map(|x| *x as i32).collect();

        type Cpu = burn_ndarray::NdArray;
        let cpu_dev = Default::default();
        let cpu_w = load_range::<Cpu>(&fix.join("model.safetensors"), &cfg, 0..cfg.n_layers, true, true, &cpu_dev).unwrap();
        let cpu_model = Model::<Cpu>::from_loaded(&cfg, cpu_w, &cpu_dev).unwrap();
        let cpu_input = Tensor::<Cpu, 2, Int>::from_data(
            TensorData::new(ids.clone(), [1, ids.len()]), &cpu_dev,
        );
        let cpu_logits: Vec<f32> = cpu_model.forward(cpu_input, 0)
            .slice([0..1, (ids.len()-1)..ids.len(), 0..cfg.vocab_size])
            .reshape([cfg.vocab_size]).to_data().to_vec().unwrap();

        type Wgpu = burn_wgpu::Wgpu;
        let wgpu_dev = burn_wgpu::WgpuDevice::default();
        let wgpu_w = load_range::<Wgpu>(&fix.join("model.safetensors"), &cfg, 0..cfg.n_layers, true, true, &wgpu_dev).unwrap();
        let wgpu_model = Model::<Wgpu>::from_loaded(&cfg, wgpu_w, &wgpu_dev).unwrap();
        let wgpu_input = Tensor::<Wgpu, 2, Int>::from_data(
            TensorData::new(ids.clone(), [1, ids.len()]), &wgpu_dev,
        );
        let wgpu_logits: Vec<f32> = wgpu_model.forward(wgpu_input, 0)
            .slice([0..1, (ids.len()-1)..ids.len(), 0..cfg.vocab_size])
            .reshape([cfg.vocab_size]).to_data().to_vec().unwrap();

        let max_diff: f32 = cpu_logits.iter().zip(wgpu_logits.iter())
            .map(|(a, b)| (a - b).abs()).fold(0., f32::max);
        eprintln!("CPU vs WGPU max diff = {max_diff}");
        assert!(max_diff < 5e-3, "CPU vs WGPU max diff = {max_diff}");
    }
}
