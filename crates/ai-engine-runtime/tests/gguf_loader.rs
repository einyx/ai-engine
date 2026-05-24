use ai_engine_runtime::arch::linear::LinearWeight;
use ai_engine_runtime::config::ModelConfig;
use ai_engine_runtime::loader::load_gguf;
use ai_engine_runtime::name_map::hf_from_gguf;
use std::path::PathBuf;

type B = burn_ndarray::NdArray;

fn gguf_fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/toy-llama-3-gguf")
}

#[test]
fn load_gguf_fixture_produces_q4_gguf_weights() {
    let cfg = ModelConfig::from_file(&gguf_fixture().join("config.json")).unwrap();
    let dev = Default::default();
    let weights = load_gguf::<B>(
        &gguf_fixture().join("model.gguf"),
        &cfg,
        0..cfg.n_layers,
        true,
        true,
        &dev,
    )
    .unwrap();
    assert_eq!(weights.layers.len(), cfg.n_layers);
    for (i, layer) in weights.layers.iter().enumerate() {
        assert!(
            matches!(layer.q_proj, LinearWeight::Q4Gguf(_)),
            "layer {i} q_proj should be Q4Gguf"
        );
        assert!(matches!(layer.k_proj, LinearWeight::Q4Gguf(_)));
        assert!(matches!(layer.v_proj, LinearWeight::Q4Gguf(_)));
        assert!(matches!(layer.o_proj, LinearWeight::Q4Gguf(_)));
        assert!(matches!(layer.ffn_gate, LinearWeight::Q4Gguf(_)));
        assert!(matches!(layer.ffn_up, LinearWeight::Q4Gguf(_)));
        assert!(matches!(layer.ffn_down, LinearWeight::Q4Gguf(_)));
    }
    assert!(weights.embedding.is_some());
}

#[test]
fn translates_layer_attn_q() {
    assert_eq!(
        hf_from_gguf("blk.0.attn_q.weight"),
        Some("model.layers.0.self_attn.q_proj.weight".to_string())
    );
    assert_eq!(
        hf_from_gguf("blk.12.attn_k.weight"),
        Some("model.layers.12.self_attn.k_proj.weight".to_string())
    );
}

#[test]
fn translates_boundary_tensors() {
    assert_eq!(
        hf_from_gguf("token_embd.weight"),
        Some("model.embed_tokens.weight".to_string())
    );
    assert_eq!(
        hf_from_gguf("output_norm.weight"),
        Some("model.norm.weight".to_string())
    );
    assert_eq!(
        hf_from_gguf("output.weight"),
        Some("lm_head.weight".to_string())
    );
}

#[test]
fn translates_ffn_layers() {
    assert_eq!(
        hf_from_gguf("blk.0.ffn_gate.weight"),
        Some("model.layers.0.mlp.gate_proj.weight".to_string())
    );
    assert_eq!(
        hf_from_gguf("blk.0.ffn_up.weight"),
        Some("model.layers.0.mlp.up_proj.weight".to_string())
    );
    assert_eq!(
        hf_from_gguf("blk.0.ffn_down.weight"),
        Some("model.layers.0.mlp.down_proj.weight".to_string())
    );
}

#[test]
fn returns_none_for_unknown() {
    assert!(hf_from_gguf("rope_freqs.weight").is_none());
    assert!(hf_from_gguf("not.a.gguf.tensor").is_none());
}
