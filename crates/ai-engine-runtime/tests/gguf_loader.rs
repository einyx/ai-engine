use ai_engine_runtime::name_map::hf_from_gguf;

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
