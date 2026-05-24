use ai_engine_runtime::config::ModelFamily;
use ai_engine_runtime::name_map::{TensorId, WeightNameMap};

#[test]
fn llama3_q_proj_layer_12() {
    let nm = WeightNameMap::for_family(ModelFamily::Llama3);
    assert_eq!(
        nm.lookup(TensorId::LayerQProj(12)),
        "model.layers.12.self_attn.q_proj.weight"
    );
    assert_eq!(
        nm.lookup(TensorId::LayerKProj(12)),
        "model.layers.12.self_attn.k_proj.weight"
    );
}

#[test]
fn llama3_boundary_tensors() {
    let nm = WeightNameMap::for_family(ModelFamily::Llama3);
    assert_eq!(nm.lookup(TensorId::Embedding), "model.embed_tokens.weight");
    assert_eq!(nm.lookup(TensorId::FinalNorm), "model.norm.weight");
    assert_eq!(nm.lookup(TensorId::OutputProjection), "lm_head.weight");
}

#[test]
fn mistral_uses_llama_pattern() {
    let nm = WeightNameMap::for_family(ModelFamily::Mistral);
    assert_eq!(
        nm.lookup(TensorId::LayerFfnGate(0)),
        "model.layers.0.mlp.gate_proj.weight"
    );
}

#[test]
fn qwen25_uses_llama_pattern() {
    let nm = WeightNameMap::for_family(ModelFamily::Qwen25);
    assert_eq!(
        nm.lookup(TensorId::LayerFfnDown(3)),
        "model.layers.3.mlp.down_proj.weight"
    );
}

#[test]
fn llama3_q_proj_scale_companion_name() {
    let nm = WeightNameMap::for_family(ModelFamily::Llama3);
    assert_eq!(
        nm.lookup(TensorId::LayerQProjScale(12)),
        "model.layers.12.self_attn.q_proj.weight.scale"
    );
    assert_eq!(
        nm.lookup(TensorId::LayerFfnGateScale(0)),
        "model.layers.0.mlp.gate_proj.weight.scale"
    );
    assert_eq!(
        nm.lookup(TensorId::OutputProjectionScale),
        "lm_head.weight.scale"
    );
}

#[test]
fn llama3_ffn_gate_scale() {
    let nm = WeightNameMap::for_family(ModelFamily::Llama3);
    assert_eq!(
        nm.lookup(TensorId::LayerFfnGateScale(7)),
        "model.layers.7.mlp.gate_proj.weight.scale"
    );
    assert_eq!(
        nm.lookup(TensorId::LayerFfnUpScale(7)),
        "model.layers.7.mlp.up_proj.weight.scale"
    );
    assert_eq!(
        nm.lookup(TensorId::LayerFfnDownScale(7)),
        "model.layers.7.mlp.down_proj.weight.scale"
    );
}

#[test]
fn llama3_output_projection_scale() {
    let nm = WeightNameMap::for_family(ModelFamily::Llama3);
    assert_eq!(
        nm.lookup(TensorId::OutputProjectionScale),
        "lm_head.weight.scale"
    );
    assert_eq!(
        nm.lookup(TensorId::LayerKProjScale(3)),
        "model.layers.3.self_attn.k_proj.weight.scale"
    );
    assert_eq!(
        nm.lookup(TensorId::LayerVProjScale(3)),
        "model.layers.3.self_attn.v_proj.weight.scale"
    );
    assert_eq!(
        nm.lookup(TensorId::LayerOProjScale(3)),
        "model.layers.3.self_attn.o_proj.weight.scale"
    );
}

#[test]
fn llama3_q4_companion_names() {
    let nm = WeightNameMap::for_family(ModelFamily::Llama3);
    assert_eq!(
        nm.lookup(TensorId::LayerQProjQ4Weight(12)),
        "model.layers.12.self_attn.q_proj.weight.q4_weight"
    );
    assert_eq!(
        nm.lookup(TensorId::LayerQProjQ4Scale(12)),
        "model.layers.12.self_attn.q_proj.weight.q4_scale"
    );
    assert_eq!(
        nm.lookup(TensorId::OutputProjectionQ4Weight),
        "lm_head.weight.q4_weight"
    );
    assert_eq!(
        nm.lookup(TensorId::OutputProjectionQ4Scale),
        "lm_head.weight.q4_scale"
    );
}
