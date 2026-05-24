//! Maps logical weight tensor identifiers to HF safetensors names per family.

use crate::config::ModelFamily;

#[derive(Debug, Clone, Copy)]
pub enum TensorId {
    Embedding,
    FinalNorm,
    OutputProjection,
    OutputProjectionScale,
    LayerAttnNorm(usize),
    LayerQProj(usize),
    LayerQProjScale(usize),
    LayerKProj(usize),
    LayerKProjScale(usize),
    LayerVProj(usize),
    LayerVProjScale(usize),
    LayerOProj(usize),
    LayerOProjScale(usize),
    LayerFfnNorm(usize),
    LayerFfnGate(usize),
    LayerFfnGateScale(usize),
    LayerFfnUp(usize),
    LayerFfnUpScale(usize),
    LayerFfnDown(usize),
    LayerFfnDownScale(usize),
}

pub struct WeightNameMap { family: ModelFamily }

impl WeightNameMap {
    pub fn for_family(family: ModelFamily) -> Self { Self { family } }

    pub fn lookup(&self, id: TensorId) -> String {
        match self.family {
            // Llama-3, Mistral, DeepSeek-V2 all use the same `model.layers.N.*` naming.
            ModelFamily::Llama3
            | ModelFamily::Mistral
            | ModelFamily::DeepSeekV2 => Self::llama_style(id),
            // Qwen 2.5 also matches Llama's convention in current HF dumps.
            ModelFamily::Qwen25 => Self::llama_style(id),
        }
    }

    fn llama_style(id: TensorId) -> String {
        use TensorId::*;
        match id {
            Embedding              => "model.embed_tokens.weight".into(),
            FinalNorm              => "model.norm.weight".into(),
            OutputProjection       => "lm_head.weight".into(),
            OutputProjectionScale  => "lm_head.weight.scale".into(),
            LayerAttnNorm(i)       => format!("model.layers.{i}.input_layernorm.weight"),
            LayerQProj(i)          => format!("model.layers.{i}.self_attn.q_proj.weight"),
            LayerQProjScale(i)     => format!("model.layers.{i}.self_attn.q_proj.weight.scale"),
            LayerKProj(i)          => format!("model.layers.{i}.self_attn.k_proj.weight"),
            LayerKProjScale(i)     => format!("model.layers.{i}.self_attn.k_proj.weight.scale"),
            LayerVProj(i)          => format!("model.layers.{i}.self_attn.v_proj.weight"),
            LayerVProjScale(i)     => format!("model.layers.{i}.self_attn.v_proj.weight.scale"),
            LayerOProj(i)          => format!("model.layers.{i}.self_attn.o_proj.weight"),
            LayerOProjScale(i)     => format!("model.layers.{i}.self_attn.o_proj.weight.scale"),
            LayerFfnNorm(i)        => format!("model.layers.{i}.post_attention_layernorm.weight"),
            LayerFfnGate(i)        => format!("model.layers.{i}.mlp.gate_proj.weight"),
            LayerFfnGateScale(i)   => format!("model.layers.{i}.mlp.gate_proj.weight.scale"),
            LayerFfnUp(i)          => format!("model.layers.{i}.mlp.up_proj.weight"),
            LayerFfnUpScale(i)     => format!("model.layers.{i}.mlp.up_proj.weight.scale"),
            LayerFfnDown(i)        => format!("model.layers.{i}.mlp.down_proj.weight"),
            LayerFfnDownScale(i)   => format!("model.layers.{i}.mlp.down_proj.weight.scale"),
        }
    }
}
