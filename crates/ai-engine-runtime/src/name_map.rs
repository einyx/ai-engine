//! Maps logical weight tensor identifiers to HF safetensors names per family.

use crate::config::ModelFamily;

#[derive(Debug, Clone, Copy)]
pub enum TensorId {
    Embedding,
    FinalNorm,
    OutputProjection,
    LayerAttnNorm(usize),
    LayerQProj(usize),
    LayerKProj(usize),
    LayerVProj(usize),
    LayerOProj(usize),
    LayerFfnNorm(usize),
    LayerFfnGate(usize),
    LayerFfnUp(usize),
    LayerFfnDown(usize),
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
            Embedding         => "model.embed_tokens.weight".into(),
            FinalNorm         => "model.norm.weight".into(),
            OutputProjection  => "lm_head.weight".into(),
            LayerAttnNorm(i)  => format!("model.layers.{i}.input_layernorm.weight"),
            LayerQProj(i)     => format!("model.layers.{i}.self_attn.q_proj.weight"),
            LayerKProj(i)     => format!("model.layers.{i}.self_attn.k_proj.weight"),
            LayerVProj(i)     => format!("model.layers.{i}.self_attn.v_proj.weight"),
            LayerOProj(i)     => format!("model.layers.{i}.self_attn.o_proj.weight"),
            LayerFfnNorm(i)   => format!("model.layers.{i}.post_attention_layernorm.weight"),
            LayerFfnGate(i)   => format!("model.layers.{i}.mlp.gate_proj.weight"),
            LayerFfnUp(i)     => format!("model.layers.{i}.mlp.up_proj.weight"),
            LayerFfnDown(i)   => format!("model.layers.{i}.mlp.down_proj.weight"),
        }
    }
}
