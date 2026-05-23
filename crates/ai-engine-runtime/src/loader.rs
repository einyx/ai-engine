//! Safetensors loader with per-layer range support.
//!
//! `load_range` mmaps a HuggingFace safetensors file and materialises only the
//! tensors required by the calling node: optionally the embedding and the
//! output/final-norm "boundary" tensors, plus a contiguous range of decoder
//! layers. This is the foundation for distributed inference where each worker
//! owns a slice of the layer stack.

use crate::config::ModelConfig;
use crate::name_map::{TensorId, WeightNameMap};
use anyhow::Context;
use burn::tensor::{backend::Backend, Tensor, TensorData};
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::ops::Range;
use std::path::Path;

pub struct LayerWeights<B: Backend> {
    pub attn_norm: Tensor<B, 1>,
    pub q_proj: Tensor<B, 2>,
    pub k_proj: Tensor<B, 2>,
    pub v_proj: Tensor<B, 2>,
    pub o_proj: Tensor<B, 2>,
    pub ffn_norm: Tensor<B, 1>,
    pub ffn_gate: Tensor<B, 2>,
    pub ffn_up: Tensor<B, 2>,
    pub ffn_down: Tensor<B, 2>,
}

pub struct LoadedWeights<B: Backend> {
    pub embedding: Option<Tensor<B, 2>>,
    pub layers: Vec<LayerWeights<B>>,
    pub final_norm: Option<Tensor<B, 1>>,
    pub output_proj: Option<Tensor<B, 2>>,
}

pub fn load_range<B: Backend>(
    path: &Path,
    cfg: &ModelConfig,
    layer_range: Range<usize>,
    hosts_embedding: bool,
    hosts_output: bool,
    device: &B::Device,
) -> anyhow::Result<LoadedWeights<B>> {
    let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mmap =
        unsafe { Mmap::map(&file) }.with_context(|| format!("mmap {}", path.display()))?;
    let st = SafeTensors::deserialize(&mmap)
        .with_context(|| format!("parse safetensors header from {}", path.display()))?;

    let nm = WeightNameMap::for_family(cfg.family);

    let load_2d = |id: TensorId| -> anyhow::Result<Tensor<B, 2>> {
        let name = nm.lookup(id);
        let view = st
            .tensor(&name)
            .with_context(|| format!("missing tensor `{name}`"))?;
        let shape = view.shape();
        if shape.len() != 2 {
            anyhow::bail!("tensor `{name}` expected 2D, got shape {:?}", shape);
        }
        let f32_data = bytes_to_f32_vec(view.data(), view.dtype())
            .with_context(|| format!("decode `{name}`"))?;
        Ok(Tensor::<B, 2>::from_data(
            TensorData::new(f32_data, [shape[0], shape[1]]),
            device,
        ))
    };

    let load_1d = |id: TensorId| -> anyhow::Result<Tensor<B, 1>> {
        let name = nm.lookup(id);
        let view = st
            .tensor(&name)
            .with_context(|| format!("missing tensor `{name}`"))?;
        let shape = view.shape();
        if shape.len() != 1 {
            anyhow::bail!("tensor `{name}` expected 1D, got shape {:?}", shape);
        }
        let f32_data = bytes_to_f32_vec(view.data(), view.dtype())
            .with_context(|| format!("decode `{name}`"))?;
        Ok(Tensor::<B, 1>::from_data(
            TensorData::new(f32_data, [shape[0]]),
            device,
        ))
    };

    // When `tie_word_embeddings=true`, some HF checkpoints only store one of
    // {`model.embed_tokens.weight`, `lm_head.weight`} and expect callers to
    // share it. Try the embedding name first, then fall back to the lm_head.
    let embedding = if hosts_embedding {
        match load_2d(TensorId::Embedding) {
            Ok(t) => Some(t),
            Err(e) if cfg.tie_word_embeddings => {
                let name = nm.lookup(TensorId::Embedding);
                if st.tensor(&name).is_err() {
                    Some(load_2d(TensorId::OutputProjection).with_context(|| {
                        "tied embeddings: neither embed_tokens nor lm_head present"
                    })?)
                } else {
                    return Err(e);
                }
            }
            Err(e) => return Err(e),
        }
    } else {
        None
    };

    let mut layers = Vec::with_capacity(layer_range.len());
    for i in layer_range.clone() {
        layers.push(LayerWeights {
            attn_norm: load_1d(TensorId::LayerAttnNorm(i))?,
            q_proj: load_2d(TensorId::LayerQProj(i))?,
            k_proj: load_2d(TensorId::LayerKProj(i))?,
            v_proj: load_2d(TensorId::LayerVProj(i))?,
            o_proj: load_2d(TensorId::LayerOProj(i))?,
            ffn_norm: load_1d(TensorId::LayerFfnNorm(i))?,
            ffn_gate: load_2d(TensorId::LayerFfnGate(i))?,
            ffn_up: load_2d(TensorId::LayerFfnUp(i))?,
            ffn_down: load_2d(TensorId::LayerFfnDown(i))?,
        });
    }

    let final_norm = if hosts_output {
        Some(load_1d(TensorId::FinalNorm)?)
    } else {
        None
    };

    let output_proj = if hosts_output && !cfg.tie_word_embeddings {
        Some(load_2d(TensorId::OutputProjection)?)
    } else {
        None
    };

    Ok(LoadedWeights {
        embedding,
        layers,
        final_norm,
        output_proj,
    })
}

fn bytes_to_f32_vec(raw: &[u8], dtype: safetensors::Dtype) -> anyhow::Result<Vec<f32>> {
    use safetensors::Dtype::*;
    match dtype {
        F32 => Ok(bytemuck::cast_slice::<u8, f32>(raw).to_vec()),
        F16 => Ok(bytemuck::cast_slice::<u8, half::f16>(raw)
            .iter()
            .map(|x| x.to_f32())
            .collect()),
        BF16 => Ok(bytemuck::cast_slice::<u8, half::bf16>(raw)
            .iter()
            .map(|x| x.to_f32())
            .collect()),
        other => anyhow::bail!("unsupported safetensors dtype: {other:?}"),
    }
}
