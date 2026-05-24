//! Safetensors loader with per-layer range support.
//!
//! `load_range` mmaps a HuggingFace safetensors file and materialises only the
//! tensors required by the calling node: optionally the embedding and the
//! output/final-norm "boundary" tensors, plus a contiguous range of decoder
//! layers. This is the foundation for distributed inference where each worker
//! owns a slice of the layer stack.

use crate::arch::linear::LinearWeight;
use crate::config::ModelConfig;
use crate::name_map::{TensorId, WeightNameMap};
use crate::quant::QuantizedTensor;
use anyhow::Context;
use burn::tensor::{backend::Backend, Tensor, TensorData};
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::ops::Range;
use std::path::Path;

pub struct LayerWeights<B: Backend> {
    pub attn_norm: Tensor<B, 1>,
    pub q_proj: LinearWeight<B>,
    pub k_proj: LinearWeight<B>,
    pub v_proj: LinearWeight<B>,
    pub o_proj: LinearWeight<B>,
    pub ffn_norm: Tensor<B, 1>,
    pub ffn_gate: LinearWeight<B>,
    pub ffn_up: LinearWeight<B>,
    pub ffn_down: LinearWeight<B>,
}

pub struct LoadedWeights<B: Backend> {
    pub embedding: Option<Tensor<B, 2>>,
    pub layers: Vec<LayerWeights<B>>,
    pub final_norm: Option<Tensor<B, 1>>,
    pub output_proj: Option<LinearWeight<B>>,
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

    // Load a 2D tensor, transparently dequantizing if it's Q8. Used for the
    // embedding fallback when a tied-weights fixture stores only `lm_head` as
    // int8.
    let load_2d_dequantizing = |weight_id: TensorId,
                                scale_id: TensorId|
     -> anyhow::Result<Tensor<B, 2>> {
        match load_linear_weight::<B>(&st, &nm, weight_id, scale_id, device)? {
            LinearWeight::Dense(t) => Ok(t),
            LinearWeight::Quantized(q) => Ok(q.dequantize()),
            LinearWeight::Q4(q) => Ok(q.dequantize()),
        }
    };

    // When `tie_word_embeddings=true`, some HF checkpoints only store one of
    // {`model.embed_tokens.weight`, `lm_head.weight`} and expect callers to
    // share it. Try the embedding name first, then fall back to the lm_head.
    // The embedding stays dense at runtime even when lm_head is Q8 — we
    // dequantize on load.
    let embedding = if hosts_embedding {
        match load_2d(TensorId::Embedding) {
            Ok(t) => Some(t),
            Err(e) if cfg.tie_word_embeddings => {
                let name = nm.lookup(TensorId::Embedding);
                if st.tensor(&name).is_err() {
                    Some(
                        load_2d_dequantizing(
                            TensorId::OutputProjection,
                            TensorId::OutputProjectionScale,
                        )
                        .with_context(|| {
                            "tied embeddings: neither embed_tokens nor lm_head present"
                        })?,
                    )
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
            q_proj: load_linear_weight::<B>(
                &st,
                &nm,
                TensorId::LayerQProj(i),
                TensorId::LayerQProjScale(i),
                device,
            )?,
            k_proj: load_linear_weight::<B>(
                &st,
                &nm,
                TensorId::LayerKProj(i),
                TensorId::LayerKProjScale(i),
                device,
            )?,
            v_proj: load_linear_weight::<B>(
                &st,
                &nm,
                TensorId::LayerVProj(i),
                TensorId::LayerVProjScale(i),
                device,
            )?,
            o_proj: load_linear_weight::<B>(
                &st,
                &nm,
                TensorId::LayerOProj(i),
                TensorId::LayerOProjScale(i),
                device,
            )?,
            ffn_norm: load_1d(TensorId::LayerFfnNorm(i))?,
            ffn_gate: load_linear_weight::<B>(
                &st,
                &nm,
                TensorId::LayerFfnGate(i),
                TensorId::LayerFfnGateScale(i),
                device,
            )?,
            ffn_up: load_linear_weight::<B>(
                &st,
                &nm,
                TensorId::LayerFfnUp(i),
                TensorId::LayerFfnUpScale(i),
                device,
            )?,
            ffn_down: load_linear_weight::<B>(
                &st,
                &nm,
                TensorId::LayerFfnDown(i),
                TensorId::LayerFfnDownScale(i),
                device,
            )?,
        });
    }

    let final_norm = if hosts_output {
        Some(load_1d(TensorId::FinalNorm)?)
    } else {
        None
    };

    let output_proj = if hosts_output && !cfg.tie_word_embeddings {
        Some(load_linear_weight::<B>(
            &st,
            &nm,
            TensorId::OutputProjection,
            TensorId::OutputProjectionScale,
            device,
        )?)
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

/// Load a 2D Linear weight, detecting Q8 (int8 + `<name>.scale` companion) and
/// constructing a `LinearWeight::Quantized` in that case. Otherwise builds a
/// `LinearWeight::Dense` via the standard f32/f16/bf16 conversion path.
fn load_linear_weight<B: Backend>(
    st: &SafeTensors<'_>,
    nm: &WeightNameMap,
    weight_id: TensorId,
    scale_id: TensorId,
    device: &B::Device,
) -> anyhow::Result<LinearWeight<B>> {
    let name = nm.lookup(weight_id);
    let view = st
        .tensor(&name)
        .with_context(|| format!("missing tensor `{name}`"))?;
    let shape = view.shape();
    if shape.len() != 2 {
        anyhow::bail!("tensor `{name}` expected 2D, got shape {:?}", shape);
    }
    let shape2 = [shape[0], shape[1]];

    match view.dtype() {
        safetensors::Dtype::I8 => {
            // Quantized path: load int8 bytes + companion scale.
            let packed_bytes = view.data();
            let packed: Vec<i8> = bytemuck::cast_slice::<u8, i8>(packed_bytes).to_vec();
            let scale_name = nm.lookup(scale_id);
            let scale_view = st.tensor(&scale_name).with_context(|| {
                format!("quantized weight `{name}` missing scale `{scale_name}`")
            })?;
            if scale_view.dtype() != safetensors::Dtype::F32 {
                anyhow::bail!(
                    "scale `{scale_name}` expected F32, got {:?}",
                    scale_view.dtype()
                );
            }
            let scale_f32: &[f32] = bytemuck::cast_slice(scale_view.data());
            if scale_f32.len() != 1 {
                anyhow::bail!(
                    "scale `{scale_name}` must be 1 element, got {}",
                    scale_f32.len()
                );
            }
            Ok(LinearWeight::Quantized(QuantizedTensor::from_packed(
                packed,
                scale_f32[0],
                shape2,
                device,
            )))
        }
        _ => {
            let f32_data = bytes_to_f32_vec(view.data(), view.dtype())
                .with_context(|| format!("decode `{name}`"))?;
            Ok(LinearWeight::Dense(Tensor::<B, 2>::from_data(
                TensorData::new(f32_data, shape2),
                device,
            )))
        }
    }
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
        I8 => anyhow::bail!(
            "int8 tensors are handled via load_linear_weight, not bytes_to_f32_vec"
        ),
        other => anyhow::bail!("unsupported safetensors dtype: {other:?}"),
    }
}
