//! Safetensors loader with per-layer range support.
//!
//! `load_range` mmaps a HuggingFace safetensors file and materialises only the
//! tensors required by the calling node: optionally the embedding and the
//! output/final-norm "boundary" tensors, plus a contiguous range of decoder
//! layers. This is the foundation for distributed inference where each worker
//! owns a slice of the layer stack.

use crate::arch::linear::LinearWeight;
use crate::config::ModelConfig;
use crate::gguf::q4_0::Q4GgufTensor;
use crate::gguf::tensor_desc::{GgmlType, TensorDesc};
use crate::name_map::{hf_from_gguf, TensorId, WeightNameMap};
use crate::quant::{Q4Tensor, Q4_GROUP_SIZE, QuantizedTensor};
use anyhow::Context;
use burn::tensor::{backend::Backend, Tensor, TensorData};
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::collections::HashMap;
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

    // Load a 2D tensor, transparently dequantizing if it's Q8 or Q4. Used for
    // the embedding fallback when a tied-weights fixture stores only `lm_head`
    // in quantized form. For Q4, the on-disk weight is already in math order
    // `[in, out]` (= `[hidden, vocab]`); the embedding expects `[vocab, hidden]`
    // so we transpose after dequantizing.
    let load_2d_dequantizing = |weight_id: TensorId,
                                scale_id: TensorId,
                                q4_weight_id: TensorId,
                                q4_scale_id: TensorId|
     -> anyhow::Result<Tensor<B, 2>> {
        match load_linear_weight::<B>(
            &st,
            &nm,
            weight_id,
            scale_id,
            q4_weight_id,
            q4_scale_id,
            device,
        )? {
            LinearWeight::Dense(t) => Ok(t),
            LinearWeight::Quantized(q) => Ok(q.dequantize()),
            // Q4 lm_head is stored pre-transposed in math order [hidden, vocab].
            // The embedding table is [vocab, hidden]; swap dims to recover it.
            LinearWeight::Q4(q) => Ok(q.dequantize().swap_dims(0, 1)),
            // The safetensors loader never produces a Q4Gguf variant — that
            // path only originates from `load_gguf`.
            LinearWeight::Q4Gguf(_) => {
                unreachable!("safetensors loader cannot produce LinearWeight::Q4Gguf")
            }
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
                            TensorId::OutputProjectionQ4Weight,
                            TensorId::OutputProjectionQ4Scale,
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
                TensorId::LayerQProjQ4Weight(i),
                TensorId::LayerQProjQ4Scale(i),
                device,
            )?,
            k_proj: load_linear_weight::<B>(
                &st,
                &nm,
                TensorId::LayerKProj(i),
                TensorId::LayerKProjScale(i),
                TensorId::LayerKProjQ4Weight(i),
                TensorId::LayerKProjQ4Scale(i),
                device,
            )?,
            v_proj: load_linear_weight::<B>(
                &st,
                &nm,
                TensorId::LayerVProj(i),
                TensorId::LayerVProjScale(i),
                TensorId::LayerVProjQ4Weight(i),
                TensorId::LayerVProjQ4Scale(i),
                device,
            )?,
            o_proj: load_linear_weight::<B>(
                &st,
                &nm,
                TensorId::LayerOProj(i),
                TensorId::LayerOProjScale(i),
                TensorId::LayerOProjQ4Weight(i),
                TensorId::LayerOProjQ4Scale(i),
                device,
            )?,
            ffn_norm: load_1d(TensorId::LayerFfnNorm(i))?,
            ffn_gate: load_linear_weight::<B>(
                &st,
                &nm,
                TensorId::LayerFfnGate(i),
                TensorId::LayerFfnGateScale(i),
                TensorId::LayerFfnGateQ4Weight(i),
                TensorId::LayerFfnGateQ4Scale(i),
                device,
            )?,
            ffn_up: load_linear_weight::<B>(
                &st,
                &nm,
                TensorId::LayerFfnUp(i),
                TensorId::LayerFfnUpScale(i),
                TensorId::LayerFfnUpQ4Weight(i),
                TensorId::LayerFfnUpQ4Scale(i),
                device,
            )?,
            ffn_down: load_linear_weight::<B>(
                &st,
                &nm,
                TensorId::LayerFfnDown(i),
                TensorId::LayerFfnDownScale(i),
                TensorId::LayerFfnDownQ4Weight(i),
                TensorId::LayerFfnDownQ4Scale(i),
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
            TensorId::OutputProjectionQ4Weight,
            TensorId::OutputProjectionQ4Scale,
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

/// Load a 2D Linear weight with three dispatched paths:
///
/// 1. **Q4 path** — if `<name>.q4_weight` (packed nibbles, U8, shape
///    `[in, out/2]`) is present in the safetensors archive, load it together
///    with `<name>.q4_scale` (f32, shape `[in/32, out]`) and build
///    `LinearWeight::Q4(Q4Tensor)`. Q4 weights are stored **pre-transposed**
///    in math order `[in, out]`, so callers must use `ensure_math_order()`
///    instead of `swap_dims(0, 1)`.
/// 2. **Q8 path** — base weight present with dtype `I8` and a single-element
///    `<name>.scale`. Build `LinearWeight::Quantized(QuantizedTensor)`.
/// 3. **Dense path** — base weight present with f32/f16/bf16 dtype. Convert
///    to f32 and build `LinearWeight::Dense`.
fn load_linear_weight<B: Backend>(
    st: &SafeTensors<'_>,
    nm: &WeightNameMap,
    weight_id: TensorId,
    scale_id: TensorId,
    q4_weight_id: TensorId,
    q4_scale_id: TensorId,
    device: &B::Device,
) -> anyhow::Result<LinearWeight<B>> {
    // 1. Q4 path: look for `<name>.q4_weight` first; if present, this weight
    // was Q4-quantized at fixture-generation time.
    let q4_weight_name = nm.lookup(q4_weight_id);
    if let Ok(packed_view) = st.tensor(&q4_weight_name) {
        if packed_view.dtype() != safetensors::Dtype::U8 {
            anyhow::bail!(
                "Q4 weight `{q4_weight_name}` expected U8 dtype, got {:?}",
                packed_view.dtype()
            );
        }
        let packed_shape = packed_view.shape();
        if packed_shape.len() != 2 {
            anyhow::bail!(
                "Q4 weight `{q4_weight_name}` expected 2D, got shape {:?}",
                packed_shape
            );
        }
        // packed_shape = [in, out/2]; reconstructed shape = [in, out].
        let in_dim = packed_shape[0];
        let out_dim = packed_shape[1] * 2;
        let packed: Vec<u8> = packed_view.data().to_vec();

        let q4_scale_name = nm.lookup(q4_scale_id);
        let scale_view = st.tensor(&q4_scale_name).with_context(|| {
            format!("Q4 weight `{q4_weight_name}` missing scale `{q4_scale_name}`")
        })?;
        if scale_view.dtype() != safetensors::Dtype::F32 {
            anyhow::bail!(
                "Q4 scale `{q4_scale_name}` expected F32, got {:?}",
                scale_view.dtype()
            );
        }
        let scale_shape = scale_view.shape();
        if scale_shape.len() != 2 {
            anyhow::bail!(
                "Q4 scale `{q4_scale_name}` expected 2D, got shape {:?}",
                scale_shape
            );
        }
        if in_dim % Q4_GROUP_SIZE != 0 {
            anyhow::bail!(
                "Q4 weight `{q4_weight_name}` in dim {in_dim} not divisible by group size {Q4_GROUP_SIZE}"
            );
        }
        let num_groups = in_dim / Q4_GROUP_SIZE;
        if scale_shape[0] != num_groups || scale_shape[1] != out_dim {
            anyhow::bail!(
                "Q4 scale `{q4_scale_name}` shape {:?} mismatches expected [{}, {}] derived from packed weight",
                scale_shape,
                num_groups,
                out_dim
            );
        }
        let scales: Vec<f32> = bytemuck::cast_slice::<u8, f32>(scale_view.data()).to_vec();
        return Ok(LinearWeight::Q4(Q4Tensor::from_packed(
            packed,
            scales,
            [in_dim, out_dim],
            device,
        )));
    }

    // 2 & 3. Dense / Q8 dispatch on the base tensor.
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

/// Alignment for the tensor-data section of a GGUF file. The format spec
/// allows `general.alignment` metadata to override this, but virtually all
/// real-world files use the default of 32 bytes.
const TENSOR_DATA_ALIGN: u64 = 32;

/// Load a GGUF Q4_0 (with optional F32/F16/BF16 boundary tensors) checkpoint.
///
/// Mirrors `load_range` for safetensors: returns a `LoadedWeights<B>` covering
/// the requested layer range plus optional embedding / final-norm / output
/// projection. Quantized linear weights are kept native (`LinearWeight::Q4Gguf`).
pub fn load_gguf<B: Backend>(
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

    // 1. Header.
    let (hdr, header_consumed) = crate::gguf::header::parse_header(&mmap)?;
    let mut cursor = header_consumed;

    // 2. Metadata.
    let mut metadata: HashMap<String, crate::gguf::metadata::GgufValue> = HashMap::new();
    for _ in 0..hdr.metadata_count {
        let (k, v, consumed) = crate::gguf::metadata::parse_kv(&mmap[cursor..])?;
        metadata.insert(k, v);
        cursor += consumed;
    }

    // 3. Tensor descriptors.
    let mut descs: Vec<TensorDesc> = Vec::with_capacity(hdr.tensor_count as usize);
    for _ in 0..hdr.tensor_count {
        let (d, consumed) = crate::gguf::tensor_desc::parse_tensor_desc(&mmap[cursor..])?;
        descs.push(d);
        cursor += consumed;
    }

    // 4. Align cursor to the start of the tensor data section.
    let alignment = TENSOR_DATA_ALIGN;
    cursor = ((cursor as u64 + alignment - 1) & !(alignment - 1)) as usize;
    let data_start = cursor;

    // 5. Build {hf_name → (desc, slice)} lookup. Slices borrow from `mmap`;
    //    every helper closure below captures `by_hf` by reference, so `mmap`
    //    must outlive both — which it does because both live in this fn.
    let mut by_hf: HashMap<String, (TensorDesc, &[u8])> = HashMap::new();
    for d in &descs {
        let Some(hf_name) = hf_from_gguf(&d.name) else {
            continue;
        };
        let off = data_start + d.offset as usize;
        let size = gguf_tensor_bytes(d)?;
        if off + size > mmap.len() {
            anyhow::bail!("tensor `{}` extends past file end", d.name);
        }
        by_hf.insert(hf_name, (d.clone(), &mmap[off..off + size]));
    }

    // 6. Load helpers.
    let load_2d = |hf_name: &str| -> anyhow::Result<Tensor<B, 2>> {
        let (d, slice) = by_hf
            .get(hf_name)
            .ok_or_else(|| anyhow::anyhow!("missing tensor `{hf_name}` in GGUF"))?;
        load_dense_2d_gguf::<B>(d, slice, device)
    };
    let load_1d = |hf_name: &str| -> anyhow::Result<Tensor<B, 1>> {
        let (d, slice) = by_hf
            .get(hf_name)
            .ok_or_else(|| anyhow::anyhow!("missing tensor `{hf_name}` in GGUF"))?;
        load_dense_1d_gguf::<B>(d, slice, device)
    };
    let load_lin = |hf_name: &str| -> anyhow::Result<LinearWeight<B>> {
        let (d, slice) = by_hf
            .get(hf_name)
            .ok_or_else(|| anyhow::anyhow!("missing tensor `{hf_name}` in GGUF"))?;
        match d.ggml_type {
            GgmlType::Q4_0 => {
                let shape = [d.shape[0] as usize, d.shape[1] as usize];
                Ok(LinearWeight::Q4Gguf(Q4GgufTensor::from_blocks(
                    slice.to_vec(),
                    shape,
                    device,
                )?))
            }
            _ => Ok(LinearWeight::Dense(load_dense_2d_gguf::<B>(
                d, slice, device,
            )?)),
        }
    };

    let embedding = if hosts_embedding {
        let from_embed = load_2d("model.embed_tokens.weight");
        match from_embed {
            Ok(t) => Some(t),
            Err(_) => {
                // Tied-embedding fallback: only `output.weight` (lm_head) is
                // present. GGUF stores it in math order `[hidden, vocab]`; the
                // embedding table needs `[vocab, hidden]`.
                let (d, slice) = by_hf.get("lm_head.weight").ok_or_else(|| {
                    anyhow::anyhow!(
                        "neither model.embed_tokens.weight nor lm_head.weight found in GGUF"
                    )
                })?;
                match d.ggml_type {
                    GgmlType::Q4_0 => {
                        let shape = [d.shape[0] as usize, d.shape[1] as usize];
                        let q = Q4GgufTensor::<B>::from_blocks(slice.to_vec(), shape, device)?;
                        Some(q.dequantize().swap_dims(0, 1))
                    }
                    _ => Some(load_dense_2d_gguf::<B>(d, slice, device)?.swap_dims(0, 1)),
                }
            }
        }
    } else {
        None
    };

    let mut layers = Vec::with_capacity(layer_range.len());
    for i in layer_range.clone() {
        layers.push(LayerWeights {
            attn_norm: load_1d(&format!("model.layers.{i}.input_layernorm.weight"))?,
            q_proj: load_lin(&format!("model.layers.{i}.self_attn.q_proj.weight"))?,
            k_proj: load_lin(&format!("model.layers.{i}.self_attn.k_proj.weight"))?,
            v_proj: load_lin(&format!("model.layers.{i}.self_attn.v_proj.weight"))?,
            o_proj: load_lin(&format!("model.layers.{i}.self_attn.o_proj.weight"))?,
            ffn_norm: load_1d(&format!(
                "model.layers.{i}.post_attention_layernorm.weight"
            ))?,
            ffn_gate: load_lin(&format!("model.layers.{i}.mlp.gate_proj.weight"))?,
            ffn_up: load_lin(&format!("model.layers.{i}.mlp.up_proj.weight"))?,
            ffn_down: load_lin(&format!("model.layers.{i}.mlp.down_proj.weight"))?,
        });
    }

    let final_norm = if hosts_output {
        Some(load_1d("model.norm.weight")?)
    } else {
        None
    };

    let output_proj = if hosts_output && !cfg.tie_word_embeddings {
        Some(load_lin("lm_head.weight")?)
    } else {
        None
    };

    let _ = metadata; // reserved for future use (chat template, vocab, etc.)
    Ok(LoadedWeights {
        embedding,
        layers,
        final_norm,
        output_proj,
    })
}

/// Byte size of a GGUF tensor on disk, given its descriptor.
fn gguf_tensor_bytes(d: &TensorDesc) -> anyhow::Result<usize> {
    let total: u64 = d.shape.iter().product();
    match d.ggml_type {
        GgmlType::F32 => Ok((total * 4) as usize),
        GgmlType::F16 | GgmlType::BF16 => Ok((total * 2) as usize),
        GgmlType::Q4_0 => {
            let block = crate::gguf::q4_0::Q4_0_BLOCK_SIZE as u64;
            let bytes_per_block = crate::gguf::q4_0::Q4_0_BYTES_PER_BLOCK as u64;
            if total % block != 0 {
                anyhow::bail!("Q4_0 tensor `{}` not multiple of 32 elements", d.name);
            }
            Ok((total / block * bytes_per_block) as usize)
        }
    }
}

/// Decode a 2D GGUF tensor (F32/F16/BF16) into burn's row-major `[in, out]`
/// layout. GGUF stores tensors in `[in, out]` math order but column-major
/// over those dims, so we transpose during decode.
fn load_dense_2d_gguf<B: Backend>(
    d: &TensorDesc,
    bytes: &[u8],
    device: &B::Device,
) -> anyhow::Result<Tensor<B, 2>> {
    if d.shape.len() != 2 {
        anyhow::bail!("`{}` expected 2D, got {} dims", d.name, d.shape.len());
    }
    let in_dim = d.shape[0] as usize;
    let out_dim = d.shape[1] as usize;
    let total = in_dim * out_dim;
    let f32_data = bytes_to_f32_vec_gguf(bytes, d.ggml_type, total)?;
    // GGUF flat is column-major over [in, out]; rewrite into row-major.
    let mut burn_flat = vec![0.0_f32; total];
    for i in 0..in_dim {
        for j in 0..out_dim {
            burn_flat[i * out_dim + j] = f32_data[j * in_dim + i];
        }
    }
    Ok(Tensor::<B, 2>::from_data(
        TensorData::new(burn_flat, [in_dim, out_dim]),
        device,
    ))
}

fn load_dense_1d_gguf<B: Backend>(
    d: &TensorDesc,
    bytes: &[u8],
    device: &B::Device,
) -> anyhow::Result<Tensor<B, 1>> {
    if d.shape.len() != 1 {
        anyhow::bail!("`{}` expected 1D, got {} dims", d.name, d.shape.len());
    }
    let total = d.shape[0] as usize;
    let f32_data = bytes_to_f32_vec_gguf(bytes, d.ggml_type, total)?;
    Ok(Tensor::<B, 1>::from_data(
        TensorData::new(f32_data, [total]),
        device,
    ))
}

fn bytes_to_f32_vec_gguf(
    bytes: &[u8],
    ggml_type: GgmlType,
    expected_elements: usize,
) -> anyhow::Result<Vec<f32>> {
    match ggml_type {
        GgmlType::F32 => {
            let v: &[f32] = bytemuck::cast_slice(bytes);
            if v.len() != expected_elements {
                anyhow::bail!(
                    "F32 element count mismatch: expected {expected_elements}, got {}",
                    v.len()
                );
            }
            Ok(v.to_vec())
        }
        GgmlType::F16 => {
            let v: &[u16] = bytemuck::cast_slice(bytes);
            Ok(v.iter().map(|b| half::f16::from_bits(*b).to_f32()).collect())
        }
        GgmlType::BF16 => {
            let v: &[u16] = bytemuck::cast_slice(bytes);
            Ok(v.iter()
                .map(|b| half::bf16::from_bits(*b).to_f32())
                .collect())
        }
        GgmlType::Q4_0 => {
            anyhow::bail!("Q4_0 cannot be loaded as dense; use the Q4Gguf path")
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
