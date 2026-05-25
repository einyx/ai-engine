//! Quantized forward stack with paged attention.

use anyhow::Context;
use candle_core::quantized::{gguf_file, QMatMul};
use candle_core::{Device, Tensor};
use candle_nn::RmsNorm;
use std::path::Path;

use crate::paged::arch::ArchConfig;
use crate::paged::rope::Rope;

#[allow(dead_code)]
struct Layer {
    attn_q: QMatMul,
    attn_k: QMatMul,
    attn_v: QMatMul,
    attn_o: QMatMul,
    attn_q_bias: Option<Tensor>,
    attn_k_bias: Option<Tensor>,
    attn_v_bias: Option<Tensor>,
    attn_q_norm: Option<RmsNorm>,
    attn_k_norm: Option<RmsNorm>,
    attn_norm: RmsNorm,
    ffn_norm: RmsNorm,
    ffn_gate: QMatMul,
    ffn_up: QMatMul,
    ffn_down: QMatMul,
}

pub struct Transformer {
    pub cfg: ArchConfig,
    #[allow(dead_code)]
    embed: Tensor,
    #[allow(dead_code)]
    layers: Vec<Layer>,
    #[allow(dead_code)]
    out_norm: RmsNorm,
    #[allow(dead_code)]
    lm_head: QMatMul,
    pub rope: Rope,
    pub device: Device,
}

fn rms_from(ct: &gguf_file::Content, r: &mut std::fs::File, name: &str, eps: f64, dev: &Device) -> anyhow::Result<RmsNorm> {
    let w = ct.tensor(r, name, dev)?.dequantize(dev)?;
    Ok(RmsNorm::new(w, eps))
}

impl Transformer {
    pub fn load(gguf_path: &Path, device: Device, max_seq: usize) -> anyhow::Result<Self> {
        let mut file = std::fs::File::open(gguf_path)
            .with_context(|| format!("open {}", gguf_path.display()))?;
        let ct = gguf_file::Content::read(&mut file)
            .map_err(|e| anyhow::anyhow!("read gguf: {e}"))?;
        let cfg = ArchConfig::from_gguf(&ct)?;
        let eps = cfg.rms_norm_eps;

        let embed_q = ct.tensor(&mut file, "token_embd.weight", &device)?;
        let embed = embed_q.dequantize(&device)?;
        let out_norm = rms_from(&ct, &mut file, "output_norm.weight", eps, &device)?;
        let lm_head_q = match ct.tensor(&mut file, "output.weight", &device) {
            Ok(t) => t,
            Err(_) => ct.tensor(&mut file, "token_embd.weight", &device)?,
        };
        let lm_head = QMatMul::from_qtensor(lm_head_q)?;

        let mut layers = Vec::with_capacity(cfg.block_count);
        for i in 0..cfg.block_count {
            let p = format!("blk.{i}");
            let attn_q_bias = if cfg.qkv_bias {
                ct.tensor(&mut file, &format!("{p}.attn_q.bias"), &device)
                    .ok()
                    .and_then(|t| t.dequantize(&device).ok())
            } else {
                None
            };
            let attn_k_bias = if cfg.qkv_bias {
                ct.tensor(&mut file, &format!("{p}.attn_k.bias"), &device)
                    .ok()
                    .and_then(|t| t.dequantize(&device).ok())
            } else {
                None
            };
            let attn_v_bias = if cfg.qkv_bias {
                ct.tensor(&mut file, &format!("{p}.attn_v.bias"), &device)
                    .ok()
                    .and_then(|t| t.dequantize(&device).ok())
            } else {
                None
            };
            let qn = if cfg.qk_norm {
                Some(rms_from(&ct, &mut file, &format!("{p}.attn_q_norm.weight"), eps, &device)?)
            } else {
                None
            };
            let kn = if cfg.qk_norm {
                Some(rms_from(&ct, &mut file, &format!("{p}.attn_k_norm.weight"), eps, &device)?)
            } else {
                None
            };
            layers.push(Layer {
                attn_q: QMatMul::from_qtensor(ct.tensor(&mut file, &format!("{p}.attn_q.weight"), &device)?)?,
                attn_k: QMatMul::from_qtensor(ct.tensor(&mut file, &format!("{p}.attn_k.weight"), &device)?)?,
                attn_v: QMatMul::from_qtensor(ct.tensor(&mut file, &format!("{p}.attn_v.weight"), &device)?)?,
                attn_o: QMatMul::from_qtensor(ct.tensor(&mut file, &format!("{p}.attn_output.weight"), &device)?)?,
                attn_q_bias,
                attn_k_bias,
                attn_v_bias,
                attn_q_norm: qn,
                attn_k_norm: kn,
                attn_norm: rms_from(&ct, &mut file, &format!("{p}.attn_norm.weight"), eps, &device)?,
                ffn_norm: rms_from(&ct, &mut file, &format!("{p}.ffn_norm.weight"), eps, &device)?,
                ffn_gate: QMatMul::from_qtensor(ct.tensor(&mut file, &format!("{p}.ffn_gate.weight"), &device)?)?,
                ffn_up: QMatMul::from_qtensor(ct.tensor(&mut file, &format!("{p}.ffn_up.weight"), &device)?)?,
                ffn_down: QMatMul::from_qtensor(ct.tensor(&mut file, &format!("{p}.ffn_down.weight"), &device)?)?,
            });
        }
        let rope = Rope::new(cfg.rope_dim, cfg.rope_freq_base, max_seq, &device)?;
        Ok(Self { cfg, embed, layers, out_norm, lm_head, rope, device })
    }
}
