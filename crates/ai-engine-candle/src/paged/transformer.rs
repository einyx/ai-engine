//! Quantized forward stack with paged attention.

use anyhow::Context;
use candle_core::quantized::{gguf_file, QMatMul};
use candle_core::{Device, Tensor};
use candle_nn::RmsNorm;
use std::path::Path;

use crate::paged::arch::ArchConfig;
use crate::paged::rope::Rope;

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
    embed: Tensor,
    layers: Vec<Layer>,
    out_norm: RmsNorm,
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

    /// Decode/prefill step: one token per row. `token_ids` len == batch.
    /// `positions[b]` = current global position of row b's new token.
    /// `seq_lens[b]` = tokens already in the KV pool for row b (before this token).
    /// `kv`: per-layer KV pools. `alloc`/`tables`: block bookkeeping per row.
    /// Returns logits (batch, vocab).
    pub fn decode_step(
        &self,
        token_ids: &[u32],
        positions: &[usize],
        seq_lens: &[usize],
        kv: &mut [crate::paged::attention::KvPool],
        alloc: &crate::paged::block_table::BlockAllocator,
        tables: &[&crate::paged::block_table::BlockTable],
    ) -> anyhow::Result<candle_core::Tensor> {
        use candle_core::Tensor;
        use candle_nn::Module;
        use crate::paged::attention::{build_mask, sdpa};
        let batch = token_ids.len();
        let (n_head, n_kv, hd) = (self.cfg.head_count, self.cfg.head_count_kv, self.cfg.head_dim);

        let ids = Tensor::new(token_ids, &self.device)?;
        let mut x = self.embed.index_select(&ids, 0)?; // (batch, embed)

        let pos_vec: Vec<u32> = positions.iter().map(|&p| p as u32).collect();
        let pos_t = Tensor::new(pos_vec.as_slice(), &self.device)?;
        let (cos, sin) = self.rope.gather(&pos_t)?; // (batch, rope_dim/2)

        let kv_len = seq_lens.iter().cloned().max().unwrap_or(0) + 1;

        for (li, layer) in self.layers.iter().enumerate() {
            let residual = x.clone();
            let xn = layer.attn_norm.forward(&x)?;
            let mut q = layer.attn_q.forward(&xn)?;
            let mut k = layer.attn_k.forward(&xn)?;
            let mut v = layer.attn_v.forward(&xn)?;
            if let Some(b) = &layer.attn_q_bias { q = q.broadcast_add(b)?; }
            if let Some(b) = &layer.attn_k_bias { k = k.broadcast_add(b)?; }
            if let Some(b) = &layer.attn_v_bias { v = v.broadcast_add(b)?; }
            let mut q = q.reshape((batch, n_head, hd))?;
            let mut k = k.reshape((batch, n_kv, hd))?;
            let v = v.reshape((batch, n_kv, hd))?;
            if let Some(qn) = &layer.attn_q_norm { q = qn.forward(&q)?; }
            if let Some(kn) = &layer.attn_k_norm { k = kn.forward(&k)?; }
            let q = crate::paged::rope::apply_rope(&q, &cos, &sin)?;
            let k = crate::paged::rope::apply_rope(&k, &cos, &sin)?;

            let mut k_rows = Vec::with_capacity(batch);
            let mut v_rows = Vec::with_capacity(batch);
            for b in 0..batch {
                let kb = k.narrow(0, b, 1)?.reshape((1, n_kv, hd))?;
                let vb = v.narrow(0, b, 1)?.reshape((1, n_kv, hd))?;
                kv[li].write(alloc, tables[b], seq_lens[b], &kb, &vb)?;
                let (gk, gv) = kv[li].gather_seq(tables[b], seq_lens[b] + 1)?;
                let pad = kv_len - (seq_lens[b] + 1);
                let (gk, gv) = if pad > 0 {
                    (gk.pad_with_zeros(0, 0, pad)?, gv.pad_with_zeros(0, 0, pad)?)
                } else { (gk, gv) };
                k_rows.push(gk.unsqueeze(0)?);
                v_rows.push(gv.unsqueeze(0)?);
            }
            let k_all = Tensor::cat(&k_rows, 0)?.transpose(1, 2)?; // (batch, n_kv, kv_len, hd)
            let v_all = Tensor::cat(&v_rows, 0)?.transpose(1, 2)?;
            let q4 = q.unsqueeze(2)?; // (batch, n_head, 1, hd)

            let valid_lens: Vec<usize> = seq_lens.iter().map(|s| s + 1).collect();
            let mask = build_mask(&valid_lens, positions, kv_len, &self.device)?;

            let attn = sdpa(&q4, &k_all, &v_all, &mask, n_head, n_kv)?; // (batch,1,n_head*hd)
            let attn = attn.reshape((batch, n_head * hd))?;
            let attn = layer.attn_o.forward(&attn)?;
            x = (residual + attn)?;

            let residual = x.clone();
            let xn = layer.ffn_norm.forward(&x)?;
            let gate = candle_nn::ops::silu(&layer.ffn_gate.forward(&xn)?)?;
            let up = layer.ffn_up.forward(&xn)?;
            let mlp = layer.ffn_down.forward(&(gate * up)?)?;
            x = (residual + mlp)?;
        }
        let x = self.out_norm.forward(&x)?;
        let logits = self.lm_head.forward(&x)?; // (batch, vocab)
        Ok(logits)
    }
}
