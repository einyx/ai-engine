//! Symmetric peer node: loads only the stages its manifest assignment owns
//! and serves them over QUIC. Generalises the leader/worker asymmetry.

use crate::protocol::codec::{decode, encode};
use crate::protocol::data::{ActivationHeader, Dtype};
use crate::protocol::peer::{CoordinatorMsg, PeerFrame, PeerMsg};
use crate::tensor_io::{tensor_from_bytes, tensor_to_bytes};
use crate::transport::frame::{read_frame, write_frame};
use ai_engine_runtime::arch::attention::Attention;
use ai_engine_runtime::arch::block::DecoderBlock;
use ai_engine_runtime::arch::embedding::{OutputProjection, TokenEmbedding};
use ai_engine_runtime::arch::ffn::SwiGluFfn;
use ai_engine_runtime::arch::linear::LinearWeight;
use ai_engine_runtime::arch::rmsnorm::RmsNorm;
use ai_engine_runtime::arch::rope::RotaryEmbedding;
use ai_engine_runtime::config::ModelConfig;
use ai_engine_runtime::kv_cache::KvCacheSlot;
use ai_engine_runtime::loader::load_weights;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use quinn::Connection;
use std::collections::HashMap;
use std::ops::Range;
use std::path::Path;
use uuid::Uuid;

pub struct PeerStages<B: Backend> {
    pub blocks: Vec<DecoderBlock<B>>,
    pub embedding: Option<TokenEmbedding<B>>,
    pub final_norm: Option<RmsNorm<B>>,
    pub output: Option<OutputProjection<B>>,
    pub cfg: ModelConfig,
}

pub fn build_peer_stages<B>(
    model_path: &Path,
    cfg: &ModelConfig,
    layer_range: Range<usize>,
    hosts_embedding: bool,
    hosts_output: bool,
) -> anyhow::Result<PeerStages<B>>
where
    B: Backend,
    B::Device: Default,
{
    // With tied word embeddings the output projection reuses the embedding
    // tensor, so the output host must *load* the embedding weights even when it
    // is not the embedding stage host (head/tail split in leaderless mode).
    // The embedding *stage* (TokenEmbedding) below still depends on the real
    // `hosts_embedding` flag — this only governs which tensors are read.
    let load_embedding = hosts_embedding || (hosts_output && cfg.tie_word_embeddings);

    let device = B::Device::default();
    let weights = load_weights::<B>(
        model_path,
        cfg,
        layer_range.clone(),
        load_embedding,
        hosts_output,
        &device,
    )?;

    // Extract embedding tensor once; it may be reused for tied output projection.
    let embed_tensor = weights.embedding;

    let embedding = if hosts_embedding {
        let t = embed_tensor
            .as_ref()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("embedding required for embedding host"))?;
        Some(TokenEmbedding::new(t))
    } else {
        None
    };

    let final_norm = if hosts_output {
        Some(RmsNorm::new(
            weights
                .final_norm
                .ok_or_else(|| anyhow::anyhow!("final_norm required for output host"))?,
            cfg.rms_norm_eps,
        ))
    } else {
        None
    };

    let output = if hosts_output {
        let lw: LinearWeight<B> = if cfg.tie_word_embeddings {
            let emb = embed_tensor
                .ok_or_else(|| anyhow::anyhow!("tied output needs embedding on the output host"))?;
            LinearWeight::dense(emb.swap_dims(0, 1))
        } else {
            weights
                .output_proj
                .ok_or_else(|| anyhow::anyhow!("untied output projection missing"))?
                .ensure_math_order()
        };
        Some(OutputProjection::new(lw))
    } else {
        None
    };

    let mut blocks: Vec<DecoderBlock<B>> = Vec::with_capacity(layer_range.len());
    for layer in weights.layers {
        let attn_norm = RmsNorm::new(layer.attn_norm, cfg.rms_norm_eps);
        let ffn_norm = RmsNorm::new(layer.ffn_norm, cfg.rms_norm_eps);
        let rope = RotaryEmbedding::new(
            cfg.head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            &device,
        );
        let attn = Attention::new(
            layer.q_proj.ensure_math_order(),
            layer.k_proj.ensure_math_order(),
            layer.v_proj.ensure_math_order(),
            layer.o_proj.ensure_math_order(),
            rope,
            cfg.n_heads,
            cfg.n_kv_heads,
            cfg.head_dim,
        );
        let ffn = SwiGluFfn::new(
            layer.ffn_gate.ensure_math_order(),
            layer.ffn_up.ensure_math_order(),
            layer.ffn_down.ensure_math_order(),
        );
        blocks.push(DecoderBlock {
            attn_norm,
            attn,
            ffn_norm,
            ffn,
        });
    }

    Ok(PeerStages {
        blocks,
        embedding,
        final_norm,
        output,
        cfg: cfg.clone(),
    })
}

/// Run one peer's serve loop over a single mesh `quinn::Connection`.
///
/// Accepts bidi streams and dispatches the first frame as a [`PeerFrame`]:
/// - `Coord(Embed)` — embedding host path: embed tokens, run blocks, reply
///   with `ActivationHeader` + activation bytes.
/// - `Relay(ActivationHeader)` — middle/tail hop: run blocks; if terminal run
///   `final_norm` + `output` and reply `PeerMsg::Logits`; otherwise forward
///   activation with a fresh header.
/// - `Coord(End)` — drop per-request KV caches.
fn make_caches<B: Backend>(
    batch: usize,
    n_blocks: usize,
    cfg: &ModelConfig,
    device: &B::Device,
) -> Vec<KvCacheSlot<B>>
where
    B::Device: Default,
{
    (0..n_blocks)
        .map(|_| {
            KvCacheSlot::<B>::new(
                batch,
                cfg.n_kv_heads,
                cfg.max_position_embeddings,
                cfg.head_dim,
                device,
            )
        })
        .collect()
}

pub async fn serve_peer<B>(conn: Connection, stages: PeerStages<B>) -> anyhow::Result<()>
where
    B: Backend,
    B::Device: Default,
{
    let device = B::Device::default();
    let cfg = &stages.cfg;
    let mut caches: HashMap<Uuid, Vec<KvCacheSlot<B>>> = HashMap::new();

    loop {
        let (mut send, mut recv) = match conn.accept_bi().await {
            Ok(s) => s,
            Err(_) => break,
        };

        let frame_bytes = read_frame(&mut recv).await?;
        let frame: PeerFrame = decode(&frame_bytes)?;

        match frame {
            // ── EMBEDDING HOST ────────────────────────────────────────────────
            PeerFrame::Coord(CoordinatorMsg::Embed {
                request_id,
                seq_pos,
                token_ids,
            }) => {
                let emb = stages.embedding.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("Embed frame sent to a non-embedding host")
                })?;

                let seq = token_ids.len();
                let ids = Tensor::<B, 2, Int>::from_data(
                    TensorData::new(token_ids, [1, seq]),
                    &device,
                );
                let mut x = emb.forward(ids);

                let start = seq_pos as usize;
                let positions: Vec<i32> = (start..start + seq).map(|p| p as i32).collect();

                let batch = x.dims()[0];
                let c = caches
                    .entry(request_id)
                    .or_insert_with(|| make_caches::<B>(batch, stages.blocks.len(), cfg, &device));

                for (block, cache) in stages.blocks.iter().zip(c.iter_mut()) {
                    x = block.forward(x, &positions, cache);
                }

                let (bytes, shape) = tensor_to_bytes(x)?;
                let hdr = ActivationHeader {
                    request_id,
                    seq_pos,
                    shape: [shape[0] as u32, shape[1] as u32, shape[2] as u32],
                    dtype: Dtype::F32,
                    // Single-node clusters (embed host == output host) are out of scope for v1;
                    // the coordinator always routes the tail via a terminal Relay.
                    is_terminal: false,
                };
                write_frame(&mut send, &encode(&hdr)?).await?;
                write_frame(&mut send, &bytes).await?;
                send.finish()?;
            }

            // ── MIDDLE / TAIL HOP ────────────────────────────────────────────
            PeerFrame::Relay(hdr) => {
                let payload = read_frame(&mut recv).await?;
                let shape = [
                    hdr.shape[0] as usize,
                    hdr.shape[1] as usize,
                    hdr.shape[2] as usize,
                ];
                let mut x = tensor_from_bytes::<B>(&payload, shape, &device)?;

                let start = hdr.seq_pos as usize;
                let positions: Vec<i32> =
                    (start..start + shape[1]).map(|p| p as i32).collect();

                let c = caches.entry(hdr.request_id).or_insert_with(|| {
                    make_caches::<B>(shape[0], stages.blocks.len(), cfg, &device)
                });

                for (block, cache) in stages.blocks.iter().zip(c.iter_mut()) {
                    x = block.forward(x, &positions, cache);
                }

                if hdr.is_terminal {
                    // Output host: apply final_norm + output projection, slice
                    // last token position, and return logits to the coordinator.
                    let fnorm = stages.final_norm.as_ref().ok_or_else(|| {
                        anyhow::anyhow!("is_terminal relay reached a non-output host")
                    })?;
                    let out_proj = stages.output.as_ref().ok_or_else(|| {
                        anyhow::anyhow!("output projection missing on output host")
                    })?;
                    let h = fnorm.forward(x);
                    let logits_full = out_proj.forward(h);
                    let last_seq = shape[1];
                    let logits = logits_full
                        .slice([0..1, (last_seq - 1)..last_seq, 0..cfg.vocab_size])
                        .reshape([cfg.vocab_size]);
                    let v: Vec<f32> = logits
                        .to_data()
                        .to_vec()
                        .map_err(|e| anyhow::anyhow!("to_vec: {e:?}"))?;
                    write_frame(
                        &mut send,
                        &encode(&PeerMsg::Logits {
                            request_id: hdr.request_id,
                            seq_pos: hdr.seq_pos,
                            logits: v,
                        })?,
                    )
                    .await?;
                    send.finish()?;
                } else {
                    // Middle hop: forward the activation to the next peer.
                    let (bytes, sh) = tensor_to_bytes(x)?;
                    let out_hdr = ActivationHeader {
                        shape: [sh[0] as u32, sh[1] as u32, sh[2] as u32],
                        ..hdr
                    };
                    write_frame(&mut send, &encode(&out_hdr)?).await?;
                    write_frame(&mut send, &bytes).await?;
                    send.finish()?;
                }
            }

            // ── END ──────────────────────────────────────────────────────────
            PeerFrame::Coord(CoordinatorMsg::End { request_id }) => {
                caches.remove(&request_id);
            }
        }
    }

    Ok(())
}
