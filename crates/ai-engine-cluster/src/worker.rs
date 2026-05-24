use crate::capability::{detect_capability, BackendKind};
use crate::protocol::codec::{decode, encode};
use crate::protocol::control::{LeaderToWorker, WorkerToLeader};
use crate::protocol::data::{ActivationHeader, Dtype};
use crate::tensor_io::{tensor_from_bytes, tensor_to_bytes};
use crate::transport::frame::{read_frame, write_frame};
use ai_engine_runtime::arch::attention::Attention;
use ai_engine_runtime::arch::block::DecoderBlock;
use ai_engine_runtime::arch::ffn::SwiGluFfn;
use ai_engine_runtime::arch::rmsnorm::RmsNorm;
use ai_engine_runtime::arch::rope::RotaryEmbedding;
use ai_engine_runtime::config::ModelConfig;
use ai_engine_runtime::kv_cache::KvCacheSlot;
use ai_engine_runtime::loader::load_range;
use burn::tensor::backend::Backend;
use quinn::Endpoint;
use std::collections::HashMap;
use std::path::PathBuf;
use uuid::Uuid;

/// Accept the leader's connection, perform the join handshake, and return
/// once the connection drops.
///
/// v0.2 scope (Task 8): one-shot handshake only.
/// 1. Accept the leader's inbound connection.
/// 2. Read `Join` from the bidi control stream.
/// 3. Reply with `JoinAck` then `Capability`.
/// 4. Wait for the leader to close the connection (Task 9/10 will extend this
///    to read `Assignment` and enter the request-serving loop).
///
/// Extension point for Task 10: after step 4, read `Assignment`, load weights
/// via `ai_engine_runtime::loader::load_range`, and call into the inference
/// loop (`run_worker_full`).
pub async fn run_worker_handshake(
    endpoint: Endpoint,
    node_id: String,
    backend: BackendKind,
) -> anyhow::Result<()> {
    let incoming = endpoint
        .accept()
        .await
        .ok_or_else(|| anyhow::anyhow!("no incoming connection"))?;
    let conn = incoming.await?;
    let (mut send, mut recv) = conn.accept_bi().await?;

    // Step 2: receive Join.
    let join_bytes = read_frame(&mut recv).await?;
    let _join: LeaderToWorker = decode(&join_bytes)?;

    // Step 3a: send JoinAck.
    let ack = WorkerToLeader::JoinAck {
        node_id: node_id.clone(),
        // Populated from the real cert DER in Plan 3 binary integration.
        certificate_sha256: [0u8; 32],
    };
    write_frame(&mut send, &encode(&ack)?).await?;

    // Step 3b: send Capability.
    let cap = detect_capability(&node_id, backend, 0, None)?;
    write_frame(&mut send, &encode(&WorkerToLeader::Capability(cap))?).await?;

    // Step 4: idle until the leader closes the connection.
    // Task 10 will replace this with Assignment handling + inference loop.
    let _ = conn.closed().await;
    Ok(())
}

/// Run a full worker: handshake, load its assigned layer range, then service
/// activation frames in a loop until the connection closes.
///
/// For each inbound bidi stream the worker reads `(ActivationHeader, payload)`
/// on the recv side, runs its blocks over the activations, then writes
/// `(ActivationHeader, payload)` back on the same stream's send side. Using
/// bidi streams (Plan 4 Task 5) instead of paired uni streams naturally pairs
/// each request with its response without any demultiplexing — quinn handles
/// concurrent bidi streams from the same connection independently, which is
/// what enables concurrent requests through one leader.
///
/// Per-request KV caches are kept in a `HashMap<Uuid, _>` keyed by the
/// `request_id` in the activation header, and freed when `is_terminal=true`
/// arrives.
pub async fn run_worker_full<B>(
    endpoint: Endpoint,
    node_id: String,
    backend: BackendKind,
    model_path: PathBuf,
    cfg: ModelConfig,
) -> anyhow::Result<()>
where
    B: Backend,
    B::Device: Default,
{
    let device = B::Device::default();

    // Handshake: same flow as run_worker_handshake.
    let incoming = endpoint
        .accept()
        .await
        .ok_or_else(|| anyhow::anyhow!("no incoming connection"))?;
    let conn = incoming.await?;
    let (mut send, mut recv) = conn.accept_bi().await?;

    let _join: LeaderToWorker = decode(&read_frame(&mut recv).await?)?;
    let ack = WorkerToLeader::JoinAck {
        node_id: node_id.clone(),
        certificate_sha256: [0u8; 32],
    };
    write_frame(&mut send, &encode(&ack)?).await?;
    let cap = detect_capability(&node_id, backend, 0, None)?;
    write_frame(&mut send, &encode(&WorkerToLeader::Capability(cap))?).await?;

    // Wait for the leader's Assignment, then extract this worker's layer range.
    let assn_bytes = read_frame(&mut recv).await?;
    let assn: LeaderToWorker = decode(&assn_bytes)?;
    let layer_range = match assn {
        LeaderToWorker::Assignment { manifest, .. } => manifest
            .for_node(&node_id)
            .ok_or_else(|| anyhow::anyhow!("no assignment for {node_id} in manifest"))?
            .layer_range
            .clone(),
        other => anyhow::bail!("expected Assignment, got {other:?}"),
    };

    // Load this worker's layer range from disk.
    let weights = load_range::<B>(&model_path, &cfg, layer_range.clone(), false, false, &device)?;

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
            layer.q_proj.swap_dims(0, 1),
            layer.k_proj.swap_dims(0, 1),
            layer.v_proj.swap_dims(0, 1),
            layer.o_proj.swap_dims(0, 1),
            rope,
            cfg.n_heads,
            cfg.n_kv_heads,
            cfg.head_dim,
        );
        let ffn = SwiGluFfn::new(
            layer.ffn_gate.swap_dims(0, 1),
            layer.ffn_up.swap_dims(0, 1),
            layer.ffn_down.swap_dims(0, 1),
        );
        blocks.push(DecoderBlock {
            attn_norm,
            attn,
            ffn_norm,
            ffn,
        });
    }

    // Per-request KV caches (one Vec<KvCacheSlot> per request_id, freed on
    // is_terminal=true).
    let mut request_caches: HashMap<Uuid, Vec<KvCacheSlot<B>>> = HashMap::new();

    loop {
        let (mut send_bi, mut recv_bi) = match conn.accept_bi().await {
            Ok(streams) => streams,
            Err(_) => break, // connection closed
        };
        let header_bytes = read_frame(&mut recv_bi).await?;
        let header: ActivationHeader = decode(&header_bytes)?;
        let payload_bytes = read_frame(&mut recv_bi).await?;

        let shape = [
            header.shape[0] as usize,
            header.shape[1] as usize,
            header.shape[2] as usize,
        ];
        let mut x = tensor_from_bytes::<B>(&payload_bytes, shape, &device)?;

        let caches = request_caches
            .entry(header.request_id)
            .or_insert_with(|| {
                (0..blocks.len())
                    .map(|_| {
                        KvCacheSlot::<B>::new(
                            shape[0],
                            cfg.n_kv_heads,
                            cfg.max_position_embeddings,
                            cfg.head_dim,
                            &device,
                        )
                    })
                    .collect()
            });

        let start_pos = header.seq_pos as usize;
        let positions: Vec<i32> = (start_pos..start_pos + shape[1])
            .map(|p| p as i32)
            .collect();
        for (block, cache) in blocks.iter().zip(caches.iter_mut()) {
            x = block.forward(x, &positions, cache);
        }

        // Send the processed activations back on the same bidi stream's send
        // side. quinn naturally pairs the request and response — no separate
        // accept needed on the leader.
        let (out_bytes, out_shape) = tensor_to_bytes(x)?;
        let out_header = ActivationHeader {
            request_id: header.request_id,
            seq_pos: header.seq_pos,
            shape: [
                out_shape[0] as u32,
                out_shape[1] as u32,
                out_shape[2] as u32,
            ],
            dtype: Dtype::F32,
            is_terminal: header.is_terminal,
        };
        write_frame(&mut send_bi, &encode(&out_header)?).await?;
        write_frame(&mut send_bi, &out_bytes).await?;
        send_bi.finish()?;

        if header.is_terminal {
            request_caches.remove(&header.request_id);
        }
    }
    Ok(())
}
