use crate::capability::Capability;
use crate::partition::{auto_partition, manual_partition, PartitionManifest};
use crate::protocol::codec::{decode, encode};
use crate::protocol::control::{LeaderToWorker, WorkerToLeader};
use crate::protocol::data::{ActivationHeader, Dtype};
use crate::session::{LeaderModel, RequestSession};
use crate::tensor_io::{tensor_from_bytes, tensor_to_bytes};
use crate::tls::NodeIdentity;
use crate::transport::frame::{read_frame, write_frame};
use crate::transport::quic::client_endpoint;
use ai_engine_runtime::arch::attention::Attention;
use ai_engine_runtime::arch::block::DecoderBlock;
use ai_engine_runtime::arch::embedding::{OutputProjection, TokenEmbedding};
use ai_engine_runtime::arch::ffn::SwiGluFfn;
use ai_engine_runtime::arch::rmsnorm::RmsNorm;
use ai_engine_runtime::arch::rope::RotaryEmbedding;
use ai_engine_runtime::config::ModelConfig;
use ai_engine_runtime::loader::load_range;
use ai_engine_runtime::sample::{sample, SamplingConfig};
use burn::tensor::{backend::Backend, Int, Tensor, TensorData};
use std::net::SocketAddr;
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

/// A worker the leader should dial during startup.
#[derive(Debug, Clone)]
pub struct WorkerEndpoint {
    pub node_id: String,
    pub addr: SocketAddr,
    pub fingerprint: String,
}

/// Inputs to `ClusterLeader::start`.
#[derive(Debug, Clone)]
pub struct LeaderConfig {
    pub cluster_id: String,
    pub leader_node_id: String,
    pub model_id: String,
    pub n_layers: usize,
    pub layer_bytes: u64,
    pub embed_output_bytes: u64,
    pub per_node_overhead: u64,
    pub workers: Vec<WorkerEndpoint>,
    /// Optional explicit partition. When `Some`, bypasses `auto_partition`
    /// and uses `manual_partition` with the provided node/range pairs.
    pub partition_override: Option<Vec<(String, Range<usize>)>>,
}

/// Per-worker connection state owned by the leader after the join handshake.
///
/// `control_send` / `control_recv` are kept open so subsequent control
/// frames (e.g. `Assignment`) can be sent on the same bidi stream.
pub struct WorkerConnection {
    pub node_id: String,
    pub conn: quinn::Connection,
    pub control_send: quinn::SendStream,
    pub control_recv: quinn::RecvStream,
}

/// Leader after startup: workers joined, capabilities collected, manifest computed.
///
/// Plan 4 Task 4 made `ClusterLeader` immutable from the inference path's
/// perspective: per-request state lives in `RequestSession`. `generate` is
/// `&self` and multiple sessions can drive the same cluster concurrently.
pub struct ClusterLeader {
    manifest: PartitionManifest,
    connections: Vec<WorkerConnection>,
}

impl ClusterLeader {
    /// Connect to every worker in `cfg.workers`, run the Join handshake, collect
    /// `Capability` advertisements, then compute an auto-partition manifest.
    ///
    /// Sequential per-worker dial keeps things simple — v0.2 clusters are small
    /// (≤ 8 nodes) and startup is one-shot.
    pub async fn start(identity: &NodeIdentity, cfg: LeaderConfig) -> anyhow::Result<Self> {
        let fingerprints: Vec<String> =
            cfg.workers.iter().map(|w| w.fingerprint.clone()).collect();
        let endpoint = client_endpoint(identity, &fingerprints)?;

        let mut connections: Vec<WorkerConnection> = Vec::with_capacity(cfg.workers.len());
        let mut capabilities: Vec<Capability> = Vec::with_capacity(cfg.workers.len());

        for w in &cfg.workers {
            let conn = endpoint
                .connect(w.addr, &w.node_id)
                .map_err(|e| anyhow::anyhow!("connect {}: {e}", w.node_id))?
                .await
                .map_err(|e| anyhow::anyhow!("handshake {}: {e}", w.node_id))?;

            let (mut send, mut recv) = conn.open_bi().await?;

            // 1. Send Join.
            let join = LeaderToWorker::Join {
                cluster_id: cfg.cluster_id.clone(),
                protocol_version: 1,
                leader_node_id: cfg.leader_node_id.clone(),
            };
            write_frame(&mut send, &encode(&join)?).await?;

            // 2. Read JoinAck.
            let ack_bytes = read_frame(&mut recv).await?;
            let ack: WorkerToLeader = decode(&ack_bytes)?;
            match ack {
                WorkerToLeader::JoinAck { node_id, .. } => {
                    if node_id != w.node_id {
                        anyhow::bail!(
                            "worker {} reported node_id {node_id} in JoinAck",
                            w.node_id
                        );
                    }
                }
                other => anyhow::bail!("expected JoinAck from {}, got {other:?}", w.node_id),
            }

            // 3. Read Capability.
            let cap_bytes = read_frame(&mut recv).await?;
            let cap_msg: WorkerToLeader = decode(&cap_bytes)?;
            let cap = match cap_msg {
                WorkerToLeader::Capability(c) => c,
                other => anyhow::bail!(
                    "expected Capability from {}, got {other:?}",
                    w.node_id
                ),
            };
            capabilities.push(cap);

            connections.push(WorkerConnection {
                node_id: w.node_id.clone(),
                conn,
                control_send: send,
                control_recv: recv,
            });
        }

        let manifest = if let Some(ranges) = cfg.partition_override.clone() {
            manual_partition(
                &cfg.model_id,
                &capabilities,
                cfg.n_layers,
                ranges,
                cfg.layer_bytes,
                cfg.embed_output_bytes,
                cfg.per_node_overhead,
            )?
        } else {
            auto_partition(
                &cfg.model_id,
                &capabilities,
                cfg.n_layers,
                cfg.layer_bytes,
                cfg.embed_output_bytes,
                cfg.per_node_overhead,
            )?
        };

        // Phase 3: distribute Assignment to each worker over the existing
        // control bidi stream.
        for wc in connections.iter_mut() {
            let assignment = LeaderToWorker::Assignment {
                manifest: manifest.clone(),
                model_id: cfg.model_id.clone(),
            };
            write_frame(&mut wc.control_send, &encode(&assignment)?).await?;
        }

        Ok(Self {
            manifest,
            connections,
        })
    }

    pub fn manifest(&self) -> &PartitionManifest {
        &self.manifest
    }

    /// Build a fresh per-request session. Loads the leader's portion of the
    /// model weights from disk (embedding, final norm, output projection,
    /// and the leader's own layer range), clones the worker
    /// `quinn::Connection`s, and allocates per-block KV caches.
    ///
    /// The returned session can be driven through `step_through_cluster_session`
    /// to run prefill + token-by-token generation. Multiple sessions can
    /// coexist on the same `ClusterLeader`.
    pub async fn build_session<B>(
        &self,
        model_path: &Path,
        cfg: &ModelConfig,
        leader_layers: Range<usize>,
    ) -> anyhow::Result<RequestSession<B>>
    where
        B: Backend,
        B::Device: Default,
    {
        let leader_model = build_leader_model::<B>(model_path, cfg, leader_layers)?;
        let worker_conns: Vec<quinn::Connection> =
            self.connections.iter().map(|wc| wc.conn.clone()).collect();
        let device = B::Device::default();
        Ok(RequestSession::new(
            Arc::new(leader_model),
            worker_conns,
            &device,
        ))
    }

    /// Autoregressive greedy/sampled generation through the cluster.
    /// Returns the generated token ids (not including the prompt).
    ///
    /// Flow:
    ///   1. Build a `RequestSession` (loads leader-side weights, allocates
    ///      KV caches, clones worker connections, mints a fresh `request_id`).
    ///   2. Prefill: feed the prompt through embedding → leader blocks →
    ///      every worker → final_norm + output proj → sample first token.
    ///   3. Token loop: embed single token at `current_pos`, run leader blocks
    ///      (advancing KV by one position), send 1-token activations to each
    ///      worker, receive back, final_norm + output, sample next.
    pub async fn generate<B>(
        &self,
        model_path: &Path,
        cfg: &ModelConfig,
        leader_layers: Range<usize>,
        prompt_ids: &[i32],
        max_tokens: usize,
        sampling: SamplingConfig,
    ) -> anyhow::Result<Vec<u32>>
    where
        B: Backend,
        B::Device: Default,
    {
        let mut session: RequestSession<B> = self
            .build_session(model_path, cfg, leader_layers)
            .await?;

        let mut produced: Vec<u32> = Vec::with_capacity(max_tokens);

        // Prefill.
        let last_logits =
            step_through_cluster_session(&mut session, prompt_ids, false).await?;
        session.current_pos = prompt_ids.len();
        produced.push(sample(&last_logits, &sampling));

        // Token loop.
        for _ in 1..max_tokens {
            let last_token = *produced.last().unwrap() as i32;
            let last_logits =
                step_through_cluster_session(&mut session, &[last_token], false).await?;
            session.current_pos += 1;
            produced.push(sample(&last_logits, &sampling));
        }

        // We don't send a terminal frame here — when the connections drop
        // at the end of the test (or request lifetime), workers exit
        // their accept loop gracefully via EOF and free their caches.
        Ok(produced)
    }
}

/// Load the leader-side model artifacts (embedding, final_norm, output proj,
/// and the leader's own decoder blocks) from disk for backend `B`.
///
/// Hoisted out of `ClusterLeader` so that `LeaderModel<B>` can be generic
/// over the backend without forcing `ClusterLeader` itself to be generic.
fn build_leader_model<B>(
    model_path: &Path,
    cfg: &ModelConfig,
    leader_layers: Range<usize>,
) -> anyhow::Result<LeaderModel<B>>
where
    B: Backend,
    B::Device: Default,
{
    let device = B::Device::default();
    let weights = load_range::<B>(model_path, cfg, leader_layers.clone(), true, true, &device)?;

    let embed_tensor = weights
        .embedding
        .ok_or_else(|| anyhow::anyhow!("embedding required for leader"))?;
    let embedding = TokenEmbedding::new(embed_tensor.clone());
    let final_norm = RmsNorm::new(
        weights
            .final_norm
            .ok_or_else(|| anyhow::anyhow!("final_norm required for leader"))?,
        cfg.rms_norm_eps,
    );
    let output_weight = if cfg.tie_word_embeddings {
        embed_tensor.swap_dims(0, 1)
    } else {
        weights
            .output_proj
            .ok_or_else(|| anyhow::anyhow!("untied output projection missing"))?
            .swap_dims(0, 1)
    };
    let output = OutputProjection::new(output_weight);

    let mut blocks: Vec<DecoderBlock<B>> = Vec::with_capacity(leader_layers.len());
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

    Ok(LeaderModel {
        embedding,
        blocks,
        final_norm,
        output,
        cfg: cfg.clone(),
    })
}

/// Run ONE forward step against the current `RequestSession`. Used for both
/// prefill (seq = prompt_len) and per-token generation (seq = 1).
///
/// Mutates `session.leader_caches` (KV state advances), but does not bump
/// `session.current_pos` — the caller does that after sampling.
pub async fn step_through_cluster_session<B>(
    session: &mut RequestSession<B>,
    token_ids: &[i32],
    is_terminal: bool,
) -> anyhow::Result<Vec<f32>>
where
    B: Backend,
    B::Device: Default,
{
    let device = B::Device::default();
    let cfg = &session.model.cfg;
    let seq = token_ids.len();
    let ids = Tensor::<B, 2, Int>::from_data(
        TensorData::new(token_ids.to_vec(), [1, seq]),
        &device,
    );
    let mut x = session.model.embedding.forward(ids);
    let positions: Vec<i32> = ((session.current_pos as i32)
        ..((session.current_pos + seq) as i32))
        .collect();
    for (block, cache) in session
        .model
        .blocks
        .iter()
        .zip(session.leader_caches.iter_mut())
    {
        x = block.forward(x, &positions, cache);
    }

    // Send through each worker in order, waiting for response between hops.
    // Each request opens its own bidi stream pair per hop (Task 5 will switch
    // the worker side to accept_bi); for now this still uses uni streams
    // matching the existing worker loop.
    let request_id = session.request_id;
    let start_pos = session.current_pos;
    for conn in &session.worker_conns {
        let (bytes, shape) = tensor_to_bytes(x)?;
        let header = ActivationHeader {
            request_id,
            seq_pos: start_pos as u32,
            shape: [shape[0] as u32, shape[1] as u32, shape[2] as u32],
            dtype: Dtype::F32,
            is_terminal,
        };
        let mut send_uni = conn.open_uni().await?;
        write_frame(&mut send_uni, &encode(&header)?).await?;
        write_frame(&mut send_uni, &bytes).await?;
        send_uni.finish()?;

        let mut recv_uni = conn.accept_uni().await?;
        let _hdr: ActivationHeader = decode(&read_frame(&mut recv_uni).await?)?;
        let payload = read_frame(&mut recv_uni).await?;
        let shape_back = [shape[0], shape[1], shape[2]];
        x = tensor_from_bytes::<B>(&payload, shape_back, &device)?;
    }

    // Final norm + output projection. Slice last position.
    let x = session.model.final_norm.forward(x);
    let logits = session.model.output.forward(x);
    let last = logits
        .slice([0..1, (seq - 1)..seq, 0..cfg.vocab_size])
        .reshape([cfg.vocab_size]);
    last.to_data()
        .to_vec()
        .map_err(|e| anyhow::anyhow!("to_vec: {e:?}"))
}
