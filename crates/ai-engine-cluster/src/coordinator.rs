//! Phase B (leaderless p2p): the coordinator role, decoupled from "leader".
//!
//! Any node that ingests a request becomes the *coordinator* for it. It drives
//! the forward pass over its full-mesh connections following the agreed
//! manifest: for each hop in [`PartitionManifest::forward_plan`], it either
//! runs that layer range in-process (the `local` step) or RPCs activations to
//! the peer holding it. Token ids start at the [`PartitionManifest::embedding_host`]
//! and logits come back from the [`PartitionManifest::output_host`] — neither of
//! which need be the coordinator.
//!
//! This module wires routing (pure, from the manifest) to transport (the mesh
//! `Connection`s). The burn forward-pass RPCs themselves are the next step.

use crate::partition::{PartitionManifest, Step};
use crate::protocol::{
    codec::{decode, encode},
    data::ActivationHeader,
    peer::{CoordinatorMsg, PeerFrame, PeerMsg},
};
use crate::transport::frame::{read_frame, write_frame};
use ai_engine_runtime::sample::{sample, SamplingConfig};
use quinn::Connection;
use std::collections::HashMap;
use uuid::Uuid;

/// Where a forward-pass hop executes.
pub enum Route<'a> {
    /// This hop runs in-process on the coordinator.
    Local,
    /// RPC to a mesh peer holding this layer range.
    Remote(&'a Connection),
}

pub struct Coordinator {
    local_id: String,
    manifest: PartitionManifest,
    /// Mesh connections to every *other* node, keyed by node_id.
    peers: HashMap<String, Connection>,
}

impl Coordinator {
    /// Build a coordinator for `local_id`. `peers` must contain a connection
    /// for every node in the manifest except `local_id` itself; otherwise
    /// [`Coordinator::missing_peers`] reports the gap and routing would fail.
    pub fn new(
        local_id: impl Into<String>,
        manifest: PartitionManifest,
        peers: HashMap<String, Connection>,
    ) -> Self {
        Self {
            local_id: local_id.into(),
            manifest,
            peers,
        }
    }

    pub fn manifest(&self) -> &PartitionManifest {
        &self.manifest
    }

    /// The ordered forward-pass plan this node executes (see
    /// [`PartitionManifest::forward_plan`]). `None` on a malformed manifest.
    pub fn plan(&self) -> Option<Vec<Step>> {
        self.manifest.forward_plan(&self.local_id)
    }

    /// Resolve a hop to either local execution or a peer connection.
    ///
    /// A node may register a loopback connection to *itself* (keyed by its own
    /// `local_id`) so that hops it hosts are driven over the same wire protocol
    /// as remote hops — in that case the self-connection wins and the hop is
    /// `Remote`. Without such a connection the hop is `Local` (v1 has no in-
    /// process stage execution on the coordinator).
    pub fn route(&self, node_id: &str) -> Option<Route<'_>> {
        if let Some(conn) = self.peers.get(node_id) {
            return Some(Route::Remote(conn));
        }
        if node_id == self.local_id {
            Some(Route::Local)
        } else {
            None
        }
    }

    /// Manifest nodes (other than this one) we lack a mesh connection to. Empty
    /// means the coordinator can route the entire pipeline; non-empty means the
    /// mesh is incomplete and this node cannot yet coordinate.
    pub fn missing_peers(&self) -> Vec<String> {
        self.manifest
            .assignments
            .iter()
            .map(|a| a.node_id.as_str())
            .filter(|id| *id != self.local_id && !self.peers.contains_key(*id))
            .map(String::from)
            .collect()
    }

    /// Resolve a hop to a peer connection; error on a local hop (v1 has no
    /// local stages) or a missing peer.
    fn conn_for(&self, id: &str) -> anyhow::Result<&Connection> {
        match self.route(id) {
            Some(Route::Remote(c)) => Ok(c),
            Some(Route::Local) => anyhow::bail!(
                "coordinator hop {id} is local; v1 coordinator holds no stages"
            ),
            None => anyhow::bail!("no mesh connection to {id}"),
        }
    }

    /// Drive prefill + token loop over the mesh. Returns generated token ids
    /// (excluding the prompt). Requires `missing_peers().is_empty()`.
    pub async fn generate(
        &self,
        prompt_ids: &[i32],
        max_tokens: usize,
        sampling: SamplingConfig,
    ) -> anyhow::Result<Vec<u32>> {
        let mut produced: Vec<u32> = Vec::with_capacity(max_tokens);
        self.drive(prompt_ids, max_tokens, sampling, |tok| {
            produced.push(tok);
            true
        })
        .await?;
        Ok(produced)
    }

    /// Streaming variant: spawn the forward loop and emit each sampled token id
    /// on the returned channel as soon as it's produced, so callers can stream
    /// SSE chunks instead of waiting for the whole sequence. On error an `Err`
    /// item is sent and the stream ends.
    pub fn generate_stream(
        self: std::sync::Arc<Self>,
        prompt_ids: Vec<i32>,
        max_tokens: usize,
        sampling: SamplingConfig,
    ) -> tokio::sync::mpsc::UnboundedReceiver<anyhow::Result<u32>> {
        // Unbounded so the synchronous `on_token` callback can send without
        // awaiting; bounded by `max_tokens`. A send error means the consumer
        // dropped, which stops the drive loop early.
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<anyhow::Result<u32>>();
        tokio::spawn(async move {
            let send_tx = tx.clone();
            let result = self
                .drive(&prompt_ids, max_tokens, sampling, move |tok| {
                    send_tx.send(Ok(tok)).is_ok()
                })
                .await;
            if let Err(e) = result {
                let _ = tx.send(Err(e));
            }
        });
        rx
    }

    /// Shared forward loop: prefill at pos 0, then one token per step. Invokes
    /// `on_token(token)` for each sampled token; if it returns `false`, driving
    /// stops early (consumer gone). Requires `missing_peers().is_empty()`.
    async fn drive(
        &self,
        prompt_ids: &[i32],
        max_tokens: usize,
        sampling: SamplingConfig,
        mut on_token: impl FnMut(u32) -> bool,
    ) -> anyhow::Result<()> {
        let missing = self.missing_peers();
        if !missing.is_empty() {
            anyhow::bail!("coordinator missing peers: {:?}", missing);
        }
        let plan = self
            .plan()
            .ok_or_else(|| anyhow::anyhow!("malformed manifest: no pipeline order"))?;
        let ids: Vec<String> = plan.iter().map(|s| s.node_id.clone()).collect();
        if ids.len() < 2 {
            anyhow::bail!(
                "single-node clusters are out of scope for v1 (need >= 2 pipeline nodes)"
            );
        }
        let request_id = Uuid::now_v7();

        // Prefill at pos 0.
        let first = self
            .forward_pass(request_id, &ids, prompt_ids.to_vec(), 0)
            .await?;
        let mut last = sample(&first, &sampling);
        let mut produced = 1usize;
        if !on_token(last) {
            return Ok(());
        }

        // Token loop: pos = prompt length + number of tokens already produced.
        for _ in 1..max_tokens {
            let pos = prompt_ids.len() + produced - 1;
            let logits = self
                .forward_pass(request_id, &ids, vec![last as i32], pos)
                .await?;
            last = sample(&logits, &sampling);
            produced += 1;
            if !on_token(last) {
                return Ok(());
            }
        }
        Ok(())
    }

    /// ids[0] = head (embedding), ids[last] = tail (output); pos is the absolute KV-cache position for this step.
    ///
    /// One forward pass: Embed at the head, relay through the middle, terminal
    /// relay at the tail → logits.
    async fn forward_pass(
        &self,
        request_id: Uuid,
        ids: &[String],
        tokens: Vec<i32>,
        pos: usize,
    ) -> anyhow::Result<Vec<f32>> {
        // Head: Embed → activation.
        let head = self.conn_for(&ids[0])?;
        let (mut s, mut r) = head.open_bi().await?;
        write_frame(
            &mut s,
            &encode(&PeerFrame::Coord(CoordinatorMsg::Embed {
                request_id,
                seq_pos: pos as u32,
                token_ids: tokens,
            }))?,
        )
        .await?;
        s.finish()?;
        let mut hdr: ActivationHeader = decode(&read_frame(&mut r).await?)?;
        let mut payload = read_frame(&mut r).await?;

        // Middle hops (everything between head and tail).
        for mid in &ids[1..ids.len() - 1] {
            let c = self.conn_for(mid)?;
            let (mut s, mut r) = c.open_bi().await?;
            write_frame(&mut s, &encode(&PeerFrame::Relay(hdr.clone()))?).await?;
            write_frame(&mut s, &payload).await?;
            s.finish()?;
            hdr = decode(&read_frame(&mut r).await?)?;
            payload = read_frame(&mut r).await?;
        }

        // Tail: terminal relay → Logits.
        let tail = self.conn_for(&ids[ids.len() - 1])?;
        let mut term = hdr.clone();
        term.is_terminal = true;
        let (mut s, mut r) = tail.open_bi().await?;
        write_frame(&mut s, &encode(&PeerFrame::Relay(term))?).await?;
        write_frame(&mut s, &payload).await?;
        s.finish()?;
        match decode::<PeerMsg>(&read_frame(&mut r).await?)? {
            PeerMsg::Logits { logits, .. } => Ok(logits),
            PeerMsg::Fault { detail, .. } => anyhow::bail!("peer fault: {detail}"),
        }
    }
}
