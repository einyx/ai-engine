//! Full-mesh QUIC transport — Phase A of leaderless p2p.
//!
//! In the star model, workers accept exactly one inbound connection (the
//! leader's) and a non-leader node has no peer connections, so it can never
//! act as a request entry point. A mesh fixes that: every node holds a
//! connection to every other node, so any node can orchestrate a forward pass
//! (Phase B) over the ring.
//!
//! One connection per pair is guaranteed by a deterministic rule: the node
//! with the lexicographically **smaller** `node_id` dials; the larger accepts.
//! Right after connecting, the dialer sends a one-frame hello carrying its
//! `node_id` so the accepting side can key the connection — the server uses
//! `no_client_auth`, so identity comes from the hello, not the TLS layer.

use crate::tls::NodeIdentity;
use crate::transport::frame::{read_frame, write_frame};
use crate::transport::quic::{client_config, server_endpoint};
use quinn::{Connection, Endpoint};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::Duration;

/// Window a dialer tolerates a peer not yet being bound/accepting. Cluster
/// nodes start as independent processes, so a lower-id dialer can race ahead
/// of a higher-id peer's bind; quinn surfaces that as a handshake error.
const DIAL_ATTEMPTS: usize = 20;
const DIAL_BACKOFF: Duration = Duration::from_millis(250);

/// Dial `addr`, retrying connect+handshake through [`DIAL_ATTEMPTS`] so a
/// late-binding peer converges instead of failing the whole mesh.
async fn dial_with_retry(
    endpoint: &Endpoint,
    addr: SocketAddr,
    sni: &str,
) -> anyhow::Result<Connection> {
    let mut last_err: Option<anyhow::Error> = None;
    for attempt in 0..DIAL_ATTEMPTS {
        match endpoint.connect(addr, sni) {
            Ok(connecting) => match connecting.await {
                Ok(conn) => return Ok(conn),
                Err(e) => last_err = Some(anyhow::anyhow!("handshake {sni}: {e}")),
            },
            Err(e) => last_err = Some(anyhow::anyhow!("dial {sni}: {e}")),
        }
        if attempt + 1 < DIAL_ATTEMPTS {
            tokio::time::sleep(DIAL_BACKOFF).await;
        }
    }
    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("dial {sni}: exhausted retries")))
}

/// A reachable cluster peer.
#[derive(Clone, Debug)]
pub struct Peer {
    pub node_id: String,
    pub addr: SocketAddr,
    pub fingerprint: String,
}

/// Build the dual-purpose endpoint a mesh node uses: it serves (accepts
/// inbound peers) *and* carries a default client config (dials outbound
/// peers), trusting every peer fingerprint.
pub fn mesh_endpoint(
    identity: &NodeIdentity,
    bind: SocketAddr,
    peer_fingerprints: &[String],
) -> anyhow::Result<Endpoint> {
    let mut endpoint = server_endpoint(identity, bind)?;
    endpoint.set_default_client_config(client_config(identity, peer_fingerprints)?);
    Ok(endpoint)
}

/// Establish a full mesh from `endpoint` to every peer in `peers`, returning a
/// map of `peer node_id -> Connection`. Dials peers with a larger `node_id`
/// and accepts inbound from peers with a smaller one, so each pair forms
/// exactly one connection regardless of which side calls first.
///
/// All peers must be running `connect_mesh` (or at least accepting) for this to
/// converge; it returns once every pair is connected.
pub async fn connect_mesh(
    endpoint: &Endpoint,
    local_id: &str,
    peers: &[Peer],
) -> anyhow::Result<HashMap<String, Connection>> {
    let to_accept = peers
        .iter()
        .filter(|p| p.node_id.as_str() < local_id)
        .count();
    let to_dial: Vec<Peer> = peers
        .iter()
        .filter(|p| p.node_id.as_str() > local_id)
        .cloned()
        .collect();

    // Accept inbound peers concurrently with our own dials: a pair (A<B) has A
    // dialing while B is in its accept loop, so both must run at once.
    let accept_ep = endpoint.clone();
    let accept_handle = tokio::spawn(async move {
        let mut map: HashMap<String, Connection> = HashMap::new();
        for _ in 0..to_accept {
            let incoming = accept_ep
                .accept()
                .await
                .ok_or_else(|| anyhow::anyhow!("mesh: endpoint closed during accept"))?;
            let conn = incoming.await?;
            let (_send, mut recv) = conn.accept_bi().await?;
            let peer_id = String::from_utf8(read_frame(&mut recv).await?)
                .map_err(|e| anyhow::anyhow!("mesh hello: bad utf8 node_id: {e}"))?;
            map.insert(peer_id, conn);
        }
        Ok::<_, anyhow::Error>(map)
    });

    let mut conns: HashMap<String, Connection> = HashMap::new();
    for p in &to_dial {
        let conn = dial_with_retry(endpoint, p.addr, &p.node_id).await?;
        let (mut send, _recv) = conn.open_bi().await?;
        write_frame(&mut send, local_id.as_bytes()).await?;
        let _ = send.finish();
        conns.insert(p.node_id.clone(), conn);
    }

    let accepted = accept_handle
        .await
        .map_err(|e| anyhow::anyhow!("mesh accept task: {e}"))??;
    conns.extend(accepted);

    if conns.len() != peers.len() {
        anyhow::bail!(
            "mesh incomplete: connected {} of {} peers",
            conns.len(),
            peers.len()
        );
    }
    Ok(conns)
}

/// Establish a loopback connection from a node to its own endpoint.
///
/// In leaderless mode the coordinator (the ingress node) is itself a pipeline
/// node, so driving a forward pass requires reaching its own hosted stages.
/// Rather than special-casing in-process execution, the node dials itself: the
/// returned [`Connection`] is keyed by the node's own id and behaves like any
/// peer connection. The accepting side of this loopback is the node's own
/// `serve_peer` loop (the caller must accept one extra inbound connection and
/// spawn that loop). The node's own fingerprint must be trusted by the
/// endpoint's client config (see [`mesh_endpoint`]).
pub async fn connect_self(
    endpoint: &Endpoint,
    local_id: &str,
    bind: SocketAddr,
) -> anyhow::Result<Connection> {
    // Loopback dials must target a routable address, not 0.0.0.0.
    let dial_addr: SocketAddr = if bind.ip().is_unspecified() {
        SocketAddr::new(std::net::Ipv4Addr::LOCALHOST.into(), bind.port())
    } else {
        bind
    };
    let conn = dial_with_retry(endpoint, dial_addr, local_id).await?;
    let (mut send, _recv) = conn.open_bi().await?;
    write_frame(&mut send, local_id.as_bytes()).await?;
    let _ = send.finish();
    Ok(conn)
}

/// Accept a single inbound connection (used to accept the loopback dial in
/// [`connect_self`]) and return it keyed by the hello-advertised node id.
pub async fn accept_one(endpoint: &Endpoint) -> anyhow::Result<(String, Connection)> {
    let incoming = endpoint
        .accept()
        .await
        .ok_or_else(|| anyhow::anyhow!("mesh: endpoint closed during accept"))?;
    let conn = incoming.await?;
    let (_send, mut recv) = conn.accept_bi().await?;
    let peer_id = String::from_utf8(read_frame(&mut recv).await?)
        .map_err(|e| anyhow::anyhow!("mesh hello: bad utf8 node_id: {e}"))?;
    Ok((peer_id, conn))
}
