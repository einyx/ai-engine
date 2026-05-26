//! Full-mesh transport (Phase A of leaderless p2p): N nodes each bind an
//! endpoint, then concurrently form a complete connection graph — every node
//! ends up with a connection to every other node, with exactly one connection
//! per pair.

use ai_engine_cluster::tls::generate_node_identity;
use ai_engine_cluster::transport::mesh::{connect_mesh, mesh_endpoint, Peer};
use std::net::SocketAddr;

#[tokio::test(flavor = "multi_thread")]
async fn three_nodes_form_complete_mesh() {
    let ids = ["node-a", "node-b", "node-c"];
    let identities: Vec<_> = ids
        .iter()
        .map(|id| generate_node_identity(id).unwrap())
        .collect();

    // Bind every node first so we know each addr before anyone dials.
    let bind: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let mut endpoints = Vec::new();
    let mut addrs = Vec::new();
    for (i, _) in ids.iter().enumerate() {
        let other_fps: Vec<String> = identities
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, idn)| idn.fingerprint.clone())
            .collect();
        let ep = mesh_endpoint(&identities[i], bind, &other_fps).unwrap();
        addrs.push(ep.local_addr().unwrap());
        endpoints.push(ep);
    }

    // Each node's peer list = every other node.
    let peers_for = |i: usize| -> Vec<Peer> {
        ids.iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(j, id)| Peer {
                node_id: id.to_string(),
                addr: addrs[j],
                fingerprint: identities[j].fingerprint.clone(),
            })
            .collect()
    };

    // Run all three concurrently — dials and accepts interleave across nodes.
    let mut handles = Vec::new();
    for (i, id) in ids.iter().enumerate() {
        let ep = endpoints[i].clone();
        let local = id.to_string();
        let peers = peers_for(i);
        handles.push(tokio::spawn(async move {
            connect_mesh(&ep, &local, &peers).await
        }));
    }

    for (i, h) in handles.into_iter().enumerate() {
        let conns = h.await.unwrap().unwrap();
        // Each node connects to the other two, by their real node_ids.
        assert_eq!(conns.len(), 2, "node {} should have 2 peers", ids[i]);
        for (j, peer_id) in ids.iter().enumerate() {
            if j != i {
                assert!(
                    conns.contains_key(*peer_id),
                    "node {} missing connection to {}",
                    ids[i],
                    peer_id
                );
            }
        }
    }
}

// A lower-id dialer that starts before the higher-id peer has bound must retry
// until the peer comes up, rather than bailing the whole mesh.
#[tokio::test(flavor = "multi_thread")]
async fn dial_retries_until_late_peer_binds() {
    let id_a = generate_node_identity("node-a").unwrap();
    let id_b = generate_node_identity("node-b").unwrap();

    // Reserve then immediately free a UDP port so node-b can bind it ~1s later;
    // node-a starts dialing this address while nothing is listening yet.
    let b_addr: SocketAddr = {
        let s = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
        s.local_addr().unwrap()
    };

    let ep_a = mesh_endpoint(&id_a, "127.0.0.1:0".parse().unwrap(), &[id_b.fingerprint.clone()])
        .unwrap();
    let a_addr = ep_a.local_addr().unwrap();
    let fp_b = id_b.fingerprint.clone();
    let a_handle = tokio::spawn(async move {
        let peers = vec![Peer { node_id: "node-b".into(), addr: b_addr, fingerprint: fp_b }];
        connect_mesh(&ep_a, "node-a", &peers).await
    });

    // node-b binds late on the reserved port and accepts node-a's retried dial.
    tokio::time::sleep(std::time::Duration::from_millis(1200)).await;
    let ep_b = mesh_endpoint(&id_b, b_addr, &[id_a.fingerprint.clone()]).unwrap();
    let peers_b = vec![Peer {
        node_id: "node-a".into(),
        addr: a_addr,
        fingerprint: id_a.fingerprint.clone(),
    }];
    let conns_b = connect_mesh(&ep_b, "node-b", &peers_b).await.unwrap();

    let conns_a = a_handle.await.unwrap().unwrap();
    assert!(conns_a.contains_key("node-b"), "node-a should reach late node-b via retry");
    assert!(conns_b.contains_key("node-a"));
}
