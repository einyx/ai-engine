use crate::capability::{detect_capability, BackendKind};
use crate::protocol::codec::{decode, encode};
use crate::protocol::control::{LeaderToWorker, WorkerToLeader};
use crate::transport::frame::{read_frame, write_frame};
use quinn::Endpoint;

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
