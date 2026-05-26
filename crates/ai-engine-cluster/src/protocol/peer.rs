//! Peer-to-peer messages for leaderless coordination (Phase B).
//!
//! In the star model the leader holds embedding + output and runs the whole
//! loop, so the only wire messages are leader↔worker control + activation
//! frames. When *any* node can coordinate, a coordinator that holds neither
//! end of the pipeline needs to (a) hand prompt/token ids to the embedding
//! host and (b) receive logits back from the output host. These two messages
//! cover that, riding the same mesh `Connection`s as the activation ring.

use crate::protocol::data::ActivationHeader;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Sent by the request coordinator to a specific peer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CoordinatorMsg {
    /// → embedding host: embed `token_ids` at `seq_pos`, run the host's layer
    /// range, and forward activations along the ring toward the output host.
    /// Used for both prefill (full prompt) and the per-token decode step.
    Embed {
        request_id: Uuid,
        seq_pos: u32,
        token_ids: Vec<i32>,
    },
    /// → any peer holding per-request state: release it (request finished or
    /// cancelled).
    End { request_id: Uuid },
}

/// Wire frame dispatched by the mesh: either a coordinator instruction or an
/// activation relay from the previous pipeline hop.
///
/// Defined here (next to `CoordinatorMsg`/`PeerMsg`) so both the `serve_peer`
/// loop and the coordinator (Task 3) can import it from one place.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PeerFrame {
    /// A coordinator-originated instruction (Embed or End).
    Coord(CoordinatorMsg),
    /// An activation relay from the upstream pipeline hop.
    Relay(ActivationHeader),
}

/// Sent back to the coordinator by a peer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PeerMsg {
    /// output host → coordinator: logits for `seq_pos`, to be sampled into the
    /// next token. `logits.len()` == vocab size.
    Logits {
        request_id: Uuid,
        seq_pos: u32,
        logits: Vec<f32>,
    },
    /// any peer → coordinator: this request aborted.
    Fault { request_id: Uuid, detail: String },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::codec::{decode, encode};

    #[test]
    fn coordinator_embed_roundtrips() {
        let m = CoordinatorMsg::Embed {
            request_id: Uuid::now_v7(),
            seq_pos: 7,
            token_ids: vec![1, 2, 3, 42],
        };
        let back: CoordinatorMsg = decode(&encode(&m).unwrap()).unwrap();
        assert_eq!(m, back);
    }

    #[test]
    fn peer_logits_roundtrips() {
        let m = PeerMsg::Logits {
            request_id: Uuid::now_v7(),
            seq_pos: 0,
            logits: vec![0.1, -2.5, 3.0, f32::MIN_POSITIVE],
        };
        let back: PeerMsg = decode(&encode(&m).unwrap()).unwrap();
        assert_eq!(m, back);
    }

    #[test]
    fn peer_frame_roundtrips() {
        use crate::protocol::data::{ActivationHeader, Dtype};
        let id = Uuid::now_v7();
        let coord = PeerFrame::Coord(CoordinatorMsg::Embed {
            request_id: id,
            seq_pos: 3,
            token_ids: vec![10, 20],
        });
        assert_eq!(
            coord,
            decode::<PeerFrame>(&encode(&coord).unwrap()).unwrap()
        );
        let relay = PeerFrame::Relay(ActivationHeader {
            request_id: id,
            seq_pos: 0,
            shape: [1, 5, 64],
            dtype: Dtype::F32,
            is_terminal: false,
        });
        assert_eq!(
            relay,
            decode::<PeerFrame>(&encode(&relay).unwrap()).unwrap()
        );
        let end = PeerFrame::Coord(CoordinatorMsg::End { request_id: id });
        assert_eq!(end, decode::<PeerFrame>(&encode(&end).unwrap()).unwrap());
    }

    #[test]
    fn end_and_fault_roundtrip() {
        let id = Uuid::now_v7();
        let e = CoordinatorMsg::End { request_id: id };
        assert_eq!(e, decode::<CoordinatorMsg>(&encode(&e).unwrap()).unwrap());
        let f = PeerMsg::Fault {
            request_id: id,
            detail: "oom".into(),
        };
        assert_eq!(f, decode::<PeerMsg>(&encode(&f).unwrap()).unwrap());
    }
}
