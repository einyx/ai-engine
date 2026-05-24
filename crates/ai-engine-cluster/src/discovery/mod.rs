//! mDNS auto-discovery for cluster nodes.
//!
//! Service type: `_ai-engine._tcp.local.`. Each node announces itself with
//! TXT records carrying its cluster_id, node_id, role, protocol_version,
//! cert fingerprint, and backend. The leader browses for matching services
//! and collects worker endpoints from the discovered TXT data.
//!
//! See `txt.rs` for the TXT-record schema.

pub mod announce;
pub mod discover;
pub mod txt;

pub use txt::{TxtRecords, SERVICE_TYPE};
