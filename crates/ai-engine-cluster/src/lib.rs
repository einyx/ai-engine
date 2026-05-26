//! ai-engine-cluster
//!
//! Distributed inference coordinator. See
//! `docs/superpowers/specs/2026-05-23-ai-engine-distributed-inference-design.md`
//! for the design.
//!
//! Implements `Provider` from `ai_engine_provider` against a cluster of
//! nodes communicating over QUIC.

pub mod coordinator;
pub mod discovery;
pub mod capability;
pub mod leader;
pub mod partition;
pub mod peer;
pub mod protocol;
pub mod provider;
pub mod session;
pub mod tensor_io;
pub mod tls;
pub mod transport;
pub mod worker;
pub mod metrics;
pub mod view;

pub use capability::{detect_capability, BackendKind, Capability};
pub use leader::{ClusterLeader, LeaderConfig, WorkerEndpoint};
pub use partition::{auto_partition, manual_partition, NodeAssignment, PartitionManifest};
pub use provider::ClusterProvider;
pub use session::{LeaderModel, RequestSession};
pub use tls::{
    fingerprint_sha256, generate_node_identity, load_or_generate_node_identity, NodeIdentity,
};

#[cfg(test)]
mod smoke_compile_test {
    #[test]
    fn crate_compiles() {
        let _: burn::tensor::Tensor<burn_ndarray::NdArray, 1> =
            burn::tensor::Tensor::zeros([4], &Default::default());
    }
}
