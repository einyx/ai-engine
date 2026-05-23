//! ai-engine-cluster
//!
//! Distributed inference coordinator. See
//! `docs/superpowers/specs/2026-05-23-ai-engine-distributed-inference-design.md`
//! for the design.
//!
//! Implements `Provider` from `ai_engine_provider` against a cluster of
//! nodes communicating over QUIC.

pub mod capability;
pub mod leader;
pub mod partition;
pub mod protocol;
pub mod provider;
pub mod tensor_io;
pub mod tls;
pub mod transport;
pub mod worker;

pub use capability::{detect_capability, BackendKind, Capability};
pub use leader::{ClusterLeader, LeaderConfig, WorkerEndpoint};
pub use partition::{auto_partition, manual_partition, NodeAssignment, PartitionManifest};
pub use provider::ClusterProvider;
pub use tls::{fingerprint_sha256, generate_node_identity, NodeIdentity};

#[cfg(test)]
mod smoke_compile_test {
    #[test]
    fn crate_compiles() {
        let _: burn::tensor::Tensor<burn_ndarray::NdArray, 1> =
            burn::tensor::Tensor::zeros([4], &Default::default());
    }
}
