//! ai-engine-cluster
//!
//! Distributed inference coordinator. Implements `Provider` from
//! `ai_engine_provider` against a cluster of nodes running QUIC.

pub mod tls;

#[cfg(test)]
mod smoke_compile_test {
    #[test]
    fn crate_compiles() {
        let _: burn::tensor::Tensor<burn_ndarray::NdArray, 1> =
            burn::tensor::Tensor::zeros([4], &Default::default());
    }
}
