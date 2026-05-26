//! Backend selection. v0.2 supports CPU (ndarray), CUDA, WGPU (covers Metal).

use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind { Cpu, Cuda, Metal, Wgpu }

impl FromStr for BackendKind {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda),
            "metal" => Ok(Self::Metal),
            "wgpu" => Ok(Self::Wgpu),
            other => anyhow::bail!("unknown backend kind: {other}"),
        }
    }
}

#[cfg(feature = "backend-cpu")]
pub type CpuBackend = burn_ndarray::NdArray;

#[cfg(feature = "backend-cuda")]
pub type CudaBackend = burn_cuda::Cuda;

#[cfg(feature = "backend-wgpu")]
pub type WgpuBackend = burn_wgpu::Wgpu;
