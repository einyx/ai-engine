use serde::{Deserialize, Serialize};
use std::time::Instant;
use sysinfo::System;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum BackendKind {
    Cpu,
    Cuda,
    Metal,
    Wgpu,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capability {
    pub node_id: String,
    pub backend: BackendKind,
    pub device_index: usize,
    pub available_memory_bytes: u64,
    pub compute_score: u32,
    pub link_mbps_to_leader: u32,
}

const SAFETY_MARGIN_BYTES: u64 = 512 * 1024 * 1024;

/// Detect this node's capability. `max_memory_mib` is an optional config override
/// that caps the memory advertised (useful for shared boxes).
pub fn detect_capability(
    node_id: &str,
    backend: BackendKind,
    device_index: usize,
    max_memory_mib: Option<u64>,
) -> anyhow::Result<Capability> {
    let detected_mem = detect_memory_bytes(backend)?;
    let with_margin = detected_mem.saturating_sub(SAFETY_MARGIN_BYTES);
    let final_mem = match max_memory_mib {
        Some(cap) => with_margin.min(cap * 1024 * 1024),
        None => with_margin,
    };
    let compute_score = microbenchmark_compute_score();

    Ok(Capability {
        node_id: node_id.to_string(),
        backend,
        device_index,
        available_memory_bytes: final_mem,
        compute_score,
        link_mbps_to_leader: 0, // populated during QUIC handshake by leader
    })
}

fn detect_memory_bytes(backend: BackendKind) -> anyhow::Result<u64> {
    match backend {
        BackendKind::Cpu => {
            let mut sys = System::new_all();
            sys.refresh_memory();
            // sysinfo 0.32: total_memory() returns bytes.
            Ok(sys.total_memory())
        }
        BackendKind::Cuda | BackendKind::Metal | BackendKind::Wgpu => {
            // Backend-specific VRAM detection is deferred — Task 4 only covers CPU.
            // Real impl uses nvml-wrapper / metal::MTLDevice / wgpu::Adapter::get_info.
            // For now: fall back to "1 GiB", which the integration tests don't depend on.
            // Plan 3 will plumb real detection through.
            Ok(1024 * 1024 * 1024)
        }
    }
}

/// One-time matmul microbenchmark normalized to ~100 for a baseline CPU.
/// Returns a dimensionless relative score. Higher is faster.
fn microbenchmark_compute_score() -> u32 {
    // Multiply two 256x256 f32 matrices. We don't use burn here — that would
    // require a generic backend in this module which is undesirable.
    // Simple naive matmul is plenty for a relative score.
    const N: usize = 256;
    let a: Vec<f32> = (0..N * N).map(|i| (i as f32 * 0.001).sin()).collect();
    let b: Vec<f32> = (0..N * N).map(|i| (i as f32 * 0.002).cos()).collect();
    let mut c = vec![0.0_f32; N * N];

    let t0 = Instant::now();
    for i in 0..N {
        for j in 0..N {
            let mut s = 0.0;
            for k in 0..N {
                s += a[i * N + k] * b[k * N + j];
            }
            c[i * N + j] = s;
        }
    }
    // Prevent the compiler from optimizing the matmul away.
    std::hint::black_box(&c);
    let elapsed_ms = t0.elapsed().as_millis() as f64;
    // Baseline: ~250 ms on a slow CPU -> score 100.
    // score = 100 * 250 / elapsed_ms
    let score = (100.0 * 250.0 / elapsed_ms.max(1.0)) as u32;
    score.max(1)
}
