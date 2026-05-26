//! Best-effort NVIDIA GPU telemetry via `nvidia-smi`. Returns an empty list
//! when no GPU / driver is present, so callers can simply hide the panel.
//!
//! `sample_cached` memoises behind a 1-second TTL so that N concurrent SSE
//! viewers still spawn at most one `nvidia-smi` per second.

use serde::Serialize;
use std::process::Command;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Default, Serialize)]
pub struct GpuInfo {
    pub index: u32,
    pub name: String,
    /// GPU core utilisation, percent.
    pub util_pct: u32,
    pub mem_used_mb: u64,
    pub mem_total_mb: u64,
    /// Core temperature, Celsius.
    pub temp_c: u32,
    /// Current board power draw, watts.
    pub power_w: f32,
    /// Enforced power limit, watts.
    pub power_limit_w: f32,
}

const QUERY: &str =
    "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit";

/// Run `nvidia-smi` once and parse one `GpuInfo` per GPU. Any failure
/// (binary missing, non-zero exit, unparseable line) yields an empty list.
pub fn sample() -> Vec<GpuInfo> {
    let out = Command::new("nvidia-smi")
        .args([
            &format!("--query-gpu={QUERY}"),
            "--format=csv,noheader,nounits",
        ])
        .output();
    let out = match out {
        Ok(o) if o.status.success() => o,
        _ => return Vec::new(),
    };
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(parse_line)
        .collect()
}

fn parse_line(line: &str) -> Option<GpuInfo> {
    let f: Vec<&str> = line.split(',').map(str::trim).collect();
    if f.len() < 8 {
        return None;
    }
    // Fields that can read "[N/A]" on some boards default to 0.
    let num = |s: &str| -> f32 { s.parse().unwrap_or(0.0) };
    Some(GpuInfo {
        index: num(f[0]) as u32,
        name: f[1].to_string(),
        util_pct: num(f[2]) as u32,
        mem_used_mb: num(f[3]) as u64,
        mem_total_mb: num(f[4]) as u64,
        temp_c: num(f[5]) as u32,
        power_w: num(f[6]),
        power_limit_w: num(f[7]),
    })
}

/// `sample()` behind a process-wide 1-second cache.
pub fn sample_cached() -> Vec<GpuInfo> {
    static CACHE: OnceLock<Mutex<(Instant, Vec<GpuInfo>)>> = OnceLock::new();
    let cell = CACHE.get_or_init(|| Mutex::new((Instant::now() - Duration::from_secs(60), Vec::new())));
    let mut guard = match cell.lock() {
        Ok(g) => g,
        Err(_) => return Vec::new(),
    };
    if guard.0.elapsed() >= Duration::from_secs(1) {
        *guard = (Instant::now(), sample());
    }
    guard.1.clone()
}
