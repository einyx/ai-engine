//! Best-effort host resource sampling (Linux) + the mDNS TXT schema shared by
//! the gateway, the Ollama advertiser, and `rustyllm-serve`. Reads
//! `/proc/loadavg`, `/proc/meminfo`, and `statvfs` — no heavy deps.

use serde::Serialize;
use std::collections::HashMap;

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct NodeResources {
    pub cpu_count: u32,
    /// 1-minute load average.
    pub load1: f32,
    pub mem_total_mb: u64,
    pub mem_avail_mb: u64,
    pub disk_avail_gb: f32,
}

impl NodeResources {
    /// The five mDNS TXT key/values carrying resources. Every advertiser
    /// (ai-engine, rustyllm-serve) emits exactly these keys.
    pub fn to_txt(&self) -> [(String, String); 5] {
        [
            ("cpu".into(), self.cpu_count.to_string()),
            ("load1".into(), format!("{:.2}", self.load1)),
            ("mem_total_mb".into(), self.mem_total_mb.to_string()),
            ("mem_avail_mb".into(), self.mem_avail_mb.to_string()),
            ("disk_avail_gb".into(), format!("{:.1}", self.disk_avail_gb)),
        ]
    }

    /// Parse resources out of a TXT map (missing/garbage fields → zero).
    pub fn from_txt(m: &HashMap<String, String>) -> Self {
        fn parse<T: std::str::FromStr + Default>(m: &HashMap<String, String>, k: &str) -> T {
            m.get(k).and_then(|v| v.parse().ok()).unwrap_or_default()
        }
        Self {
            cpu_count: parse(m, "cpu"),
            load1: parse(m, "load1"),
            mem_total_mb: parse(m, "mem_total_mb"),
            mem_avail_mb: parse(m, "mem_avail_mb"),
            disk_avail_gb: parse(m, "disk_avail_gb"),
        }
    }

    /// True if any field is non-zero — i.e. we actually have data.
    pub fn is_some(&self) -> bool {
        self.cpu_count != 0 || self.mem_total_mb != 0 || self.disk_avail_gb != 0.0
    }
}

/// Sample this host's resources. Best-effort: unreadable fields stay zero.
pub fn sample() -> NodeResources {
    let cpu_count = std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(0);
    let load1 = std::fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|s| s.split_whitespace().next().and_then(|f| f.parse().ok()))
        .unwrap_or(0.0);
    let (mem_total_mb, mem_avail_mb) = meminfo();
    NodeResources {
        cpu_count,
        load1,
        mem_total_mb,
        mem_avail_mb,
        disk_avail_gb: disk_avail_gb("/"),
    }
}

/// (MemTotal, MemAvailable) in MB from `/proc/meminfo` (kB there).
fn meminfo() -> (u64, u64) {
    let text = match std::fs::read_to_string("/proc/meminfo") {
        Ok(t) => t,
        Err(_) => return (0, 0),
    };
    let field = |name: &str| -> u64 {
        text.lines()
            .find(|l| l.starts_with(name))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|kb| kb.parse::<u64>().ok())
            .map(|kb| kb / 1024)
            .unwrap_or(0)
    };
    (field("MemTotal:"), field("MemAvailable:"))
}

/// Free disk (GB) on the filesystem holding `path`, via `statvfs`.
fn disk_avail_gb(path: &str) -> f32 {
    let c = match std::ffi::CString::new(path) {
        Ok(c) => c,
        Err(_) => return 0.0,
    };
    // SAFETY: `stat` is zeroed and only read after a successful statvfs.
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(c.as_ptr(), &mut stat) } != 0 {
        return 0.0;
    }
    (stat.f_bavail as u64 * stat.f_frsize as u64) as f32 / 1e9
}
