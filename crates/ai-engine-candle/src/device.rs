//! Device spec parsing and auto-detection for the candle backend.

use candle_core::Device;

/// Resolve a device spec string into a candle `Device`.
///
/// - `"auto"`  : CUDA(0) if available and the `cuda` feature is on, else Metal
///   if the `metal` feature is on, else CPU.
/// - `"cpu"`   : CPU.
/// - `"cuda:N"`: CUDA device N (requires `cuda` feature).
/// - `"metal"` : Metal device 0 (requires `metal` feature).
pub fn resolve_device(spec: &str) -> anyhow::Result<Device> {
    match spec {
        "auto" => {
            #[cfg(feature = "cuda")]
            {
                if let Ok(d) = Device::new_cuda(0) {
                    tracing::info!("candle device: cuda:0");
                    return Ok(d);
                }
            }
            #[cfg(feature = "metal")]
            {
                if let Ok(d) = Device::new_metal(0) {
                    tracing::info!("candle device: metal:0");
                    return Ok(d);
                }
            }
            tracing::info!("candle device: cpu (auto fallback)");
            Ok(Device::Cpu)
        }
        "cpu" => Ok(Device::Cpu),
        "metal" => {
            #[cfg(feature = "metal")]
            {
                Ok(Device::new_metal(0)?)
            }
            #[cfg(not(feature = "metal"))]
            {
                anyhow::bail!("device 'metal' requested but ai-engine-candle was built without the 'metal' feature")
            }
        }
        other if other.starts_with("cuda:") => {
            let idx: usize = other
                .strip_prefix("cuda:")
                .unwrap()
                .parse()
                .map_err(|_| anyhow::anyhow!("invalid cuda device index in '{other}'"))?;
            #[cfg(feature = "cuda")]
            {
                Ok(Device::new_cuda(idx)?)
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = idx;
                anyhow::bail!("device '{other}' requested but ai-engine-candle was built without the 'cuda' feature")
            }
        }
        other => anyhow::bail!("unknown device spec '{other}' (expected auto|cpu|metal|cuda:N)"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_cpu_spec() {
        let d = resolve_device("cpu").unwrap();
        assert!(d.is_cpu());
    }

    #[test]
    fn resolve_auto_does_not_error() {
        let d = resolve_device("auto").unwrap();
        let _ = d;
    }

    #[test]
    fn resolve_unknown_spec_errors() {
        assert!(resolve_device("banana").is_err());
    }
}
