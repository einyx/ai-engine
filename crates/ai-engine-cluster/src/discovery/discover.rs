//! mDNS service discovery for cluster nodes.
//!
//! [`discover_workers`] browses for the `_ai-engine._tcp.local.` service
//! type, filters resolved instances by `cluster_id` (from TXT records) and
//! `role=="worker"`, and returns up to `expected_count` distinct workers
//! (by `node_id`) — or whatever has been seen when `timeout` fires.
//!
//! TOFU: the first announcement seen for a given `node_id` wins; later
//! announcements (e.g. with a different fingerprint after a cert rotation)
//! are ignored until the next discovery cycle.

use crate::discovery::txt::{TxtRecords, ROLE_OLLAMA, SERVICE_TYPE};
use mdns_sd::{ServiceDaemon, ServiceEvent};
use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::time::Duration;

/// A worker endpoint resolved from an mDNS announcement.
#[derive(Debug, Clone)]
pub struct DiscoveredWorker {
    pub node_id: String,
    pub addr: SocketAddr,
    pub fingerprint: String,
    pub backend: String,
}

/// Browse for `_ai-engine._tcp.local.` services and return endpoints matching
/// `cluster_id`. Returns when up to `expected_count` distinct workers have
/// been resolved OR when `timeout` elapses — whichever comes first.
///
/// An empty return value is not an error; callers decide how to react.
pub async fn discover_workers(
    cluster_id: &str,
    expected_count: usize,
    timeout: Duration,
) -> anyhow::Result<Vec<DiscoveredWorker>> {
    let daemon = ServiceDaemon::new().map_err(|e| anyhow::anyhow!("mdns daemon: {e}"))?;
    let receiver = daemon
        .browse(SERVICE_TYPE)
        .map_err(|e| anyhow::anyhow!("mdns browse: {e}"))?;

    let mut found: HashMap<String, DiscoveredWorker> = HashMap::new();
    let deadline = tokio::time::Instant::now() + timeout;

    loop {
        if found.len() >= expected_count {
            break;
        }
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            break;
        }

        // mdns-sd's receiver is a flume::Receiver; its blocking `recv_timeout`
        // would stall a tokio worker. Bridge via spawn_blocking so the runtime
        // stays responsive while we wait for the next mDNS event.
        let recv_clone = receiver.clone();
        let join = tokio::task::spawn_blocking(move || recv_clone.recv_timeout(remaining)).await;

        let event = match join {
            Ok(Ok(ev)) => ev,
            Ok(Err(_timeout_or_disc)) => break,
            Err(_join_err) => break,
        };

        if let ServiceEvent::ServiceResolved(resolved) = event {
            let txt_map: HashMap<String, String> = resolved
                .get_properties()
                .iter()
                .map(|prop| (prop.key().to_string(), prop.val_str().to_string()))
                .collect();
            let txt = match TxtRecords::from_map(&txt_map) {
                Ok(t) => t,
                Err(_) => continue,
            };
            if txt.cluster_id != cluster_id {
                continue;
            }
            if txt.role != "worker" {
                continue;
            }
            // First-seen TOFU semantics.
            if found.contains_key(&txt.node_id) {
                continue;
            }

            // Prefer the first IPv4 address; IPv6 is currently skipped.
            let Some(ip) = resolved
                .get_addresses()
                .iter()
                .map(|scoped| scoped.to_ip_addr())
                .find(|ip| matches!(ip, IpAddr::V4(_)))
            else {
                continue;
            };
            let port = resolved.get_port();

            found.insert(
                txt.node_id.clone(),
                DiscoveredWorker {
                    node_id: txt.node_id,
                    addr: SocketAddr::new(ip, port),
                    fingerprint: txt.fingerprint,
                    backend: txt.backend,
                },
            );
        }
        // Other events (SearchStarted, ServiceFound, ServiceRemoved, etc.)
        // carry no TXT/address payload and are not actionable here.
    }

    let _ = daemon.shutdown();
    let mut out: Vec<DiscoveredWorker> = found.into_values().collect();
    out.sort_by(|a, b| a.node_id.cmp(&b.node_id));
    Ok(out)
}

/// An Ollama endpoint resolved from an mDNS announcement (`role == "ollama"`).
#[derive(Debug, Clone, PartialEq)]
pub struct DiscoveredOllama {
    /// Short host label from the TXT `label` field (e.g. `"xps"`).
    pub label: String,
    /// Base URL of the Ollama HTTP API, e.g. `http://192.168.190.175:11434`.
    pub url: String,
    /// Model names advertised via `/api/tags`.
    pub models: Vec<String>,
    /// Resolved socket address (IPv4) from the mDNS A record.
    pub addr: SocketAddr,
    /// Host resources advertised by the node (zeros if not advertised).
    pub resources: ai_engine_core::resources::NodeResources,
}

/// Browse `_ai-engine._tcp.local.` for `role == "ollama"` advertisements until
/// `timeout` elapses, returning one entry per distinct `label` (first-seen
/// wins). Never errors on "nothing found" — an empty vec is a valid result.
pub async fn discover_ollamas(timeout: Duration) -> anyhow::Result<Vec<DiscoveredOllama>> {
    let daemon = ServiceDaemon::new().map_err(|e| anyhow::anyhow!("mdns daemon: {e}"))?;
    let receiver = daemon
        .browse(SERVICE_TYPE)
        .map_err(|e| anyhow::anyhow!("mdns browse: {e}"))?;

    let mut found: HashMap<String, DiscoveredOllama> = HashMap::new();
    let deadline = tokio::time::Instant::now() + timeout;

    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            break;
        }
        let recv_clone = receiver.clone();
        let join = tokio::task::spawn_blocking(move || recv_clone.recv_timeout(remaining)).await;
        let event = match join {
            Ok(Ok(ev)) => ev,
            _ => break,
        };

        if let ServiceEvent::ServiceResolved(resolved) = event {
            let txt: HashMap<String, String> = resolved
                .get_properties()
                .iter()
                .map(|prop| (prop.key().to_string(), prop.val_str().to_string()))
                .collect();
            if txt.get("role").map(String::as_str) != Some(ROLE_OLLAMA) {
                continue;
            }
            let Some(url) = txt.get("ollama_url").cloned() else {
                continue;
            };
            let label = txt
                .get("label")
                .cloned()
                .unwrap_or_else(|| resolved.get_fullname().to_string());
            if found.contains_key(&label) {
                continue;
            }
            let models: Vec<String> = txt
                .get("models")
                .map(|s| {
                    s.split(',')
                        .map(str::trim)
                        .filter(|m| !m.is_empty())
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default();
            let Some(ip) = resolved
                .get_addresses()
                .iter()
                .map(|scoped| scoped.to_ip_addr())
                .find(|ip| matches!(ip, IpAddr::V4(_)))
            else {
                continue;
            };
            found.insert(
                label.clone(),
                DiscoveredOllama {
                    label,
                    url,
                    models,
                    addr: SocketAddr::new(ip, resolved.get_port()),
                    resources: ai_engine_core::resources::NodeResources::from_txt(&txt),
                },
            );
        }
    }

    let _ = daemon.shutdown();
    let mut out: Vec<DiscoveredOllama> = found.into_values().collect();
    out.sort_by(|a, b| a.label.cmp(&b.label));
    Ok(out)
}
