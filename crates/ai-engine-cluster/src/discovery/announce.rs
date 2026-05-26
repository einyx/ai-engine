//! mDNS service announcement.
//!
//! [`Announcer`] wraps an `mdns_sd::ServiceDaemon` that registers a single
//! cluster-node service and keeps re-announcing it until [`Announcer::shutdown`]
//! is called (or the handle is dropped).
//!
//! The daemon spawns its own background OS thread; calling `register` is
//! non-blocking from the caller's perspective.

use crate::discovery::txt::{TxtRecords, SERVICE_TYPE};
use mdns_sd::{ServiceDaemon, ServiceInfo};
use std::collections::HashMap;
use std::net::IpAddr;

/// Handle for an ongoing mDNS service registration. Holds the daemon alive
/// for the lifetime of the handle.
///
/// On `Drop`, a best-effort `unregister` + `shutdown` is issued; for explicit
/// teardown (e.g. waiting until withdrawal has been broadcast), call
/// [`Announcer::shutdown`].
pub struct Announcer {
    daemon: Option<ServiceDaemon>,
    fullname: String,
}

impl Announcer {
    /// Register a service. The daemon stays alive until this handle is dropped
    /// or [`Announcer::shutdown`] is called.
    ///
    /// `bind_ip` is what the worker advertises in the A/AAAA record — usually
    /// the QUIC listener's local IP. Pass `127.0.0.1` for loopback tests.
    /// `host_name` should be a fully-qualified `.local.` hostname (e.g.
    /// `"worker1.local."`).
    pub fn register(
        bind_ip: IpAddr,
        port: u16,
        host_name: &str,
        txt: TxtRecords,
    ) -> anyhow::Result<Self> {
        let daemon = ServiceDaemon::new().map_err(|e| anyhow::anyhow!("mdns daemon: {e}"))?;

        let instance_name = format!("{}.{}", txt.node_id, txt.cluster_id);

        let info = ServiceInfo::new(
            SERVICE_TYPE,
            &instance_name,
            host_name,
            bind_ip,
            port,
            txt.to_map(),
        )
        .map_err(|e| anyhow::anyhow!("mdns ServiceInfo: {e}"))?;

        let fullname = info.get_fullname().to_string();

        daemon
            .register(info)
            .map_err(|e| anyhow::anyhow!("mdns register: {e}"))?;

        Ok(Self {
            daemon: Some(daemon),
            fullname,
        })
    }

    /// Register a service with an arbitrary TXT map under `SERVICE_TYPE`.
    ///
    /// Used for non-worker advertisements (e.g. an Ollama endpoint) that don't
    /// fit the typed [`TxtRecords`] schema. `instance_name` must be unique on
    /// the LAN for this service type.
    pub fn register_raw(
        bind_ip: IpAddr,
        port: u16,
        host_name: &str,
        instance_name: &str,
        txt: HashMap<String, String>,
    ) -> anyhow::Result<Self> {
        let daemon = ServiceDaemon::new().map_err(|e| anyhow::anyhow!("mdns daemon: {e}"))?;
        let info = ServiceInfo::new(SERVICE_TYPE, instance_name, host_name, bind_ip, port, txt)
            .map_err(|e| anyhow::anyhow!("mdns ServiceInfo: {e}"))?;
        let fullname = info.get_fullname().to_string();
        daemon
            .register(info)
            .map_err(|e| anyhow::anyhow!("mdns register: {e}"))?;
        Ok(Self {
            daemon: Some(daemon),
            fullname,
        })
    }

    /// Service instance fullname (e.g. `worker-1.test-loop._ai-engine._tcp.local.`).
    pub fn fullname(&self) -> &str {
        &self.fullname
    }

    /// Explicitly unregister + shut down the daemon. Dropping does the same
    /// thing implicitly, but this lets the caller make the teardown intent
    /// explicit at a call site.
    pub fn shutdown(mut self) {
        if let Some(daemon) = self.daemon.take() {
            let _ = daemon.unregister(&self.fullname);
            let _ = daemon.shutdown();
        }
    }
}

impl Drop for Announcer {
    fn drop(&mut self) {
        if let Some(daemon) = self.daemon.take() {
            // Best-effort: ignore errors since the daemon may already be
            // closing. mdns-sd 0.19 does not block on unregister/shutdown
            // (they return Receivers we drop), so the unregister "goodbye"
            // packet may not actually be flushed before our process exits.
            // For long-lived daemons this is fine; the discoverer's TOFU
            // cache will simply hold a stale entry until next announce cycle.
            let _ = daemon.unregister(&self.fullname);
            let _ = daemon.shutdown();
        }
    }
}
