# Plan 8 — v0.3.0-alpha.4: mDNS auto-discovery

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Nodes find each other on the LAN via mDNS — no more pasting cert fingerprints into every `[[cluster.node]]` block. Workers announce themselves with TXT records carrying their fingerprint + capabilities; the leader discovers them by service type and pins the announced fingerprint on first connect (TOFU).

**Architecture:** A new `discovery` module in `ai-engine-cluster` wrapping `mdns-sd` (pure-Rust mDNS). Two long-lived tokio tasks: an `Announcer` (each worker registers its service on startup, kept alive) and a `Discoverer` (the leader scans for matching service types on startup, collects N workers within a timeout). The existing `[[cluster.node]]` static path stays — mDNS is additive. A new `[[cluster.discover]]` TOML block flips the leader into discovery mode; workers can advertise even in static-config mode (harmless). Discovered worker fingerprints feed into the existing `LeaderConfig.workers: Vec<WorkerEndpoint>` field, after which the existing connect / handshake / Assignment path runs unchanged.

**Tech Stack:** `mdns-sd = "0.13"` (or current stable). No other new deps. Reuses the existing tokio runtime + the `LeaderConfig` / `WorkerEndpoint` data flow established in Plans 2 and 3.

**Scope rule:** Plan 8 ships **static cluster startup with mDNS replacing the fingerprint-in-config requirement.** Dynamic membership (workers joining / leaving a running cluster), worker failover with KV reconstruction, hot-reload of discovered topology — all explicitly deferred to a later plan.

**Baseline:** Branch `main` at `v0.3.0-alpha.3`. 194 tests + 4 ignored. Clippy clean.

---

## Wire format (locked in here)

mDNS service type: **`_ai-engine._tcp.local.`** (the trailing dot is part of the mDNS convention).

Each worker advertises one mDNS service instance with:
- **Service type**: `_ai-engine._tcp.local.`
- **Instance name**: `<node_id>.<cluster_id>` (e.g., `worker-1.home-lab`)
- **Hostname**: this node's hostname (mdns-sd auto-fills)
- **Port**: the worker's QUIC `quic_bind` port (the same port `[[cluster.node]].addr` would have provided)
- **TXT records** (key=value strings):
  - `cluster_id=<cluster_id>`
  - `node_id=<node_id>`
  - `role=worker` (leaders may also advertise with `role=leader` for symmetry; not used in v0.3.0-alpha.4)
  - `protocol_version=1`
  - `fingerprint=sha256:<64 hex chars>` (the node's TLS cert fingerprint)
  - `backend=cpu|cuda|metal|wgpu`

**TOFU semantics**: the leader trusts the fingerprint from the FIRST mDNS announcement it sees for a given `node_id` within a cluster. Later contradictory announcements are ignored with a warning (rare in practice unless a node restarts with a different cert AND races discovery).

The IP address comes from the mDNS A/AAAA record (the worker's host IP, auto-filled by mdns-sd from the host's network interfaces).

---

## File structure

```
crates/ai-engine-cluster/
├── src/
│   ├── lib.rs                       # MODIFY: pub mod discovery
│   ├── discovery/
│   │   ├── mod.rs                   # NEW: pub re-exports
│   │   ├── announce.rs              # NEW: Announcer task — register + keep alive
│   │   ├── discover.rs              # NEW: Discoverer task — browse + collect endpoints
│   │   └── txt.rs                   # NEW: TXT record encode/decode
│   └── leader.rs                    # MODIFY: alternate constructor LeaderConfig::from_discovery
└── tests/
    ├── discovery_txt.rs             # NEW: TXT roundtrip
    ├── discovery_loopback.rs        # NEW: announcer + discoverer on the same host
    └── discovery_cluster.rs         # NEW: 3-node mDNS cluster (no fingerprints in config)
```

```
crates/ai-engine-config/
├── src/
│   ├── lib.rs                       # MODIFY: ClusterDiscover struct + Cluster.discover field
│   └── validate.rs                  # MODIFY: discover validation
└── tests/
    └── load.rs                      # MODIFY: discover-block parse test
```

```
crates/ai-engine/
├── src/
│   ├── app.rs                       # MODIFY: leader-mode build_app_state calls Discoverer when [[cluster.discover]] is set
│   └── worker_main.rs               # MODIFY: workers start Announcer task before run_worker_full
└── tests/
    └── multiproc_smoke_mdns.rs      # NEW: 3-process cluster, no fingerprints in config, mDNS discovery
```

---

## Important pre-flight notes

- **`mdns-sd`** is the canonical pure-Rust mDNS crate. It supports announce + browse with TXT records and provides tokio-friendly async APIs (or sync stream APIs that can be polled in a task). Version pinned in Task 1.
- **Loopback testing**: mDNS broadcasts on the default multicast interface. On a single machine, registering and browsing for the same service type works as expected. On CI, the runner needs IP multicast enabled (most cloud runners do). If the integration test in Task 7 fails on CI but works locally, it's an IP-multicast permissions issue — note as a `#[ignore]`-able test.
- **Fingerprint persistence**: workers persist their cert in `~/.ai-engine/node.{key,crt}` (added in Plan 3). The Announcer reads the same identity. After a restart, the same fingerprint is advertised, so the leader's TOFU pin stays valid across worker restarts.
- **Worker IP address**: the IP advertised by mDNS comes from the host's network interface that owns the default route to the multicast group. On a machine with multiple interfaces, mdns-sd picks one (typically the first non-loopback). For the in-process test in Task 6, mdns-sd may advertise the loopback interface — that's fine; the leader connects to 127.0.0.1.
- **Multiple `[[cluster]]` blocks** with different discovery configs are supported — each spawns its own Discoverer/Announcer pair, distinguished by the `cluster_id` TXT field.

---

### Task 1: Add `mdns-sd` dep + module scaffold

**Files:**
- Modify: root `Cargo.toml` (workspace deps)
- Modify: `crates/ai-engine-cluster/Cargo.toml`
- Create: `crates/ai-engine-cluster/src/discovery/mod.rs`
- Modify: `crates/ai-engine-cluster/src/lib.rs`

- [ ] **Step 1: Workspace dep addition**

Append to root `Cargo.toml` `[workspace.dependencies]`:

```toml
mdns-sd = "0.13"
```

(Verify against current crates.io — if 0.13 is yanked or has API changes, pin to whatever the latest stable is. Note actual version in commit body.)

- [ ] **Step 2: Crate dep**

Add to `crates/ai-engine-cluster/Cargo.toml` `[dependencies]`:

```toml
mdns-sd.workspace = true
```

- [ ] **Step 3: Module skeleton**

`crates/ai-engine-cluster/src/discovery/mod.rs`:

```rust
//! mDNS auto-discovery for cluster nodes.
//!
//! Service type: `_ai-engine._tcp.local.`. Each node announces itself with
//! TXT records carrying its cluster_id, node_id, role, protocol_version,
//! cert fingerprint, and backend. The leader browses for matching services
//! and collects worker endpoints from the discovered TXT data.
//!
//! See `txt.rs` for the TXT-record schema.

pub mod announce;
pub mod discover;
pub mod txt;

pub use announce::Announcer;
pub use discover::{discover_workers, DiscoveredWorker};
pub use txt::{TxtRecords, SERVICE_TYPE};
```

- [ ] **Step 4: Stubs for the not-yet-implemented files**

`crates/ai-engine-cluster/src/discovery/announce.rs`:

```rust
//! Filled in by Task 3.
```

`crates/ai-engine-cluster/src/discovery/discover.rs`:

```rust
//! Filled in by Task 4.
```

`crates/ai-engine-cluster/src/discovery/txt.rs`:

```rust
//! Filled in by Task 2.
```

- [ ] **Step 5: Wire module + verify it compiles**

`crates/ai-engine-cluster/src/lib.rs` (append):

```rust
pub mod discovery;
```

```bash
cd /home/alessio/aip/airproxy
cargo check -p ai-engine-cluster
cargo clippy --workspace --all-targets -- -D warnings
```

The clippy run will warn about unused stubs — accept those for one commit (or add `#[allow(dead_code)]` at the module level temporarily — the next task fills it in).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(cluster): scaffold for mDNS discovery module"
```

NO Co-Authored-By footer (global preference).

---

### Task 2: TXT record encode / decode

**Files:**
- Modify: `crates/ai-engine-cluster/src/discovery/txt.rs`
- Create: `crates/ai-engine-cluster/tests/discovery_txt.rs`

`mdns-sd` represents TXT records as a `HashMap<String, String>`. Build a typed wrapper.

- [ ] **Step 1: Failing test**

`crates/ai-engine-cluster/tests/discovery_txt.rs`:

```rust
use ai_engine_cluster::discovery::txt::{TxtRecords, SERVICE_TYPE};
use std::collections::HashMap;

#[test]
fn service_type_constant() {
    assert_eq!(SERVICE_TYPE, "_ai-engine._tcp.local.");
}

#[test]
fn txt_records_roundtrip() {
    let r = TxtRecords {
        cluster_id: "home-lab".into(),
        node_id: "worker-1".into(),
        role: "worker".into(),
        protocol_version: 1,
        fingerprint: "sha256:abc123".into(),
        backend: "cpu".into(),
    };
    let map = r.to_map();
    assert_eq!(map.get("cluster_id"), Some(&"home-lab".to_string()));
    assert_eq!(map.get("node_id"), Some(&"worker-1".to_string()));
    assert_eq!(map.get("role"), Some(&"worker".to_string()));
    assert_eq!(map.get("protocol_version"), Some(&"1".to_string()));
    assert_eq!(map.get("fingerprint"), Some(&"sha256:abc123".to_string()));
    assert_eq!(map.get("backend"), Some(&"cpu".to_string()));

    let back = TxtRecords::from_map(&map).unwrap();
    assert_eq!(back.cluster_id, r.cluster_id);
    assert_eq!(back.node_id, r.node_id);
    assert_eq!(back.role, r.role);
    assert_eq!(back.protocol_version, r.protocol_version);
    assert_eq!(back.fingerprint, r.fingerprint);
    assert_eq!(back.backend, r.backend);
}

#[test]
fn missing_required_field_errors() {
    let mut map: HashMap<String, String> = HashMap::new();
    map.insert("cluster_id".into(), "x".into());
    map.insert("node_id".into(), "y".into());
    // role / protocol_version / fingerprint / backend missing
    let err = TxtRecords::from_map(&map).unwrap_err().to_string();
    assert!(err.to_lowercase().contains("missing"), "got: {err}");
}

#[test]
fn malformed_protocol_version_errors() {
    let mut map: HashMap<String, String> = HashMap::new();
    map.insert("cluster_id".into(), "x".into());
    map.insert("node_id".into(), "y".into());
    map.insert("role".into(), "worker".into());
    map.insert("protocol_version".into(), "not-a-number".into());
    map.insert("fingerprint".into(), "sha256:abc".into());
    map.insert("backend".into(), "cpu".into());
    let err = TxtRecords::from_map(&map).unwrap_err().to_string();
    assert!(err.to_lowercase().contains("protocol_version"), "got: {err}");
}
```

- [ ] **Step 2: Confirm fails**

```bash
cargo test -p ai-engine-cluster --test discovery_txt 2>&1 | tail -10
# Expected: compile error — TxtRecords doesn't exist.
```

- [ ] **Step 3: Implement `txt.rs`**

```rust
use std::collections::HashMap;

pub const SERVICE_TYPE: &str = "_ai-engine._tcp.local.";

#[derive(Debug, Clone)]
pub struct TxtRecords {
    pub cluster_id: String,
    pub node_id: String,
    pub role: String,           // "worker" | "leader"
    pub protocol_version: u16,
    pub fingerprint: String,    // "sha256:<64 hex chars>"
    pub backend: String,        // "cpu" | "cuda" | "metal" | "wgpu"
}

impl TxtRecords {
    pub fn to_map(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("cluster_id".into(), self.cluster_id.clone());
        m.insert("node_id".into(), self.node_id.clone());
        m.insert("role".into(), self.role.clone());
        m.insert("protocol_version".into(), self.protocol_version.to_string());
        m.insert("fingerprint".into(), self.fingerprint.clone());
        m.insert("backend".into(), self.backend.clone());
        m
    }

    pub fn from_map(m: &HashMap<String, String>) -> anyhow::Result<Self> {
        let get = |k: &str| -> anyhow::Result<String> {
            m.get(k)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("TXT records missing required field `{k}`"))
        };
        let cluster_id = get("cluster_id")?;
        let node_id = get("node_id")?;
        let role = get("role")?;
        let protocol_version: u16 = get("protocol_version")?
            .parse()
            .map_err(|e| anyhow::anyhow!("malformed protocol_version: {e}"))?;
        let fingerprint = get("fingerprint")?;
        let backend = get("backend")?;
        Ok(Self { cluster_id, node_id, role, protocol_version, fingerprint, backend })
    }
}
```

- [ ] **Step 4: Verify + commit**

```bash
cargo test -p ai-engine-cluster --test discovery_txt
cargo clippy --workspace --all-targets -- -D warnings
git add -A
git commit -m "feat(cluster): TxtRecords schema for mDNS announcement payload"
```

NO Co-Authored-By.

---

### Task 3: Announcer task

**Files:**
- Modify: `crates/ai-engine-cluster/src/discovery/announce.rs`

A long-lived task that registers a service with `mdns-sd::ServiceDaemon`. The daemon needs to stay alive (mDNS protocol requires re-announcements at intervals); we hold onto its handle in an `Announcer` wrapper.

- [ ] **Step 1: Implement `announce.rs`**

```rust
use crate::discovery::txt::{TxtRecords, SERVICE_TYPE};
use mdns_sd::{ServiceDaemon, ServiceInfo};
use std::net::IpAddr;

/// Handle for an ongoing mDNS service registration. Dropping it stops the
/// daemon and unregisters the service.
pub struct Announcer {
    daemon: ServiceDaemon,
    fullname: String,
}

impl Announcer {
    /// Register a service. The daemon stays alive until this is dropped.
    /// Bind IP is what the worker advertises in the A/AAAA record — usually
    /// the QUIC listener's local IP. Pass `127.0.0.1` for loopback tests.
    pub fn register(
        bind_ip: IpAddr,
        port: u16,
        host_name: &str,
        txt: TxtRecords,
    ) -> anyhow::Result<Self> {
        let daemon = ServiceDaemon::new()
            .map_err(|e| anyhow::anyhow!("mdns daemon: {e}"))?;

        let instance_name = format!("{}.{}", txt.node_id, txt.cluster_id);
        let fullname = format!("{instance_name}.{SERVICE_TYPE}");

        let info = ServiceInfo::new(
            SERVICE_TYPE,
            &instance_name,
            host_name,
            bind_ip,
            port,
            txt.to_map(),
        ).map_err(|e| anyhow::anyhow!("mdns ServiceInfo: {e}"))?;

        daemon.register(info)
            .map_err(|e| anyhow::anyhow!("mdns register: {e}"))?;

        Ok(Self { daemon, fullname })
    }

    pub fn fullname(&self) -> &str { &self.fullname }

    /// Explicitly unregister + shut down the daemon.
    /// Dropping does the same thing implicitly, but this lets the caller
    /// await the shutdown to complete.
    pub fn shutdown(self) {
        let _ = self.daemon.unregister(&self.fullname);
        let _ = self.daemon.shutdown();
    }
}
```

mdns-sd's exact API for `ServiceInfo::new` varies between versions. Verify the constructor signature with `cargo doc -p mdns-sd` after the dep is added — the parameter order and TXT-records form might differ slightly. The pattern is: bind to a service type + instance name, give it a port + IP + TXT.

- [ ] **Step 2: No standalone test for the Announcer** — it's exercised end-to-end in Task 4's loopback test and Task 7's multi-proc test. Standalone testing of an mDNS registration without a discoverer adds little; the round-trip is what matters.

- [ ] **Step 3: Verify it compiles**

```bash
cargo check -p ai-engine-cluster
cargo clippy --workspace --all-targets -- -D warnings
```

If clippy complains about unused fields, gate with `#[allow(dead_code)]` until Task 4 wires it.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat(cluster): Announcer wraps mdns-sd ServiceDaemon for cluster nodes"
```

NO Co-Authored-By.

---

### Task 4: Discoverer task + loopback test

**Files:**
- Modify: `crates/ai-engine-cluster/src/discovery/discover.rs`
- Create: `crates/ai-engine-cluster/tests/discovery_loopback.rs`

The leader-side counterpart. Browses for `_ai-engine._tcp.local.` services, collects matching ones, returns a vector once it has seen `expected_count` workers OR the timeout fires.

- [ ] **Step 1: Failing test**

`crates/ai-engine-cluster/tests/discovery_loopback.rs`:

```rust
use ai_engine_cluster::discovery::{discover_workers, Announcer, DiscoveredWorker, TxtRecords};
use std::net::IpAddr;
use std::time::Duration;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn discover_two_workers_on_loopback() {
    // Each worker registers its service. The leader then browses and collects 2.
    let txt1 = TxtRecords {
        cluster_id: "test-loop".into(),
        node_id: "worker-1".into(),
        role: "worker".into(),
        protocol_version: 1,
        fingerprint: "sha256:1111111111111111111111111111111111111111111111111111111111111111".into(),
        backend: "cpu".into(),
    };
    let _ann1 = Announcer::register(
        IpAddr::V4("127.0.0.1".parse().unwrap()),
        50001,
        "worker1.local.",
        txt1.clone(),
    ).unwrap();

    let txt2 = TxtRecords {
        cluster_id: "test-loop".into(),
        node_id: "worker-2".into(),
        role: "worker".into(),
        protocol_version: 1,
        fingerprint: "sha256:2222222222222222222222222222222222222222222222222222222222222222".into(),
        backend: "cpu".into(),
    };
    let _ann2 = Announcer::register(
        IpAddr::V4("127.0.0.1".parse().unwrap()),
        50002,
        "worker2.local.",
        txt2.clone(),
    ).unwrap();

    // Give mDNS a moment to propagate.
    tokio::time::sleep(Duration::from_millis(200)).await;

    let found = discover_workers(
        "test-loop",
        /*expected_count=*/2,
        Duration::from_secs(5),
    ).await.unwrap();

    assert_eq!(found.len(), 2, "expected 2 workers, found {}", found.len());
    let ids: Vec<&str> = found.iter().map(|w| w.node_id.as_str()).collect();
    assert!(ids.contains(&"worker-1"), "missing worker-1 in {:?}", ids);
    assert!(ids.contains(&"worker-2"), "missing worker-2 in {:?}", ids);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn discover_returns_partial_after_timeout() {
    let txt = TxtRecords {
        cluster_id: "test-timeout".into(),
        node_id: "only-worker".into(),
        role: "worker".into(),
        protocol_version: 1,
        fingerprint: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into(),
        backend: "cpu".into(),
    };
    let _ann = Announcer::register(
        IpAddr::V4("127.0.0.1".parse().unwrap()),
        50003,
        "only-worker.local.",
        txt,
    ).unwrap();

    tokio::time::sleep(Duration::from_millis(200)).await;

    // Wait for 2 workers but only 1 exists; timeout returns whatever was found.
    let found = discover_workers(
        "test-timeout",
        /*expected_count=*/2,
        Duration::from_millis(800),
    ).await.unwrap();

    assert_eq!(found.len(), 1);
    assert_eq!(found[0].node_id, "only-worker");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn discover_filters_by_cluster_id() {
    let txt_a = TxtRecords {
        cluster_id: "cluster-A".into(),
        node_id: "n1".into(),
        role: "worker".into(),
        protocol_version: 1,
        fingerprint: "sha256:0000000000000000000000000000000000000000000000000000000000000001".into(),
        backend: "cpu".into(),
    };
    let txt_b = TxtRecords {
        cluster_id: "cluster-B".into(),
        node_id: "n2".into(),
        role: "worker".into(),
        protocol_version: 1,
        fingerprint: "sha256:0000000000000000000000000000000000000000000000000000000000000002".into(),
        backend: "cpu".into(),
    };
    let _a = Announcer::register(
        IpAddr::V4("127.0.0.1".parse().unwrap()), 50004, "n1.local.", txt_a,
    ).unwrap();
    let _b = Announcer::register(
        IpAddr::V4("127.0.0.1".parse().unwrap()), 50005, "n2.local.", txt_b,
    ).unwrap();

    tokio::time::sleep(Duration::from_millis(200)).await;

    let found_a = discover_workers("cluster-A", 1, Duration::from_secs(2)).await.unwrap();
    assert_eq!(found_a.len(), 1);
    assert_eq!(found_a[0].node_id, "n1");

    let found_b = discover_workers("cluster-B", 1, Duration::from_secs(2)).await.unwrap();
    assert_eq!(found_b.len(), 1);
    assert_eq!(found_b[0].node_id, "n2");
}
```

- [ ] **Step 2: Confirm fails**

```bash
cargo test -p ai-engine-cluster --test discovery_loopback 2>&1 | tail -10
# Expected: compile errors — DiscoveredWorker, discover_workers don't exist.
```

- [ ] **Step 3: Implement `discover.rs`**

```rust
use crate::discovery::txt::{TxtRecords, SERVICE_TYPE};
use mdns_sd::{ServiceDaemon, ServiceEvent};
use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct DiscoveredWorker {
    pub node_id: String,
    pub addr: SocketAddr,
    pub fingerprint: String,
    pub backend: String,
}

/// Browse for `_ai-engine._tcp.local.` services and return endpoints matching
/// `cluster_id`. Returns up to `expected_count` workers OR whatever was found
/// when `timeout` fires, whichever comes first. Empty result is fine —
/// caller decides how to react.
pub async fn discover_workers(
    cluster_id: &str,
    expected_count: usize,
    timeout: Duration,
) -> anyhow::Result<Vec<DiscoveredWorker>> {
    let daemon = ServiceDaemon::new()
        .map_err(|e| anyhow::anyhow!("mdns daemon: {e}"))?;
    let receiver = daemon.browse(SERVICE_TYPE)
        .map_err(|e| anyhow::anyhow!("mdns browse: {e}"))?;

    let mut found: HashMap<String, DiscoveredWorker> = HashMap::new();
    let deadline = tokio::time::Instant::now() + timeout;

    loop {
        if found.len() >= expected_count { break; }
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() { break; }

        // mdns-sd's receiver is a flume channel; convert recv to tokio-friendly
        // wait via spawn_blocking or use the channel's `recv_timeout`. The cleanest
        // path is `tokio::task::spawn_blocking` with the channel's blocking recv.
        let recv_clone = receiver.clone();
        let r = tokio::task::spawn_blocking(move || {
            recv_clone.recv_timeout(remaining)
        }).await;

        let event = match r {
            Ok(Ok(ev)) => ev,
            Ok(Err(_timeout)) => break,    // mDNS recv timeout = no more events for now
            Err(_join) => break,
        };

        match event {
            ServiceEvent::ServiceResolved(info) => {
                let txt_map: HashMap<String, String> = info.get_properties()
                    .iter()
                    .map(|prop| (prop.key().to_string(), prop.val_str().to_string()))
                    .collect();
                let txt = match TxtRecords::from_map(&txt_map) {
                    Ok(t) => t,
                    Err(_) => continue,
                };
                if txt.cluster_id != cluster_id { continue; }
                if txt.role != "worker" { continue; }
                // First-seen TOFU semantics: ignore later announcements for same node_id.
                if found.contains_key(&txt.node_id) { continue; }

                // Pick the first IPv4 address from the service info.
                let Some(ip) = info.get_addresses().iter().find_map(|a| match a {
                    IpAddr::V4(v) => Some(IpAddr::V4(*v)),
                    _ => None,
                }) else { continue; };
                let port = info.get_port();

                found.insert(txt.node_id.clone(), DiscoveredWorker {
                    node_id: txt.node_id,
                    addr: SocketAddr::new(ip, port),
                    fingerprint: txt.fingerprint,
                    backend: txt.backend,
                });
            }
            _ => {}    // other events (search started, host resolved, etc.) — ignore
        }
    }

    let _ = daemon.shutdown();
    let mut out: Vec<DiscoveredWorker> = found.into_values().collect();
    out.sort_by(|a, b| a.node_id.cmp(&b.node_id));
    Ok(out)
}
```

The exact API for accessing TXT records via `ServiceInfo::get_properties()` and the IPv4 form via `get_addresses()` may differ slightly between mdns-sd minor versions. Verify with `cargo doc -p mdns-sd`. The pattern is: `ServiceResolved` events carry a `ServiceInfo` with TXT, addresses, port.

- [ ] **Step 4: Run + verify**

```bash
cargo test -p ai-engine-cluster --test discovery_loopback -- --nocapture
```

mDNS loopback can be timing-sensitive. If a test fails with "expected 2, found 1" intermittently:
- Increase the propagation sleep to 500ms.
- Bump the discover timeout to 10 seconds.
- mDNS uses multicast 224.0.0.251 — on some Docker containers / VMs, multicast is disabled. Mark `#[ignore]` if the dev environment lacks multicast.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(cluster): Discoverer for mDNS-based worker enumeration"
```

NO Co-Authored-By.

---

### Task 5: Config schema — `[[cluster.discover]]` block

**Files:**
- Modify: `crates/ai-engine-config/src/lib.rs`
- Modify: `crates/ai-engine-config/src/validate.rs`
- Modify: `crates/ai-engine-config/tests/load.rs`

A new TOML block that tells the leader to discover workers via mDNS instead of using static `[[cluster.node]]` entries. Both can coexist (static = fingerprint-pinned, discover = TOFU).

- [ ] **Step 1: Failing tests**

Append to `crates/ai-engine-config/tests/load.rs`:

```rust
#[test]
fn parses_cluster_discover_block() {
    let toml = r#"
[server]
bind = "127.0.0.1:0"

[auth]
mode = "passthrough"

[[cluster]]
id = "home"
leader = "node-a"
quic_bind = "0.0.0.0:7700"

[cluster.model]
id = "llama-3-70b"
config_path = "/srv/models/llama-3-70b/config.json"
weights_path = "/srv/models/llama-3-70b/model.safetensors"
tokenizer_path = "/srv/models/llama-3-70b/tokenizer.json"

[cluster.discover]
expected_workers = 2
timeout_secs = 30

[[cluster.node]]
id = "node-a"
addr = "127.0.0.1:7700"
cert_fingerprint = "sha256:abc123"
backend = "cuda"

[[provider]]
id = "home-cluster"
kind = "local-cluster"
cluster = "home"

[[route]]
match = { model = "llama-3-70b" }
provider = "home-cluster"

[pipeline."/v1/chat/completions"]
stages = ["auth", "model_route", "forward", "log"]
"#;
    let cfg = ai_engine_config::Config::from_str(toml).unwrap();
    let cluster = &cfg.clusters[0];
    let disc = cluster.discover.as_ref().expect("cluster.discover present");
    assert_eq!(disc.expected_workers, 2);
    assert_eq!(disc.timeout_secs, 30);
}

#[test]
fn cluster_discover_defaults_timeout() {
    let toml = r#"
[server]
bind = "127.0.0.1:0"
[auth]
mode = "passthrough"

[[cluster]]
id = "home"
leader = "node-a"
quic_bind = "0.0.0.0:7700"

[cluster.model]
id = "m"
config_path = "x"
weights_path = "x"
tokenizer_path = "x"

[cluster.discover]
expected_workers = 3

[[cluster.node]]
id = "node-a"
addr = "127.0.0.1:7700"
cert_fingerprint = "sha256:abc"
backend = "cpu"

[[provider]]
id = "p"
kind = "local-cluster"
cluster = "home"

[pipeline."/v1/chat/completions"]
stages = ["forward", "log"]
"#;
    let cfg = ai_engine_config::Config::from_str(toml).unwrap();
    let disc = cfg.clusters[0].discover.as_ref().unwrap();
    assert_eq!(disc.expected_workers, 3);
    assert_eq!(disc.timeout_secs, 30);   // default
}

#[test]
fn cluster_discover_with_zero_workers_rejected() {
    let toml = r#"
[server]
bind = "127.0.0.1:0"
[auth]
mode = "passthrough"

[[cluster]]
id = "home"
leader = "node-a"
quic_bind = "0.0.0.0:7700"

[cluster.model]
id = "m"
config_path = "x"
weights_path = "x"
tokenizer_path = "x"

[cluster.discover]
expected_workers = 0

[[cluster.node]]
id = "node-a"
addr = "127.0.0.1:7700"
cert_fingerprint = "sha256:abc"
backend = "cpu"

[[provider]]
id = "p"
kind = "local-cluster"
cluster = "home"

[pipeline."/v1/chat/completions"]
stages = ["forward", "log"]
"#;
    let err = ai_engine_config::Config::from_str(toml).unwrap_err().to_string();
    assert!(err.to_lowercase().contains("expected_workers"), "got: {err}");
}
```

- [ ] **Step 2: Implement schema**

In `crates/ai-engine-config/src/lib.rs`, add to `Cluster`:

```rust
#[derive(Debug, Clone, Deserialize)]
pub struct Cluster {
    pub id: String,
    pub leader: String,
    pub quic_bind: String,
    #[serde(default = "default_protocol_version")]
    pub protocol_version: u16,
    #[serde(default = "default_join_timeout")]
    pub join_timeout_secs: u64,
    #[serde(default = "default_heartbeat")]
    pub heartbeat_interval_secs: u64,
    pub model: ClusterModel,
    #[serde(default, rename = "node")]
    pub nodes: Vec<ClusterNode>,
    #[serde(default, rename = "partition_override")]
    pub partition_override: Vec<PartitionOverride>,
    #[serde(default)]
    pub discover: Option<ClusterDiscover>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ClusterDiscover {
    pub expected_workers: usize,
    #[serde(default = "default_discover_timeout")]
    pub timeout_secs: u64,
}
fn default_discover_timeout() -> u64 { 30 }
```

- [ ] **Step 3: Validation**

In `crates/ai-engine-config/src/validate.rs`, inside the cluster-validation loop:

```rust
if let Some(disc) = &cluster.discover {
    if disc.expected_workers == 0 {
        anyhow::bail!(
            "cluster `{}` discover.expected_workers must be > 0",
            cluster.id
        );
    }
}
```

- [ ] **Step 4: Verify + commit**

```bash
cargo test -p ai-engine-config
cargo clippy --workspace --all-targets -- -D warnings
git add -A
git commit -m "feat(config): [[cluster.discover]] schema for mDNS auto-discovery"
```

NO Co-Authored-By.

---

### Task 6: `LeaderConfig::from_discovered` constructor + leader integration

**Files:**
- Modify: `crates/ai-engine-cluster/src/leader.rs`
- Create: `crates/ai-engine-cluster/tests/discovery_cluster.rs`

The existing `LeaderConfig` takes a `Vec<WorkerEndpoint>` directly. Add a constructor that converts `Vec<DiscoveredWorker>` to that.

- [ ] **Step 1: Implement converter**

In `crates/ai-engine-cluster/src/leader.rs`:

```rust
use crate::discovery::DiscoveredWorker;

impl WorkerEndpoint {
    /// Build a WorkerEndpoint from an mDNS-discovered worker.
    pub fn from_discovered(d: DiscoveredWorker) -> Self {
        Self {
            node_id: d.node_id,
            addr: d.addr,
            fingerprint: d.fingerprint,
        }
    }
}

impl LeaderConfig {
    /// Build a LeaderConfig where worker endpoints come from mDNS discovery
    /// rather than static TOML entries.
    pub fn from_discovered(
        cluster_id: impl Into<String>,
        leader_node_id: impl Into<String>,
        model_id: impl Into<String>,
        n_layers: usize,
        workers: Vec<DiscoveredWorker>,
        partition_override: Option<Vec<(String, std::ops::Range<usize>)>>,
    ) -> Self {
        Self {
            cluster_id: cluster_id.into(),
            leader_node_id: leader_node_id.into(),
            model_id: model_id.into(),
            n_layers,
            layer_bytes: 256 * 1024,
            embed_output_bytes: 256 * 1024,
            per_node_overhead: 64 * 1024,
            workers: workers.into_iter().map(WorkerEndpoint::from_discovered).collect(),
            partition_override,
        }
    }
}
```

- [ ] **Step 2: End-to-end cluster test with mDNS**

`crates/ai-engine-cluster/tests/discovery_cluster.rs`:

```rust
//! 3-node cluster started entirely via mDNS — no fingerprints in any test config.
//!
//! Each "worker" is a tokio task that:
//!   1. Generates its identity (cert + fingerprint).
//!   2. Announces itself via mDNS with the fingerprint as a TXT record.
//!   3. Starts its QUIC server endpoint at a random port.
//!   4. Runs `run_worker_full` (waits for Assignment, etc.).
//!
//! The leader:
//!   1. Calls `discover_workers` to find both workers.
//!   2. Builds a `LeaderConfig` from the discovered endpoints.
//!   3. Calls `ClusterLeader::start(...)` as usual — the rest is unchanged.

use std::net::IpAddr;
use std::path::PathBuf;
use std::time::Duration;

type B = burn_ndarray::NdArray;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .join("ai-engine-runtime/fixtures/toy-llama-3")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn mdns_cluster_generates_chat_completion() {
    let cluster_id = "mdns-test-cluster";

    let fix = fixture();
    let cfg = ai_engine_runtime::config::ModelConfig::from_file(&fix.join("config.json")).unwrap();
    let tok = ai_engine_tokenizer::HfTokenizer::from_path(fix.join("tokenizer.json")).unwrap();

    // --- Worker 1 ---
    let w1_id = ai_engine_cluster::tls::generate_node_identity("w1").unwrap();
    let w1_ep = ai_engine_cluster::transport::quic::server_endpoint(
        &w1_id, "127.0.0.1:0".parse().unwrap(),
    ).unwrap();
    let w1_port = w1_ep.local_addr().unwrap().port();
    let w1_txt = ai_engine_cluster::discovery::TxtRecords {
        cluster_id: cluster_id.into(),
        node_id: "w1".into(),
        role: "worker".into(),
        protocol_version: 1,
        fingerprint: w1_id.fingerprint.clone(),
        backend: "cpu".into(),
    };
    let _w1_ann = ai_engine_cluster::discovery::Announcer::register(
        IpAddr::V4("127.0.0.1".parse().unwrap()),
        w1_port,
        "w1.local.",
        w1_txt,
    ).unwrap();
    let model_path = fix.join("model.safetensors");
    let cfg_w1 = cfg.clone();
    let mp1 = model_path.clone();
    tokio::spawn(async move {
        ai_engine_cluster::worker::run_worker_full::<B>(
            w1_ep, "w1".into(),
            ai_engine_cluster::capability::BackendKind::Cpu,
            mp1, cfg_w1,
        ).await
    });

    // --- Worker 2 ---
    let w2_id = ai_engine_cluster::tls::generate_node_identity("w2").unwrap();
    let w2_ep = ai_engine_cluster::transport::quic::server_endpoint(
        &w2_id, "127.0.0.1:0".parse().unwrap(),
    ).unwrap();
    let w2_port = w2_ep.local_addr().unwrap().port();
    let w2_txt = ai_engine_cluster::discovery::TxtRecords {
        cluster_id: cluster_id.into(),
        node_id: "w2".into(),
        role: "worker".into(),
        protocol_version: 1,
        fingerprint: w2_id.fingerprint.clone(),
        backend: "cpu".into(),
    };
    let _w2_ann = ai_engine_cluster::discovery::Announcer::register(
        IpAddr::V4("127.0.0.1".parse().unwrap()),
        w2_port,
        "w2.local.",
        w2_txt,
    ).unwrap();
    let cfg_w2 = cfg.clone();
    let mp2 = model_path.clone();
    tokio::spawn(async move {
        ai_engine_cluster::worker::run_worker_full::<B>(
            w2_ep, "w2".into(),
            ai_engine_cluster::capability::BackendKind::Cpu,
            mp2, cfg_w2,
        ).await
    });

    // Let mDNS propagate.
    tokio::time::sleep(Duration::from_millis(500)).await;

    // --- Leader: discover, then start ---
    let discovered = ai_engine_cluster::discovery::discover_workers(
        cluster_id, 2, Duration::from_secs(10),
    ).await.unwrap();
    assert_eq!(discovered.len(), 2, "expected 2 discovered workers, got {}", discovered.len());

    let leader_id = ai_engine_cluster::tls::generate_node_identity("leader").unwrap();
    let lcfg = ai_engine_cluster::leader::LeaderConfig::from_discovered(
        cluster_id,
        "leader",
        "toy-mdns",
        cfg.n_layers,
        discovered,
        Some(vec![("w1".into(), 0..2), ("w2".into(), 2..4)]),
    );
    let leader = ai_engine_cluster::leader::ClusterLeader::start(leader_id, lcfg).await.unwrap();

    let prompt = "The quick brown fox";
    let ids: Vec<u32> = ai_engine_tokenizer::Tokenizer::encode(&tok, prompt).unwrap();
    let ids_i32: Vec<i32> = ids.iter().map(|x| *x as i32).collect();
    let tokens = leader.generate::<B>(
        &model_path, &cfg, 0..0, &ids_i32, 3,
        ai_engine_runtime::sample::SamplingConfig {
            temperature: 0.0, top_p: None, top_k: None, seed: 0,
        },
    ).await.unwrap();

    assert_eq!(tokens.len(), 3, "expected 3 generated tokens");
}
```

- [ ] **Step 3: Run + iterate**

```bash
cargo test -p ai-engine-cluster --test discovery_cluster -- --nocapture
```

Expected: passes within ~3-5 seconds. If it hangs at `discover_workers`, mDNS propagation isn't working — bump the sleep before discover to 1 second.

- [ ] **Step 4: Commit**

```bash
cargo clippy --workspace --all-targets -- -D warnings
git add -A
git commit -m "feat(cluster): LeaderConfig::from_discovered + end-to-end mDNS cluster test"
```

NO Co-Authored-By.

---

### Task 7: Worker binary announces; leader build_app_state uses discovery

**Files:**
- Modify: `crates/ai-engine/src/worker_main.rs` (workers announce before serving)
- Modify: `crates/ai-engine/src/app.rs` (leader-mode discovers workers when [[cluster.discover]] is set)

- [ ] **Step 1: Worker announces**

In `crates/ai-engine/src/worker_main.rs`, after `generate_node_identity` and before `run_worker_full`, register an Announcer:

```rust
use ai_engine_cluster::discovery::{Announcer, TxtRecords};
use std::net::IpAddr;

pub async fn run_worker(
    cfg: &ai_engine_config::Config,
    node_id: &str,
    cluster_id: &str,
) -> anyhow::Result<()> {
    let cluster = cfg.clusters.iter()
        .find(|c| c.id == cluster_id)
        .ok_or_else(|| anyhow::anyhow!("cluster `{cluster_id}` not found in config"))?;
    let me = cluster.nodes.iter()
        .find(|n| n.id == node_id)
        .ok_or_else(|| anyhow::anyhow!("node `{node_id}` not in cluster `{cluster_id}`"))?;

    let identity = ai_engine_cluster::tls::load_or_generate_node_identity(
        node_id,
        &dirs::home_dir().unwrap_or_default().join(".ai-engine"),
    )?;
    eprintln!("ai-engine worker `{}` fingerprint: {}", node_id, identity.fingerprint);

    let bind: std::net::SocketAddr = me.addr.parse()?;
    let endpoint = ai_engine_cluster::transport::quic::server_endpoint(&identity, bind)?;

    // mDNS announcement — runs alongside the QUIC server.
    let txt = TxtRecords {
        cluster_id: cluster_id.into(),
        node_id: node_id.into(),
        role: "worker".into(),
        protocol_version: 1,
        fingerprint: identity.fingerprint.clone(),
        backend: me.backend.clone(),
    };
    let ann_ip: IpAddr = bind.ip();
    let ann_host = format!("{}.local.", node_id);
    let _announcer = Announcer::register(ann_ip, bind.port(), &ann_host, txt)?;
    eprintln!("ai-engine worker `{}` announcing on mDNS", node_id);

    let model_cfg = ai_engine_runtime::config::ModelConfig::from_file(
        std::path::Path::new(&cluster.model.config_path)
    )?;
    let model_path: std::path::PathBuf = (&cluster.model.weights_path).into();

    let backend = match me.backend.as_str() {
        "cpu" => ai_engine_cluster::capability::BackendKind::Cpu,
        "cuda" => ai_engine_cluster::capability::BackendKind::Cuda,
        "metal" => ai_engine_cluster::capability::BackendKind::Metal,
        "wgpu" => ai_engine_cluster::capability::BackendKind::Wgpu,
        other => anyhow::bail!("unknown backend kind: {other}"),
    };

    ai_engine_cluster::worker::run_worker_full::<burn_ndarray::NdArray>(
        endpoint, node_id.to_string(), backend, model_path, model_cfg,
    ).await
}
```

- [ ] **Step 2: Leader uses discovery when configured**

In `crates/ai-engine/src/app.rs`, inside the `NodeRole::Leader` branch of `build_app_state`, before constructing `WorkerEndpoint`s from static `cluster.nodes`, check whether `cluster.discover` is set:

```rust
let worker_endpoints: Vec<ai_engine_cluster::leader::WorkerEndpoint> = if let Some(disc) = &cluster_cfg.discover {
    // mDNS discovery path.
    let timeout = std::time::Duration::from_secs(disc.timeout_secs);
    tracing::info!(cluster_id = %cluster_id, expected = disc.expected_workers,
        timeout_secs = disc.timeout_secs, "discovering workers via mDNS");
    let discovered = ai_engine_cluster::discovery::discover_workers(
        cluster_id, disc.expected_workers, timeout,
    ).await?;
    if discovered.is_empty() {
        anyhow::bail!("mDNS discovery yielded zero workers for cluster `{cluster_id}` within {} sec", disc.timeout_secs);
    }
    tracing::info!(cluster_id = %cluster_id, found = discovered.len(),
        "mDNS discovery complete");
    discovered.into_iter()
        .map(ai_engine_cluster::leader::WorkerEndpoint::from_discovered)
        .collect()
} else {
    // Static config path (existing behavior).
    cluster_cfg.nodes.iter()
        .filter(|n| n.id != cluster_cfg.leader)
        .map(|n| ai_engine_cluster::leader::WorkerEndpoint {
            node_id: n.id.clone(),
            addr: n.addr.parse().expect("addr"),
            fingerprint: n.cert_fingerprint.clone(),
        })
        .collect()
};
```

The rest of the leader-mode flow (LeaderConfig construction, ClusterLeader::start, etc.) is unchanged — it just gets a different source of worker endpoints.

- [ ] **Step 3: Verify the existing multiproc smoke test still works**

The existing `crates/ai-engine/tests/multiproc_smoke.rs` uses STATIC fingerprints in its TOML. The change in app.rs makes mDNS only fire when `[[cluster.discover]]` is present, so static-config smokes are unaffected.

```bash
cargo build --workspace --release
cargo test --workspace
cargo test -p ai-engine --test multiproc_smoke -- --ignored --nocapture
cargo test -p ai-engine --test streaming_smoke -- --ignored --nocapture
```

All must pass.

- [ ] **Step 4: Commit**

```bash
cargo clippy --workspace --all-targets -- -D warnings
git add -A
git commit -m "feat(bin): workers announce on mDNS; leader discovers when configured"
```

NO Co-Authored-By.

---

### Task 8: Multi-process mDNS smoke test

**Files:**
- Create: `crates/ai-engine/tests/multiproc_smoke_mdns.rs`

A real cross-process test: 3 OS processes, no fingerprints in config, mDNS does all the discovery.

- [ ] **Step 1: Test**

`crates/ai-engine/tests/multiproc_smoke_mdns.rs`:

```rust
//! Multi-process mDNS-discovery smoke. The leader's config has NO worker
//! fingerprints; workers announce themselves and the leader finds them.
//!
//! Each child process runs ai-engine binary; the test orchestrates lifecycle.

use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

const FIXTURE_PATH: &str = "../ai-engine-runtime/fixtures/toy-llama-3";

fn fixture_abspath() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(FIXTURE_PATH)
        .canonicalize()
        .expect("fixture canonicalize")
}

fn free_port() -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let p = l.local_addr().unwrap().port();
    drop(l);
    p
}

fn release_binary() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/release/ai-engine")
        .canonicalize()
        .expect("release binary — run `cargo build --release` first")
}

fn write_config(
    dir: &std::path::Path,
    fix: &std::path::Path,
    leader_http_port: u16,
    leader_quic_port: u16,
    w1_quic_port: u16,
    w2_quic_port: u16,
) -> PathBuf {
    let toml = format!(
        r#"
[server]
bind = "127.0.0.1:{leader_http_port}"
log_format = "pretty"
log_level = "warn"

[auth]
mode = "passthrough"

[[cluster]]
id = "smoke-mdns"
leader = "leader"
quic_bind = "127.0.0.1:0"

[cluster.model]
id = "toy-llama"
config_path = "{fix_path}/config.json"
weights_path = "{fix_path}/model.safetensors"
tokenizer_path = "{fix_path}/tokenizer.json"

[cluster.discover]
expected_workers = 2
timeout_secs = 15

[[cluster.partition_override]]
node = "worker-1"
layers = "0..2"

[[cluster.partition_override]]
node = "worker-2"
layers = "2..4"

[[cluster.node]]
id = "leader"
addr = "127.0.0.1:{leader_quic_port}"
cert_fingerprint = "sha256:0000000000000000000000000000000000000000000000000000000000000000"
backend = "cpu"

[[cluster.node]]
id = "worker-1"
addr = "127.0.0.1:{w1_quic_port}"
cert_fingerprint = "sha256:0000000000000000000000000000000000000000000000000000000000000000"
backend = "cpu"

[[cluster.node]]
id = "worker-2"
addr = "127.0.0.1:{w2_quic_port}"
cert_fingerprint = "sha256:0000000000000000000000000000000000000000000000000000000000000000"
backend = "cpu"

[[provider]]
id = "smoke-cluster"
kind = "local-cluster"
cluster = "smoke-mdns"

[[route]]
match = {{ model = "toy-llama" }}
provider = "smoke-cluster"

[pipeline."/v1/chat/completions"]
stages = ["auth", "model_route", "forward", "log"]
"#,
        fix_path = fix.display(),
    );
    let path = dir.join("ai-engine.toml");
    std::fs::write(&path, toml).unwrap();
    path
}

#[test]
#[ignore = "multi-process mDNS smoke; requires release build + LAN multicast; run with --ignored"]
fn three_process_cluster_with_mdns_discovery_serves_chat() {
    let bin = release_binary();
    let fix = fixture_abspath();
    let leader_http_port = free_port();
    let leader_quic_port = free_port();
    let w1_quic_port = free_port();
    let w2_quic_port = free_port();

    let workdir = tempfile::tempdir().unwrap();
    let leader_home = workdir.path().join("leader-home");
    let w1_home = workdir.path().join("w1-home");
    let w2_home = workdir.path().join("w2-home");
    std::fs::create_dir_all(&leader_home).unwrap();
    std::fs::create_dir_all(&w1_home).unwrap();
    std::fs::create_dir_all(&w2_home).unwrap();

    let cfg_path = write_config(
        workdir.path(), &fix,
        leader_http_port, leader_quic_port, w1_quic_port, w2_quic_port,
    );

    // Spawn workers (they announce on mDNS).
    let mut w1 = Command::new(&bin)
        .arg("--config").arg(&cfg_path)
        .arg("--node-id").arg("worker-1")
        .env("HOME", &w1_home)
        .stderr(Stdio::piped())
        .stdout(Stdio::null())
        .spawn().unwrap();
    let mut w2 = Command::new(&bin)
        .arg("--config").arg(&cfg_path)
        .arg("--node-id").arg("worker-2")
        .env("HOME", &w2_home)
        .stderr(Stdio::piped())
        .stdout(Stdio::null())
        .spawn().unwrap();

    // Give the workers time to announce on mDNS.
    std::thread::sleep(Duration::from_secs(1));

    // Spawn leader — it discovers workers via mDNS, ignores the placeholder fingerprints in [[cluster.node]].
    let mut leader = Command::new(&bin)
        .arg("--config").arg(&cfg_path)
        .arg("--node-id").arg("leader")
        .env("HOME", &leader_home)
        .stderr(Stdio::piped())
        .stdout(Stdio::null())
        .spawn().unwrap();

    let leader_url = format!("http://127.0.0.1:{leader_http_port}");
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(2))
        .build().unwrap();

    let mut ready = false;
    for _ in 0..100 {
        std::thread::sleep(Duration::from_millis(200));
        if let Ok(r) = client.get(format!("{leader_url}/healthz")).send() {
            if r.status().as_u16() == 200 { ready = true; break; }
        }
    }
    assert!(ready, "leader didn't become ready within 20s");

    let response = client.post(format!("{leader_url}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "toy-llama",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 3
        }))
        .send().expect("POST chat completion");
    assert_eq!(response.status().as_u16(), 200);
    let body: serde_json::Value = response.json().expect("JSON body");
    let usage = &body["usage"];
    assert_eq!(usage["completion_tokens"], 3, "expected exactly 3 completion tokens");

    let _ = leader.kill(); let _ = w1.kill(); let _ = w2.kill();
    let _ = leader.wait(); let _ = w1.wait(); let _ = w2.wait();
}
```

Critical note about the placeholder fingerprints in the static `[[cluster.node]]` entries: the leader uses `[[cluster.discover]]` (overriding) to find workers, so the static fingerprints are unused. But the config schema still requires `cert_fingerprint` to be present on each `[[cluster.node]]` (validator gates on it). The placeholders satisfy the validator without affecting the runtime path.

This is awkward — we should ideally make `cert_fingerprint` optional when `[[cluster.discover]]` is set. **Note as a follow-up cleanup**, not blocking Plan 8.

- [ ] **Step 2: Run + commit**

```bash
cargo build --workspace --release
cargo test -p ai-engine --test multiproc_smoke_mdns -- --ignored --nocapture
```

If the test fails because the runner lacks mDNS multicast (common on CI), it's marked `#[ignore]` and explicitly run with `--ignored` — failure here doesn't break the default workspace test.

```bash
git add -A
git commit -m "test(smoke): 3-process cluster with mDNS discovery (no fingerprints in config)"
```

NO Co-Authored-By.

---

### Task 9: README + tag v0.3.0-alpha.4

**Files:**
- Modify: `README.md`
- Tag: `v0.3.0-alpha.4`

- [ ] **Step 1: Final verification**

```bash
cd /home/alessio/aip/airproxy
cargo test --workspace 2>&1 | grep "test result" | awk '{p += $4; ig += $8} END {print "PASSED=" p " IGNORED=" ig}'
cargo clippy --workspace --all-targets -- -D warnings
cargo build --workspace --release
cargo test -p ai-engine --test multiproc_smoke -- --ignored --nocapture 2>&1 | tail -5
cargo test -p ai-engine --test streaming_smoke -- --ignored --nocapture 2>&1 | tail -5
cargo test -p ai-engine --test multiproc_smoke_mdns -- --ignored --nocapture 2>&1 | tail -5
```

- [ ] **Step 2: README**

Append:

```markdown
### v0.3.0-alpha.4 — mDNS auto-discovery

ai-engine v0.3.0-alpha.4 lets cluster nodes find each other on the LAN
via mDNS. No more pasting cert fingerprints into every `[[cluster.node]]`
block.

How it works:
- Workers announce themselves on startup with TXT records: cluster_id,
  node_id, role=worker, protocol_version, fingerprint, backend.
- The leader, when `[[cluster.discover]]` is set in its config, browses
  for `_ai-engine._tcp.local.` services and TOFU-pins the announced
  fingerprints.
- The existing static `[[cluster.node]]` path is unchanged for deployments
  that prefer explicit configuration.

Config:

\`\`\`toml
[[cluster]]
id = "home-lab"
leader = "leader"
quic_bind = "0.0.0.0:7700"

[cluster.discover]
expected_workers = 2
timeout_secs = 30

[cluster.model]
id = "llama-3-70b"
# ...
\`\`\`

Known limitations:
- TOFU only on first announcement; mismatching announcements for the
  same node_id are ignored (not error-reported).
- Dynamic membership not supported — workers joining a running cluster
  still require restart. Deferred.
- mDNS depends on LAN multicast; some Docker setups + restrictive
  networks disable it. The `multiproc_smoke_mdns` test is `#[ignore]`d
  to keep CI portable.
```

- [ ] **Step 3: Commit + tag**

```bash
git add README.md
git commit -m "docs: announce v0.3.0-alpha.4 mDNS auto-discovery release"
git tag v0.3.0-alpha.4
git log --oneline -5
git tag
```

NO Co-Authored-By.

## Report
- Status
- Test count + ignored
- All verification exit codes
- Final `git tag` listing
- `git log --oneline -5`

---

## Self-review

**Spec coverage:**
- mDNS service announcement (TXT records) → Tasks 2, 3
- mDNS service discovery → Task 4
- `[[cluster.discover]]` config block → Task 5
- Leader integration (build_app_state uses discovery when configured) → Task 7
- Workers announce via the binary → Task 7
- End-to-end multi-process test with no fingerprints in config → Task 8

**Placeholder scan:**

The plan has no `TBD` / `fill in later` markers. The follow-up note in Task 8 about making `cert_fingerprint` optional when `[[cluster.discover]]` is set is a future-work note, not a placeholder — the current schema still works (operators put placeholder zeros in the discover case, which the runtime ignores).

The Task 3 step that says "no standalone test for the Announcer" is intentional — it's not a placeholder, it's a deliberate decision: the round-trip with Discoverer in Task 4 is the meaningful test.

**Type consistency:**
- `TxtRecords` (Task 2) → consumed by `Announcer::register` (Task 3) and produced by `DiscoveredWorker` derivation in `discover_workers` (Task 4). ✓
- `DiscoveredWorker` (Task 4) → consumed by `WorkerEndpoint::from_discovered` and `LeaderConfig::from_discovered` (Task 6). ✓
- `ClusterDiscover` config (Task 5) → consumed by leader-mode `build_app_state` (Task 7). Fields match. ✓
- `SERVICE_TYPE` constant (Task 2) → used by both Announcer and Discoverer. ✓

**Acknowledged risks:**

1. **mDNS multicast on test runners.** Local dev usually works; cloud CI without multicast permission breaks. Mitigated by `#[ignore]` annotation on the multi-process mDNS smoke. The in-process loopback tests (Task 4) bypass UDP entirely via mDNS daemon's internal loopback and should work everywhere.
2. **mdns-sd API churn.** Version 0.13's exact API for TXT records and IP enumeration may differ. The plan documents the algorithm; the implementer adapts to the actual signatures.
3. **TOFU on contradictory announcements** — if a worker restarts with a new identity but the leader has cached the old fingerprint, connection will fail. Not unique to mDNS (same applies to static config when the cert rotates), and persistent-identity (Plan 3) mitigates it.

---

## Execution Handoff

Plan 8 saved to `docs/superpowers/plans/2026-05-24-plan-8-mdns-discovery.md`. 9 tasks.

**Subagent-Driven (recommended)** — Tasks 1, 2 are bounded primitives. Task 3 is the Announcer. Task 4 is the Discoverer + loopback test (high-value gate). Task 5 is config. Task 6 is the in-process cluster integration. Task 7 wires the binary. Task 8 is the cross-process smoke. Task 9 ships.

Plan 9 candidates after v0.3.0-alpha.4: wire GGUF loader into TOML config (model.gguf path), more GGUF ggml_types, or real-model validation.
