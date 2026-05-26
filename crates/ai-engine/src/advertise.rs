//! Ollama mDNS advertiser.
//!
//! Runs on a box hosting an Ollama instance (which itself emits nothing on
//! mDNS) and announces that endpoint under the `_ai-engine._tcp.local.` service
//! type with `role = "ollama"`, so ai-engine gateways on the LAN can
//! auto-discover and register it. See `ai_engine_cluster::discovery`.

use ai_engine_cluster::discovery::{Announcer, ROLE_OLLAMA};
use std::collections::HashMap;
use std::net::{IpAddr, UdpSocket};
use std::time::Duration;

/// How often to re-emit an unsolicited announcement. Kept short so a gateway's
/// startup browse window reliably overlaps at least one announcement even when
/// another mDNS responder (e.g. avahi) on this host contends for query traffic.
const ANNOUNCE_INTERVAL: Duration = Duration::from_secs(5);

/// Re-probe `/api/tags` every Nth announcement (≈30s) to refresh the model list.
const PROBE_EVERY: u32 = 6;

/// Advertise `ollama_url` on mDNS until the process is killed. Re-probes the
/// model list periodically and re-registers when it changes.
pub async fn run(ollama_url: String, label: Option<String>) -> anyhow::Result<()> {
    let ollama_url = ollama_url.trim_end_matches('/').to_string();
    let label = label.unwrap_or_else(|| {
        hostname::get()
            .ok()
            .and_then(|s| s.into_string().ok())
            .unwrap_or_else(|| "ollama".into())
    });
    let ip = primary_lan_ip()
        .ok_or_else(|| anyhow::anyhow!("could not determine a primary LAN IPv4 address"))?;
    let port = port_of(&ollama_url).unwrap_or(11434);
    let host_name = format!("{label}.local.");
    let instance_name = format!("ollama-{label}");
    // What we *advertise* must be reachable from other hosts — always the LAN
    // IP, never the loopback/hostname we use to probe locally.
    let advertised_url = format!("http://{ip}:{port}");

    let client = reqwest::Client::new();
    let mut models = probe_models(&client, &ollama_url).await.unwrap_or_default();
    eprintln!(
        "ai-engine advertising Ollama `{label}` as {advertised_url} on mDNS \
         (probing {ollama_url}, {} model(s))",
        models.len()
    );

    // Held only to keep the mDNS registration alive; dropping sends a goodbye.
    let mut announcer = register(ip, port, &host_name, &instance_name, &advertised_url, &models)?;

    let mut tick: u32 = 0;
    loop {
        tokio::time::sleep(ANNOUNCE_INTERVAL).await;
        tick = tick.wrapping_add(1);

        // Periodically refresh the model list from the local Ollama.
        if tick % PROBE_EVERY == 0 {
            match probe_models(&client, &ollama_url).await {
                Ok(m) => models = m,
                Err(e) => tracing::warn!(error = %e, "ollama /api/tags probe failed; keeping last model list"),
            }
        }

        // Re-emit the announcement so browsing gateways reliably catch it.
        // Replacing the registration re-broadcasts the unsolicited PTR/SRV/TXT.
        drop(announcer);
        announcer = register(ip, port, &host_name, &instance_name, &advertised_url, &models)?;
    }
}

fn register(
    ip: IpAddr,
    port: u16,
    host_name: &str,
    instance_name: &str,
    ollama_url: &str,
    models: &[String],
) -> anyhow::Result<Announcer> {
    let mut txt = HashMap::new();
    txt.insert("role".to_string(), ROLE_OLLAMA.to_string());
    txt.insert("ollama_url".to_string(), ollama_url.to_string());
    txt.insert("models".to_string(), models.join(","));
    txt.insert(
        "label".to_string(),
        instance_name.trim_start_matches("ollama-").to_string(),
    );
    txt.insert("protocol_version".to_string(), "1".to_string());
    // Host resources (sampled fresh each re-announce) so the gateway can show
    // this node's CPU/mem/disk. Same schema as rustyllm-serve.
    for (k, v) in ai_engine_core::resources::sample().to_txt() {
        txt.insert(k, v);
    }
    Announcer::register_raw(ip, port, host_name, instance_name, txt)
}

/// GET `<base>/api/tags` and return the advertised model names.
async fn probe_models(client: &reqwest::Client, base: &str) -> anyhow::Result<Vec<String>> {
    let body = client
        .get(format!("{base}/api/tags"))
        .timeout(Duration::from_secs(5))
        .send()
        .await?
        .error_for_status()?
        .text()
        .await?;
    let json: serde_json::Value = serde_json::from_str(&body)?;
    let mut models: Vec<String> = json
        .get("models")
        .and_then(|m| m.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m.get("name").and_then(|n| n.as_str()).map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    models.sort();
    Ok(models)
}

/// Determine the primary outbound LAN IPv4 by opening a UDP socket toward a
/// public address. No packets are sent; the kernel just resolves the route's
/// source address. Falls back to `None` if no IPv4 route exists.
fn primary_lan_ip() -> Option<IpAddr> {
    let sock = UdpSocket::bind("0.0.0.0:0").ok()?;
    sock.connect("8.8.8.8:80").ok()?;
    let ip = sock.local_addr().ok()?.ip();
    matches!(ip, IpAddr::V4(_)).then_some(ip)
}

fn port_of(url: &str) -> Option<u16> {
    url.rsplit(':').next()?.parse().ok()
}
