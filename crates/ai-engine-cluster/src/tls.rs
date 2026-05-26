use rcgen::{CertificateParams, KeyPair};
use sha2::{Digest, Sha256};
use std::path::Path;

pub struct NodeIdentity {
    pub cert_pem: String,
    pub key_pem: String,
    pub cert_der: Vec<u8>,
    pub key_der: Vec<u8>,
    pub fingerprint: String, // "sha256:<64 hex chars>"
}

pub fn generate_node_identity(node_id: &str) -> anyhow::Result<NodeIdentity> {
    let key = KeyPair::generate_for(&rcgen::PKCS_ED25519)
        .map_err(|e| anyhow::anyhow!("keypair: {e}"))?;

    let mut params = CertificateParams::new(vec![node_id.to_string()])
        .map_err(|e| anyhow::anyhow!("cert params: {e}"))?;
    params.distinguished_name.push(rcgen::DnType::CommonName, node_id);

    let cert = params
        .self_signed(&key)
        .map_err(|e| anyhow::anyhow!("self-sign: {e}"))?;

    let cert_pem = cert.pem();
    let key_pem = key.serialize_pem();
    let cert_der = cert.der().to_vec();
    let key_der = key.serialize_der();
    let fingerprint = fingerprint_sha256(&cert_der);

    Ok(NodeIdentity {
        cert_pem,
        key_pem,
        cert_der,
        key_der,
        fingerprint,
    })
}

/// Load this node's TLS identity from `<dir>/node.crt` and `<dir>/node.key` if
/// both files exist, otherwise generate a fresh identity and persist it to
/// disk for re-use on subsequent runs.
///
/// Persistence is required so that a node's fingerprint stays stable across
/// restarts — without it, peers would need their `cert_fingerprint` config
/// updated every time a node bounces.
pub fn load_or_generate_node_identity(node_id: &str, dir: &Path) -> anyhow::Result<NodeIdentity> {
    let cert_path = dir.join("node.crt");
    let key_path = dir.join("node.key");
    if cert_path.exists() && key_path.exists() {
        let cert_pem = std::fs::read_to_string(&cert_path)
            .map_err(|e| anyhow::anyhow!("read {}: {e}", cert_path.display()))?;
        let key_pem = std::fs::read_to_string(&key_path)
            .map_err(|e| anyhow::anyhow!("read {}: {e}", key_path.display()))?;
        let cert_der = pem::parse(&cert_pem)
            .map_err(|e| anyhow::anyhow!("parse cert PEM {}: {e}", cert_path.display()))?
            .into_contents();
        let key_der = pem::parse(&key_pem)
            .map_err(|e| anyhow::anyhow!("parse key PEM {}: {e}", key_path.display()))?
            .into_contents();
        let fingerprint = fingerprint_sha256(&cert_der);
        return Ok(NodeIdentity {
            cert_pem,
            key_pem,
            cert_der,
            key_der,
            fingerprint,
        });
    }
    let id = generate_node_identity(node_id)?;
    std::fs::create_dir_all(dir)
        .map_err(|e| anyhow::anyhow!("create_dir_all {}: {e}", dir.display()))?;
    std::fs::write(&cert_path, &id.cert_pem)
        .map_err(|e| anyhow::anyhow!("write {}: {e}", cert_path.display()))?;
    std::fs::write(&key_path, &id.key_pem)
        .map_err(|e| anyhow::anyhow!("write {}: {e}", key_path.display()))?;
    Ok(id)
}

pub fn fingerprint_sha256(cert_der: &[u8]) -> String {
    let digest = Sha256::digest(cert_der);
    let mut out = String::with_capacity(7 + 64);
    out.push_str("sha256:");
    for byte in digest.iter() {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}
