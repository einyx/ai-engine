use rcgen::{CertificateParams, KeyPair};
use sha2::{Digest, Sha256};

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

pub fn fingerprint_sha256(cert_der: &[u8]) -> String {
    let digest = Sha256::digest(cert_der);
    let mut out = String::with_capacity(7 + 64);
    out.push_str("sha256:");
    for byte in digest.iter() {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}
