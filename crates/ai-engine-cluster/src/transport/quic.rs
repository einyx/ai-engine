use crate::tls::NodeIdentity;
use quinn::{ClientConfig, Endpoint, ServerConfig};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use sha2::{Digest, Sha256};
use std::net::SocketAddr;
use std::sync::Arc;

const ALPN: &[u8] = b"ai-engine-cluster/1";

pub fn server_endpoint(identity: &NodeIdentity, bind: SocketAddr) -> anyhow::Result<Endpoint> {
    let _ = rustls::crypto::ring::default_provider().install_default();

    let cert_chain = vec![CertificateDer::from(identity.cert_der.clone())];
    let key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(identity.key_der.clone()));

    let mut rustls_cfg = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(cert_chain, key)
        .map_err(|e| anyhow::anyhow!("server tls: {e}"))?;
    rustls_cfg.alpn_protocols = vec![ALPN.to_vec()];

    let server_cfg = ServerConfig::with_crypto(Arc::new(
        quinn::crypto::rustls::QuicServerConfig::try_from(rustls_cfg)
            .map_err(|e| anyhow::anyhow!("quinn server cfg: {e}"))?,
    ));
    Endpoint::server(server_cfg, bind).map_err(|e| anyhow::anyhow!("bind: {e}"))
}

pub fn client_endpoint(
    identity: &NodeIdentity,
    trusted_fingerprints: &[String],
) -> anyhow::Result<Endpoint> {
    let _ = rustls::crypto::ring::default_provider().install_default();

    let cert_chain = vec![CertificateDer::from(identity.cert_der.clone())];
    let key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(identity.key_der.clone()));

    let verifier = Arc::new(FingerprintVerifier {
        trusted: trusted_fingerprints.iter().cloned().collect(),
    });

    let mut rustls_cfg = rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(verifier)
        .with_client_auth_cert(cert_chain, key)
        .map_err(|e| anyhow::anyhow!("client tls: {e}"))?;
    rustls_cfg.alpn_protocols = vec![ALPN.to_vec()];

    let client_cfg = ClientConfig::new(Arc::new(
        quinn::crypto::rustls::QuicClientConfig::try_from(rustls_cfg)
            .map_err(|e| anyhow::anyhow!("quinn client cfg: {e}"))?,
    ));

    let mut endpoint = Endpoint::client("0.0.0.0:0".parse().unwrap())
        .map_err(|e| anyhow::anyhow!("bind client: {e}"))?;
    endpoint.set_default_client_config(client_cfg);
    Ok(endpoint)
}

#[derive(Debug)]
struct FingerprintVerifier {
    trusted: std::collections::HashSet<String>,
}

impl rustls::client::danger::ServerCertVerifier for FingerprintVerifier {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        let digest = Sha256::digest(end_entity.as_ref());
        let mut fp = String::from("sha256:");
        for b in digest.iter() {
            fp.push_str(&format!("{b:02x}"));
        }
        if self.trusted.contains(&fp) {
            Ok(rustls::client::danger::ServerCertVerified::assertion())
        } else {
            Err(rustls::Error::General(format!(
                "untrusted server fingerprint: {fp}"
            )))
        }
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        vec![
            rustls::SignatureScheme::ED25519,
            rustls::SignatureScheme::ECDSA_NISTP256_SHA256,
        ]
    }
}
