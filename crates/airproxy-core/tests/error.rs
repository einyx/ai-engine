use airproxy_core::error::GatewayError;

#[test]
fn status_codes_match_spec() {
    assert_eq!(GatewayError::BadRequest("x".into()).http_status(), 400);
    assert_eq!(GatewayError::Unauthorized.http_status(), 401);
    assert_eq!(GatewayError::PayloadTooLarge.http_status(), 413);
    assert_eq!(
        GatewayError::NoRouteForModel { model: "gpt-x".into() }.http_status(),
        502
    );
    assert_eq!(GatewayError::Internal(anyhow::anyhow!("oops")).http_status(), 500);
}

#[test]
fn provider_errors_map_to_correct_statuses() {
    use airproxy_provider::error::ProviderError;
    use bytes::Bytes;
    assert_eq!(
        GatewayError::Provider(ProviderError::Connect("dns".into())).http_status(),
        502
    );
    assert_eq!(
        GatewayError::Provider(ProviderError::Timeout).http_status(),
        504
    );
    assert_eq!(
        GatewayError::Provider(ProviderError::Status { status: 429, body: Bytes::new() }).http_status(),
        429
    );
    assert_eq!(
        GatewayError::Provider(ProviderError::InvalidResponse("bad json".into())).http_status(),
        502
    );
    assert_eq!(
        GatewayError::Provider(ProviderError::Stream("eof".into())).http_status(),
        502
    );
    assert_eq!(
        GatewayError::Provider(ProviderError::Unsupported).http_status(),
        502
    );
}
