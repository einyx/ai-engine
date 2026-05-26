use ai_engine_core::error::GatewayError;
use ai_engine_provider::error::ProviderError;
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

#[derive(Clone, Copy)]
pub enum Format {
    OpenAi,
    Anthropic,
}

/// Produce an HTTP response in the endpoint-native error envelope.
pub fn envelope(format: Format, err: &GatewayError) -> Response {
    // Provider Status passthrough: keep upstream status + raw body so the
    // caller's SDK sees the upstream's original error verbatim.
    if let GatewayError::Provider(ProviderError::Status { status, body }) = err {
        let status_code = StatusCode::from_u16(*status).unwrap_or(StatusCode::BAD_GATEWAY);
        return (status_code, body.clone()).into_response();
    }

    let status =
        StatusCode::from_u16(err.http_status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let body = match format {
        Format::OpenAi => json!({
            "error": {
                "message": err.to_string(),
                "type": classify(err),
                "code": classify(err),
            }
        }),
        Format::Anthropic => json!({
            "type": "error",
            "error": { "type": classify(err), "message": err.to_string() }
        }),
    };
    (status, Json(body)).into_response()
}

fn classify(e: &GatewayError) -> &'static str {
    match e {
        GatewayError::BadRequest(_) => "invalid_request_error",
        GatewayError::Unauthorized => "authentication_error",
        GatewayError::PayloadTooLarge => "request_too_large",
        GatewayError::NoRouteForModel { .. } => "upstream_error",
        GatewayError::Provider(_) => "upstream_error",
        GatewayError::Internal(_) => "internal_error",
    }
}
