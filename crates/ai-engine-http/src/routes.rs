use std::sync::Arc;

use ai_engine_core::ctx::{RequestBody, RequestCtx, ResponseSlot};
use ai_engine_provider::{anthropic, openai};
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response, Sse};
use serde_json::json;

use crate::error::{envelope, Format};
use crate::sse::{encode_anthropic, encode_openai};
use crate::AppState;

pub async fn healthz() -> &'static str {
    "ok"
}

pub async fn readyz(State(state): State<Arc<AppState>>) -> Response {
    if state.ready.load(std::sync::atomic::Ordering::Relaxed) {
        (StatusCode::OK, "ready").into_response()
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "starting").into_response()
    }
}

pub async fn models(State(state): State<Arc<AppState>>) -> Response {
    let data: Vec<_> = state
        .openai_models
        .iter()
        .map(|m| {
            json!({
                "id": m,
                "object": "model",
                "owned_by": "ai-engine",
            })
        })
        .collect();
    axum::Json(json!({ "object": "list", "data": data })).into_response()
}

pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let req: openai::ChatRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            return envelope(
                Format::OpenAi,
                &ai_engine_core::error::GatewayError::BadRequest(format!("invalid JSON: {e}")),
            )
        }
    };
    let route = "/v1/chat/completions";
    let raw_len = body.len();
    handle(
        state,
        route,
        headers,
        raw_len,
        RequestBody::OpenAiChat(req),
        Format::OpenAi,
    )
    .await
}

pub async fn messages(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let req: anthropic::MessagesRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            return envelope(
                Format::Anthropic,
                &ai_engine_core::error::GatewayError::BadRequest(format!("invalid JSON: {e}")),
            )
        }
    };
    let route = "/v1/messages";
    let raw_len = body.len();
    handle(
        state,
        route,
        headers,
        raw_len,
        RequestBody::AnthropicMessages(req),
        Format::Anthropic,
    )
    .await
}

pub async fn embeddings(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let req: openai::EmbeddingsRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            return envelope(
                Format::OpenAi,
                &ai_engine_core::error::GatewayError::BadRequest(format!("invalid JSON: {e}")),
            )
        }
    };
    let route = "/v1/embeddings";
    let raw_len = body.len();
    handle(
        state,
        route,
        headers,
        raw_len,
        RequestBody::OpenAiEmbeddings(req),
        Format::OpenAi,
    )
    .await
}

async fn handle(
    state: Arc<AppState>,
    route: &'static str,
    headers: HeaderMap,
    raw_len: usize,
    body: RequestBody,
    format: Format,
) -> Response {
    let Some(pipeline_handle) = state.pipelines.get(route) else {
        return envelope(
            format,
            &ai_engine_core::error::GatewayError::Internal(anyhow::anyhow!(
                "no pipeline configured for route {route}"
            )),
        );
    };
    let pipeline = pipeline_handle.load_full();

    let mut ctx = RequestCtx::new(route, headers, raw_len, body);
    ctx.usage_slot = Some(Arc::new(std::sync::Mutex::new(None)));

    pipeline.execute(&mut ctx).await;

    if let Some(err) = ctx.error.as_ref() {
        return envelope(format, err);
    }

    match std::mem::replace(&mut ctx.response, ResponseSlot::Pending) {
        ResponseSlot::Full(r) => {
            let status = if r.status == 0 {
                StatusCode::OK
            } else {
                StatusCode::from_u16(r.status).unwrap_or(StatusCode::OK)
            };
            let mut response = (status, r.body).into_response();
            for (k, v) in r.headers.iter() {
                response.headers_mut().insert(k, v.clone());
            }
            response
        }
        ResponseSlot::Stream(inner) => match format {
            Format::OpenAi => Sse::new(encode_openai(inner)).into_response(),
            Format::Anthropic => Sse::new(encode_anthropic(inner)).into_response(),
        },
        ResponseSlot::Pending => envelope(
            format,
            &ai_engine_core::error::GatewayError::Internal(anyhow::anyhow!(
                "pipeline produced no response and no error"
            )),
        ),
    }
}
