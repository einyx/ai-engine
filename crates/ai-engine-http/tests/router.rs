use ai_engine_core::ctx::{GatewayResponse, RequestCtx, ResponseSlot};
use ai_engine_core::error::GatewayError;
use ai_engine_core::pipeline::Pipeline;
use ai_engine_core::stage::{Stage, StageError, StageOutcome};
use ai_engine_http::{build_router, AppState};
use arc_swap::ArcSwap;
use axum::body::to_bytes;
use axum::http::{HeaderMap, Request, StatusCode};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tower::ServiceExt;

struct AlwaysOk;

#[async_trait::async_trait]
impl Stage for AlwaysOk {
    fn name(&self) -> &'static str {
        "always_ok"
    }
    async fn process(&self, ctx: &mut RequestCtx) -> Result<StageOutcome, StageError> {
        ctx.response = ResponseSlot::Full(GatewayResponse {
            status: 200,
            headers: HeaderMap::new(),
            body: br#"{"id":"chatcmpl-x","model":"gpt-4o","choices":[],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}"#.to_vec().into(),
        });
        Ok(StageOutcome::Continue)
    }
}

struct AlwaysUnauthorized;

#[async_trait::async_trait]
impl Stage for AlwaysUnauthorized {
    fn name(&self) -> &'static str {
        "deny"
    }
    async fn process(&self, _ctx: &mut RequestCtx) -> Result<StageOutcome, StageError> {
        Err(StageError {
            stage: self.name(),
            error: GatewayError::Unauthorized,
        })
    }
}

fn make_state(route: &'static str, stage: Arc<dyn Stage>) -> Arc<AppState> {
    let mut s = AppState::new();
    let pipeline = Pipeline::new(vec![stage]);
    s.pipelines.insert(route, ArcSwap::new(Arc::new(pipeline)));
    s.ready.store(true, Ordering::Relaxed);
    Arc::new(s)
}

#[tokio::test]
async fn healthz_returns_200() {
    let state = Arc::new(AppState::new());
    let app = build_router(state);
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/healthz")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn readyz_503_when_not_ready_200_when_ready() {
    let state = Arc::new(AppState::new());
    let app = build_router(state.clone());
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/readyz")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    state.ready.store(true, Ordering::Relaxed);
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/readyz")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn chat_returns_200_with_json_body_from_pipeline() {
    let state = make_state("/v1/chat/completions", Arc::new(AlwaysOk));
    let app = build_router(state);
    let req = Request::builder()
        .uri("/v1/chat/completions")
        .method("POST")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(
            r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}"#,
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn chat_invalid_json_returns_400_openai_envelope() {
    let state = make_state("/v1/chat/completions", Arc::new(AlwaysOk));
    let app = build_router(state);
    let req = Request::builder()
        .uri("/v1/chat/completions")
        .method("POST")
        .header("content-type", "application/json")
        .body(axum::body::Body::from("not json"))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body_bytes = to_bytes(resp.into_body(), 4096).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
    assert!(v.get("error").is_some());
    assert_eq!(v["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn chat_unauthorized_returns_401_openai_envelope() {
    let state = make_state("/v1/chat/completions", Arc::new(AlwaysUnauthorized));
    let app = build_router(state);
    let req = Request::builder()
        .uri("/v1/chat/completions")
        .method("POST")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(
            r#"{"model":"gpt-4o","messages":[]}"#,
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    let body_bytes = to_bytes(resp.into_body(), 4096).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(v["error"]["type"], "authentication_error");
}

#[tokio::test]
async fn messages_unauthorized_returns_anthropic_envelope() {
    let state = make_state("/v1/messages", Arc::new(AlwaysUnauthorized));
    let app = build_router(state);
    let req = Request::builder()
        .uri("/v1/messages")
        .method("POST")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(
            r#"{"model":"claude-3-5","messages":[],"max_tokens":100}"#,
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    let body_bytes = to_bytes(resp.into_body(), 4096).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(v["type"], "error");
    assert_eq!(v["error"]["type"], "authentication_error");
}

#[tokio::test]
async fn models_returns_route_table_models() {
    let mut s = AppState::new();
    s.openai_models = vec!["gpt-4o".into(), "claude-3-5".into()];
    s.ready.store(true, Ordering::Relaxed);
    let app = build_router(Arc::new(s));
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body_bytes = to_bytes(resp.into_body(), 4096).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(v["object"], "list");
    assert_eq!(v["data"].as_array().unwrap().len(), 2);
}
