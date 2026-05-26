use std::sync::Arc;

use ai_engine_http::{build_router, AppState, ServiceInfo};
use axum::body::Body;
use axum::http::Request;
use http_body_util::BodyExt;
use tower::ServiceExt;

#[tokio::test]
async fn services_route_returns_configured_services() {
    let mut state = AppState::new();
    state.services = vec![ServiceInfo {
        id: "ollama-local".into(),
        kind: "openai".into(),
        endpoint: Some("http://localhost:11434/v1".into()),
        models: vec!["qwen2.5-coder:7b".into()],
        device: None,
        weights: None,
        local: true,
    }];
    let app = build_router(Arc::new(state));
    let resp = app
        .oneshot(
            Request::get("/cluster/services")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let arr = json.as_array().unwrap();
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0]["id"], "ollama-local");
    assert_eq!(arr[0]["kind"], "openai");
    assert_eq!(arr[0]["endpoint"], "http://localhost:11434/v1");
    assert_eq!(arr[0]["models"][0], "qwen2.5-coder:7b");
}

#[tokio::test]
async fn services_route_returns_empty_array_when_no_services() {
    let state = AppState::new();
    let app = build_router(Arc::new(state));
    let resp = app
        .oneshot(
            Request::get("/cluster/services")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(json.as_array().unwrap().is_empty());
}
