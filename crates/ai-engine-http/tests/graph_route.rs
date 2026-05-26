use std::sync::Arc;

use ai_engine_http::{build_router, AppState};
use axum::body::Body;
use axum::http::Request;
use http_body_util::BodyExt;
use tower::ServiceExt;

#[tokio::test]
async fn graph_route_returns_nodes_and_edges_shape() {
    // Point the scanner at a non-existent root so the response is deterministic.
    std::env::set_var("AI_ENGINE_GRAPH_ROOT", "/no/such/dir");

    let state = Arc::new(AppState::new());
    let app = build_router(state);
    let resp = app
        .oneshot(Request::get("/graph").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);

    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(json.get("nodes").unwrap().is_array());
    assert!(json.get("edges").unwrap().is_array());
}
