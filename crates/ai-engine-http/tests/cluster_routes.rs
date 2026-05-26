use std::sync::Arc;

use ai_engine_core::cluster_view::{ClusterView, NodeTopology, TopologySnapshot};
use ai_engine_http::{build_router, AppState};
use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use tower::ServiceExt;

struct FakeView;
impl ClusterView for FakeView {
    fn topology(&self) -> TopologySnapshot {
        TopologySnapshot {
            model_id: Some("m".into()),
            nodes: vec![NodeTopology {
                node_id: "a".into(),
                backend: "Cuda".into(),
                device_index: 0,
                available_memory_bytes: 1,
                compute_score: 2,
                link_mbps_to_leader: 3,
                layer_start: 0,
                layer_end: 4,
                hosts_embedding: true,
                hosts_output: true,
                previous_node: None,
                next_node: None,
            }],
        }
    }
    fn total_tokens(&self) -> u64 {
        0
    }
}

#[tokio::test]
async fn topology_returns_nodes_when_clustered() {
    let mut state = AppState::new();
    state.cluster = Some(Arc::new(FakeView));
    let app = build_router(Arc::new(state));

    let resp = app
        .oneshot(Request::get("/cluster/topology").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["model_id"], "m");
    assert_eq!(v["nodes"][0]["node_id"], "a");
}

#[tokio::test]
async fn topology_empty_when_gateway_only() {
    let app = build_router(Arc::new(AppState::new())); // cluster: None
    let resp = app
        .oneshot(Request::get("/cluster/topology").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["model_id"], serde_json::Value::Null);
    assert!(v["nodes"].as_array().unwrap().is_empty());
}
