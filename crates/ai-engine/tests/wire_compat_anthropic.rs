mod common;

use serde_json::json;
use wiremock::{
    matchers::{method, path},
    Mock, MockServer, ResponseTemplate,
};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn anthropic_messages_response_shape_passes_through() {
    let upstream = MockServer::start().await;
    let canned = json!({
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-sonnet-20240620",
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 2}
    });
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(canned.clone()))
        .mount(&upstream)
        .await;

    let cfg = common::config_for("anthropic", &upstream.uri(), true);
    let gw_base = common::spawn(&cfg).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{gw_base}/v1/messages"))
        .header("x-api-key", "user-token")
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .body(
            json!({
                "model": "claude-3-5-sonnet-20240620",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100
            })
            .to_string(),
        )
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    let v: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(v["id"], "msg_test");
    assert_eq!(v["content"][0]["text"], "hello");
    assert_eq!(v["usage"]["input_tokens"], 5);
}
