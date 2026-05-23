use ai_engine_anthropic::AnthropicProvider;
use ai_engine_provider::{
    anthropic::{Message, MessageContent, MessagesRequest},
    provider::{CallCtx, Credentials, Provider},
};
use uuid::Uuid;
use wiremock::{matchers::{header, method, path}, Mock, MockServer, ResponseTemplate};

fn canned_response() -> serde_json::Value {
    serde_json::json!({
        "id": "msg_01",
        "model": "claude-3-5-sonnet-20240620",
        "role": "assistant",
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 2}
    })
}

fn ctx(model: &str) -> CallCtx {
    CallCtx { request_id: Uuid::now_v7(), deadline: None, upstream_model: model.into() }
}

fn req() -> MessagesRequest {
    MessagesRequest {
        model: "ignored".into(),
        messages: vec![Message { role: "user".into(), content: MessageContent::Text("hi".into()), extras: Default::default() }],
        max_tokens: 1024,
        system: None,
        stream: None,
        extras: Default::default(),
    }
}

#[tokio::test]
async fn messages_sends_x_api_key_and_version_headers() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-api-key", "sk-ant-test"))
        .and(header("anthropic-version", "2023-06-01"))
        .respond_with(ResponseTemplate::new(200).set_body_json(canned_response()))
        .mount(&upstream).await;

    let provider = AnthropicProvider::new("anth".into(), upstream.uri(), 30);
    let creds = Credentials { api_key: Some("sk-ant-test".into()), raw_bearer: None, extra_headers: vec![] };
    let resp = provider.messages(req(), &creds, &ctx("claude-3-5-sonnet-20240620")).await.expect("ok");
    assert_eq!(resp.id, "msg_01");
    assert_eq!(resp.usage.input_tokens, 5);
}

#[tokio::test]
async fn messages_uses_upstream_model_override() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST")).and(path("/v1/messages"))
        .and(wiremock::matchers::body_partial_json(serde_json::json!({"model": "claude-3-5-sonnet-20240620"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(canned_response()))
        .mount(&upstream).await;
    let provider = AnthropicProvider::new("a".into(), upstream.uri(), 30);
    let creds = Credentials { api_key: Some("k".into()), raw_bearer: None, extra_headers: vec![] };
    provider.messages(req(), &creds, &ctx("claude-3-5-sonnet-20240620")).await.expect("ok");
}

#[tokio::test]
async fn messages_propagates_upstream_status() {
    use ai_engine_provider::error::ProviderError;
    let upstream = MockServer::start().await;
    Mock::given(method("POST")).and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(400).set_body_string("{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\"}}"))
        .mount(&upstream).await;
    let provider = AnthropicProvider::new("a".into(), upstream.uri(), 30);
    let creds = Credentials { api_key: Some("k".into()), raw_bearer: None, extra_headers: vec![] };
    let err = provider.messages(req(), &creds, &ctx("claude-3-5-sonnet-20240620")).await.unwrap_err();
    matches!(err, ProviderError::Status { status: 400, .. });
}

#[tokio::test]
async fn messages_stream_parses_sse_events() {
    use futures::StreamExt;
    let upstream = MockServer::start().await;
    let sse = "event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\"}}\n\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\"hi\"}}\n\n\
event: message_stop\n\
data: {\"type\":\"message_stop\"}\n\n";
    Mock::given(method("POST")).and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200)
            .set_body_string(sse)
            .insert_header("content-type", "text/event-stream"))
        .mount(&upstream).await;

    let provider = AnthropicProvider::new("a".into(), upstream.uri(), 30);
    let creds = Credentials { api_key: Some("k".into()), raw_bearer: None, extra_headers: vec![] };
    let mut r = req();
    r.stream = Some(true);
    let mut stream = provider.messages_stream(r, &creds, &ctx("claude-3-5-sonnet-20240620")).await.expect("ok");
    let mut count = 0;
    while let Some(ev) = stream.next().await {
        ev.expect("event");
        count += 1;
    }
    assert_eq!(count, 3, "should yield 3 events");
}
