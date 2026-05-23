use ai_engine_openai::OpenAiProvider;
use ai_engine_provider::{
    openai::{ChatMessage, ChatRequest, ChatContent, EmbeddingsInput, EmbeddingsRequest},
    provider::{CallCtx, Credentials, Provider},
};
use uuid::Uuid;
use wiremock::{
    matchers::{header, method, path},
    Mock, MockServer, ResponseTemplate,
};

fn canned_chat() -> serde_json::Value {
    serde_json::json!({
        "id": "chatcmpl-abc",
        "model": "gpt-4o-2024-08-06",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "hello"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
    })
}

fn ctx() -> CallCtx {
    CallCtx { request_id: Uuid::now_v7(), deadline: None, upstream_model: "gpt-4o-2024-08-06".into() }
}

fn req() -> ChatRequest {
    ChatRequest {
        model: "ignored-overridden-by-ctx".into(),
        messages: vec![ChatMessage { role: "user".into(), content: ChatContent::Text("hi".into()), extras: Default::default() }],
        stream: None,
        temperature: None,
        max_tokens: None,
        stream_options: None,
        extras: Default::default(),
    }
}

#[tokio::test]
async fn chat_sends_bearer_and_returns_response() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(canned_chat()))
        .mount(&upstream)
        .await;

    let provider = OpenAiProvider::new(
        "openai-test".into(),
        upstream.uri().to_string(),
        30,
        true,
    );
    let creds = Credentials { api_key: Some("test-key".into()), raw_bearer: None, extra_headers: vec![] };
    let resp = provider.chat(req(), &creds, &ctx()).await.expect("chat ok");
    assert_eq!(resp.id, "chatcmpl-abc");
    assert_eq!(resp.choices.len(), 1);
}

#[tokio::test]
async fn chat_uses_upstream_model_from_ctx_not_request_model() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(wiremock::matchers::body_partial_json(serde_json::json!({"model": "gpt-4o-2024-08-06"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(canned_chat()))
        .mount(&upstream)
        .await;

    let provider = OpenAiProvider::new("p".into(), upstream.uri(), 30, true);
    let creds = Credentials { api_key: Some("k".into()), raw_bearer: None, extra_headers: vec![] };
    provider.chat(req(), &creds, &ctx()).await.expect("ok");
}

#[tokio::test]
async fn chat_passes_raw_bearer_in_passthrough_mode() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer caller-token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(canned_chat()))
        .mount(&upstream)
        .await;
    let provider = OpenAiProvider::new("p".into(), upstream.uri(), 30, true);
    let creds = Credentials { api_key: Some("ignored".into()), raw_bearer: Some("Bearer caller-token".into()), extra_headers: vec![] };
    provider.chat(req(), &creds, &ctx()).await.expect("ok");
}

#[tokio::test]
async fn chat_propagates_upstream_status_with_body() {
    use ai_engine_provider::error::ProviderError;
    let upstream = MockServer::start().await;
    Mock::given(method("POST")).and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(429).set_body_string("{\"error\":\"rate_limit\"}"))
        .mount(&upstream).await;

    let provider = OpenAiProvider::new("p".into(), upstream.uri(), 30, true);
    let creds = Credentials { api_key: Some("k".into()), raw_bearer: None, extra_headers: vec![] };
    let err = provider.chat(req(), &creds, &ctx()).await.unwrap_err();
    match err {
        ProviderError::Status { status, body } => {
            assert_eq!(status, 429);
            assert!(body.windows(10).any(|w| w == b"rate_limit"));
        }
        other => panic!("expected Status, got {:?}", other),
    }
}

#[tokio::test]
async fn embeddings_works() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST")).and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "data": [{"index": 0, "embedding": [0.1, 0.2], "object": "embedding"}],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 3, "completion_tokens": 0, "total_tokens": 3}
        })))
        .mount(&upstream).await;

    let provider = OpenAiProvider::new("p".into(), upstream.uri(), 30, true);
    let creds = Credentials { api_key: Some("k".into()), raw_bearer: None, extra_headers: vec![] };
    let ctx = CallCtx { request_id: Uuid::now_v7(), deadline: None, upstream_model: "text-embedding-3-small".into() };
    let req = EmbeddingsRequest { model: "x".into(), input: EmbeddingsInput::Single("hi".into()), extras: Default::default() };
    let resp = provider.embeddings(req, &creds, &ctx).await.expect("ok");
    assert_eq!(resp.data.len(), 1);
}

#[tokio::test]
async fn chat_stream_parses_sse_chunks() {
    use futures::StreamExt;
    let upstream = MockServer::start().await;
    let sse = "data: {\"id\":\"1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hel\"}}]}\n\n\
data: {\"id\":\"1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"}}]}\n\n\
data: [DONE]\n\n";
    Mock::given(method("POST")).and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200)
            .set_body_string(sse)
            .insert_header("content-type", "text/event-stream"))
        .mount(&upstream).await;

    let provider = OpenAiProvider::new("p".into(), upstream.uri(), 30, true);
    let creds = Credentials { api_key: Some("k".into()), raw_bearer: None, extra_headers: vec![] };
    let mut r = req();
    r.stream = Some(true);
    let mut stream = provider.chat_stream(r, &creds, &ctx()).await.expect("ok");
    let mut count = 0;
    while let Some(ev) = stream.next().await {
        ev.expect("event");
        count += 1;
    }
    assert_eq!(count, 2, "should yield 2 events (DONE terminates)");
}
