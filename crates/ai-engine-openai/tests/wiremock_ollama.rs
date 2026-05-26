use ai_engine_openai::OpenAiProvider;
use ai_engine_provider::{
    openai::{ChatMessage, ChatRequest, ChatContent},
    provider::{CallCtx, Credentials, Provider},
};
use uuid::Uuid;
use wiremock::{
    matchers::{method, path},
    Match, Mock, MockServer, Request, ResponseTemplate,
};

/// Custom matcher: rejects any request that has an "authorization" header.
struct NoAuthHeader;
impl Match for NoAuthHeader {
    fn matches(&self, request: &Request) -> bool {
        !request.headers.contains_key("authorization")
    }
}

#[tokio::test]
async fn ollama_no_api_key_omits_authorization_header() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(NoAuthHeader)
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "chatcmpl-llama",
            "model": "llama3.2",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from Ollama"},
                "finish_reason": "stop"
            }]
        })))
        .mount(&upstream)
        .await;

    let provider = OpenAiProvider::new(
        "ollama-local".into(),
        upstream.uri(),  // pretend this is http://localhost:11434/v1
        30,
        false,           // http2 off — Ollama default is http/1.1
    );
    let creds = Credentials::none();
    let ctx = CallCtx { request_id: Uuid::now_v7(), deadline: None, upstream_model: "llama3.2".into() };
    let req = ChatRequest {
        model: "llama3.2".into(),
        messages: vec![ChatMessage { role: "user".into(), content: ChatContent::Text("hi".into()), extras: Default::default() }],
        stream: None, temperature: None, max_tokens: None, stream_options: None, extras: Default::default(),
    };
    let resp = provider.chat(req, &creds, &ctx).await.expect("chat ok against ollama-shaped upstream");
    assert_eq!(resp.id, "chatcmpl-llama");
    assert_eq!(resp.choices[0].message.role, "assistant");
}
