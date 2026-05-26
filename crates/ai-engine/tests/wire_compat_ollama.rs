mod common;

use serde_json::json;
use wiremock::{
    matchers::{method, path},
    Match, Mock, MockServer, Request, ResponseTemplate,
};

/// Asserts the upstream NEVER sees an `authorization` header — Ollama default config
/// won't tolerate one, so ai-engine must omit it when api_key is unset.
struct NoAuthHeader;
impl Match for NoAuthHeader {
    fn matches(&self, request: &Request) -> bool {
        !request.headers.contains_key("authorization")
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ollama_no_api_key_works_end_to_end() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(NoAuthHeader)
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-llama",
            "object": "chat.completion",
            "created": 1_700_000_000_u64,
            "model": "llama3.2",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from Ollama"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}
        })))
        .mount(&upstream)
        .await;

    // Build ai-engine config WITHOUT api_key on the openai-kind provider (Ollama scenario).
    let cfg = common::config_for("openai", &upstream.uri(), /*with_api_key=*/ false);
    let gw_base = common::spawn(&cfg).await;

    // Hit ai-engine WITHOUT sending an Authorization header. Passthrough auth records
    // raw_bearer=None, the provider has no default api_key, so the outbound call to
    // the mock upstream omits Authorization — exactly what Ollama requires.
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{gw_base}/v1/chat/completions"))
        .header("content-type", "application/json")
        // intentionally NO Authorization header
        .body(
            json!({
                "model": "llama3.2",
                "messages": [{"role": "user", "content": "hi"}]
            })
            .to_string(),
        )
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status().as_u16(),
        200,
        "Ollama-shaped request through ai-engine succeeds"
    );
    let v: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(v["choices"][0]["message"]["content"], "Hello from Ollama");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ollama_via_async_openai_sdk() {
    // Equivalent test using the async-openai SDK against ai-engine → Ollama mock.
    // The SDK always sends Authorization; ai-engine forwards it (passthrough),
    // and we don't constrain headers here.
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-llama-sdk",
            "object": "chat.completion",
            "created": 1_700_000_000_u64,
            "model": "llama3.2",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "via SDK"},
                "finish_reason": "stop"
            }]
        })))
        .mount(&upstream)
        .await;

    let cfg = common::config_for("openai", &upstream.uri(), false);
    let gw_base = common::spawn(&cfg).await;

    let client = async_openai::Client::with_config(
        async_openai::config::OpenAIConfig::new()
            .with_api_base(format!("{gw_base}/v1"))
            .with_api_key("anything"), // SDK requires non-empty
    );
    let req = async_openai::types::CreateChatCompletionRequestArgs::default()
        .model("llama3.2")
        .messages([
            async_openai::types::ChatCompletionRequestUserMessageArgs::default()
                .content("hi")
                .build()
                .unwrap()
                .into(),
        ])
        .build()
        .unwrap();
    let resp = client
        .chat()
        .create(req)
        .await
        .expect("SDK call against ai-engine → Ollama mock");
    assert_eq!(resp.choices[0].message.content.as_deref(), Some("via SDK"));
}
