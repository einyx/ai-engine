mod common;

use serde_json::json;
use wiremock::{
    matchers::{method, path},
    Mock, MockServer, ResponseTemplate,
};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn openai_chat_via_async_openai_sdk() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1_700_000_000_u64,
            "model": "gpt-4o-2024-08-06",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hello via airproxy"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
        })))
        .mount(&upstream)
        .await;

    let cfg = common::config_for("openai", &upstream.uri(), true);
    let gw_base = common::spawn(&cfg).await;

    let client = async_openai::Client::with_config(
        async_openai::config::OpenAIConfig::new()
            .with_api_base(format!("{gw_base}/v1"))
            .with_api_key("user-token"),
    );
    let req = async_openai::types::CreateChatCompletionRequestArgs::default()
        .model("gpt-4o")
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
        .expect("sdk accepts response");
    assert_eq!(resp.choices.len(), 1);
    assert_eq!(
        resp.choices[0].message.content.as_deref(),
        Some("hello via airproxy")
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn openai_chat_stream_via_async_openai_sdk() {
    use futures::StreamExt;
    let upstream = MockServer::start().await;
    let sse = "data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hel\"}}]}\n\n\
data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"}}]}\n\n\
data: [DONE]\n\n";
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&upstream)
        .await;

    let cfg = common::config_for("openai", &upstream.uri(), true);
    let gw_base = common::spawn(&cfg).await;

    let client = async_openai::Client::with_config(
        async_openai::config::OpenAIConfig::new()
            .with_api_base(format!("{gw_base}/v1"))
            .with_api_key("user-token"),
    );
    let req = async_openai::types::CreateChatCompletionRequestArgs::default()
        .model("gpt-4o")
        .stream(true)
        .messages([
            async_openai::types::ChatCompletionRequestUserMessageArgs::default()
                .content("hi")
                .build()
                .unwrap()
                .into(),
        ])
        .build()
        .unwrap();
    let mut stream = client
        .chat()
        .create_stream(req)
        .await
        .expect("stream open");
    let mut chunks = 0;
    while let Some(item) = stream.next().await {
        let _ = item.expect("chunk parses");
        chunks += 1;
    }
    assert!(chunks >= 2, "expected at least 2 chunks, got {chunks}");
}
