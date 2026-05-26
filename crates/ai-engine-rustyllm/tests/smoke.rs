//! Env-gated real-model smoke for the rustyllm provider.
//!
//! Set RUSTYLLM_MODEL_DIR to a local Llama-family HF safetensors model
//! directory (config.json + tokenizer.json + *.safetensors), then:
//!
//!   RUSTYLLM_MODEL_DIR=/path/to/model \
//!     cargo test -p ai-engine-rustyllm -- --nocapture
//!
//! Skips silently when the env var is unset.

use ai_engine_provider::openai;
use ai_engine_provider::provider::{CallCtx, Credentials, Provider};
use ai_engine_rustyllm::RustyllmProvider;
use futures::StreamExt;

fn req(prompt: &str, max_tokens: u32) -> openai::ChatRequest {
    openai::ChatRequest {
        model: "rustyllm-test".into(),
        messages: vec![openai::ChatMessage {
            role: "user".into(),
            content: openai::ChatContent::Text(prompt.into()),
            extras: Default::default(),
        }],
        stream: None,
        temperature: Some(0.0),
        max_tokens: Some(max_tokens),
        stream_options: None,
        extras: Default::default(),
    }
}

fn ctx() -> CallCtx {
    CallCtx {
        request_id: uuid::Uuid::now_v7(),
        deadline: None,
        upstream_model: "rustyllm-test".into(),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn smoke_chat_and_stream() {
    let Ok(model_dir) = std::env::var("RUSTYLLM_MODEL_DIR") else {
        eprintln!("RUSTYLLM_MODEL_DIR unset — skipping rustyllm smoke");
        return;
    };
    let device = std::env::var("RUSTYLLM_DEVICE").unwrap_or_else(|_| "cpu".into());

    let provider = RustyllmProvider::new("rustyllm-test", &model_dir, &device, 512)
        .expect("load model");

    // Non-streaming
    let resp = provider
        .chat(req("The capital of France is", 4), &Credentials::none(), &ctx())
        .await
        .expect("chat");
    let text = match &resp.choices[0].message.content {
        openai::ChatContent::Text(s) => s.clone(),
        _ => panic!("expected text"),
    };
    eprintln!("[chat] -> {text:?}");
    assert!(!text.is_empty(), "completion should be non-empty");
    let usage = resp.usage.expect("usage present");
    assert!(usage.prompt_tokens > 0 && usage.completion_tokens > 0);

    // Streaming: collect deltas
    let mut stream = provider
        .chat_stream(req("Once upon a time", 4), &Credentials::none(), &ctx())
        .await
        .expect("chat_stream");
    let mut pieces = 0usize;
    let mut acc = String::new();
    while let Some(ev) = stream.next().await {
        let ev = ev.expect("stream event");
        if let Some(c) = ev.raw["choices"][0]["delta"]["content"].as_str() {
            acc.push_str(c);
            pieces += 1;
        }
    }
    eprintln!("[stream] {pieces} content chunks -> {acc:?}");
    assert!(pieces >= 1, "expected at least one streamed content chunk");
}
