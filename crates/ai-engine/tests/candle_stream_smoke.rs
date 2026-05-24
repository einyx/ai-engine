//! Env-gated streaming smoke for the candle-local provider.
//!   AI_ENGINE_REAL_GGUF=/path/to/model.gguf
//! Run: cargo test -p ai-engine --test candle_stream_smoke --features backend-candle -- --ignored --nocapture

#![cfg(feature = "backend-candle")]

use ai_engine_candle::CandleProvider;
use ai_engine_provider::openai::{self, ChatContent, ChatMessage, ChatRequest};
use ai_engine_provider::provider::{CallCtx, Credentials, Provider};
use futures::StreamExt;
use uuid::Uuid;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn candle_local_chat_stream_decodes_coherently() {
    let gguf = match std::env::var("AI_ENGINE_REAL_GGUF") {
        Ok(p) => std::path::PathBuf::from(p),
        Err(_) => {
            eprintln!("SKIP: set AI_ENGINE_REAL_GGUF");
            return;
        }
    };
    if !gguf.exists() {
        eprintln!("SKIP: missing {}", gguf.display());
        return;
    }

    let provider = CandleProvider::new("llama-gpu", &gguf, "auto", 1)
        .expect("build CandleProvider");

    let req = ChatRequest {
        model: "llama-gpu".into(),
        messages: vec![ChatMessage {
            role: "user".into(),
            content: ChatContent::Text("Hello, who are you?".into()),
            extras: Default::default(),
        }],
        stream: Some(true),
        temperature: Some(0.0),
        max_tokens: Some(20),
        stream_options: None,
        extras: Default::default(),
    };
    let ctx = CallCtx {
        request_id: Uuid::now_v7(),
        deadline: None,
        upstream_model: "llama-gpu".into(),
    };

    let t0 = std::time::Instant::now();
    let mut stream = provider
        .chat_stream(req, &Credentials::none(), &ctx)
        .await
        .expect("stream ok");

    let mut full = String::new();
    let mut delta_count = 0usize;
    let mut saw_finish = false;
    let mut total_chunks = 0usize;

    while let Some(item) = stream.next().await {
        let chunk: openai::ChatStreamEvent = item.expect("no stream error");
        total_chunks += 1;

        assert_eq!(
            chunk.raw["object"].as_str(),
            Some("chat.completion.chunk"),
            "chunk object != chat.completion.chunk: {}",
            chunk.raw
        );

        if let Some(piece) = chunk.raw["choices"][0]["delta"]["content"].as_str() {
            if !piece.is_empty() {
                full.push_str(piece);
                delta_count += 1;
            }
        }
        if chunk.raw["choices"][0]["finish_reason"].as_str() == Some("stop") {
            saw_finish = true;
        }
    }
    let dt = t0.elapsed();

    eprintln!(
        "STREAMED TEXT: {full:?}  (deltas={delta_count}, chunks={total_chunks}, finish={saw_finish}, {:.3}s)",
        dt.as_secs_f64()
    );

    assert!(delta_count >= 1, "expected at least one delta chunk");
    assert!(
        !full.trim().is_empty(),
        "concatenated deltas must be non-empty"
    );
    assert!(saw_finish, "expected a finish_reason=stop terminal chunk");
}
