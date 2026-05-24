//! Env-gated real-model smoke for the candle-local provider.
//!   AI_ENGINE_REAL_GGUF=/path/to/Llama-3.2-1B-Instruct-Q4_0.gguf
//! Run: cargo test -p ai-engine --test candle_smoke --features backend-candle -- --ignored --nocapture

#![cfg(feature = "backend-candle")]

use ai_engine_candle::CandleProvider;
use ai_engine_provider::openai::{ChatContent, ChatMessage, ChatRequest};
use ai_engine_provider::provider::{CallCtx, Credentials, Provider};
use uuid::Uuid;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn candle_local_real_model_chat_is_coherent() {
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
        stream: None,
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
    let resp = provider
        .chat(req, &Credentials::none(), &ctx)
        .await
        .expect("chat ok");
    let dt = t0.elapsed();

    assert_eq!(resp.choices.len(), 1, "exactly one choice expected");
    let text = match &resp.choices[0].message.content {
        ChatContent::Text(t) => t.clone(),
        ChatContent::Parts(_) => panic!("expected Text content from candle"),
    };
    eprintln!("CANDLE CHAT RESPONSE: {text:?}");

    let usage = resp.usage.as_ref().expect("usage must be populated");
    let tps = usage.completion_tokens as f64 / dt.as_secs_f64();
    eprintln!(
        "CANDLE PERF: {} completion tokens in {:.3}s = {:.3} tok/s",
        usage.completion_tokens,
        dt.as_secs_f64(),
        tps
    );

    assert!(usage.completion_tokens >= 1, "expected at least one token");
    assert!(!text.trim().is_empty(), "expected non-empty text");
}
