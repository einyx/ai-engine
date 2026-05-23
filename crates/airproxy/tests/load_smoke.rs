//! Streaming load smoke. Marked `#[ignore]` so CI runs only on demand.
//!
//! Run with: `cargo test --release -p airproxy --test load_smoke -- --ignored --nocapture`

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use futures::StreamExt;
use serde_json::json;
use wiremock::{matchers::{method, path}, Mock, MockServer, ResponseTemplate};

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "load test; run with `cargo test --release ... -- --ignored`"]
async fn load_smoke_500_concurrent_streams() {
    // Upstream returns a fixed SSE response — 4 events plus [DONE].
    let sse = "data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"a\"}}]}\n\n\
data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"b\"}}]}\n\n\
data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"c\"}}]}\n\n\
data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"d\"}}]}\n\n\
data: [DONE]\n\n";

    let upstream = MockServer::start().await;
    Mock::given(method("POST")).and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200)
            .set_body_string(sse)
            .insert_header("content-type", "text/event-stream"))
        .mount(&upstream).await;

    let cfg = common::config_for("openai", &upstream.uri(), true);
    let gw_base = common::spawn(&cfg).await;

    const CONCURRENCY: usize = 500;
    let success = Arc::new(AtomicUsize::new(0));
    let total_events = Arc::new(AtomicUsize::new(0));
    let start = Instant::now();

    let mut handles = Vec::with_capacity(CONCURRENCY);
    for i in 0..CONCURRENCY {
        let base = gw_base.clone();
        let success = success.clone();
        let total = total_events.clone();
        handles.push(tokio::spawn(async move {
            let body = json!({
                "model": "gpt-4o",
                "stream": true,
                "messages": [{"role": "user", "content": format!("req-{i}")}]
            }).to_string();
            let resp = reqwest::Client::new()
                .post(format!("{base}/v1/chat/completions"))
                .header("authorization", "Bearer x")
                .header("content-type", "application/json")
                .body(body)
                .send().await;
            let Ok(resp) = resp else { return; };
            if !resp.status().is_success() { return; }
            let mut stream = resp.bytes_stream();
            let mut count = 0;
            while let Some(chunk) = stream.next().await {
                if let Ok(bytes) = chunk {
                    // Count `data:` frames as a rough event count.
                    count += bytes.windows(5).filter(|w| w == b"data:").count();
                } else {
                    return;
                }
            }
            success.fetch_add(1, Ordering::Relaxed);
            total.fetch_add(count, Ordering::Relaxed);
        }));
    }
    for h in handles { let _ = h.await; }
    let elapsed = start.elapsed();

    let ok = success.load(Ordering::Relaxed);
    let events = total_events.load(Ordering::Relaxed);
    eprintln!("[load] {ok}/{CONCURRENCY} streams completed in {elapsed:?}; total events seen: {events}");

    // Assertions: at least 99% should complete successfully under load,
    // and each successful stream should see at least 4 data frames
    // (4 content chunks; the [DONE] sentinel airproxy appends counts too,
    // making the floor 4 to be conservative).
    let min_ok = (CONCURRENCY * 99) / 100;
    assert!(ok >= min_ok, "expected ≥ {min_ok}/{CONCURRENCY} streams to complete, got {ok}");
    let avg_events = if ok > 0 { events / ok } else { 0 };
    assert!(avg_events >= 4, "expected ≥ 4 events per stream on average, got {avg_events}");
}
