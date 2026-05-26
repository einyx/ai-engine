use std::sync::Arc;

use ai_engine_core::ctx::{RequestBody, RequestCtx, ResponseSlot};
use ai_engine_provider::{anthropic, openai};
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response, Sse};
use serde_json::json;

use crate::error::{envelope, Format};
use crate::sse::{encode_anthropic, encode_openai};
use crate::AppState;

pub async fn healthz() -> &'static str {
    "ok"
}

pub async fn readyz(State(state): State<Arc<AppState>>) -> Response {
    if state.ready.load(std::sync::atomic::Ordering::Relaxed) {
        (StatusCode::OK, "ready").into_response()
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "starting").into_response()
    }
}

pub async fn models(State(state): State<Arc<AppState>>) -> Response {
    let data: Vec<_> = state
        .openai_models
        .iter()
        .map(|m| {
            json!({
                "id": m,
                "object": "model",
                "owned_by": "ai-engine",
            })
        })
        .collect();
    axum::Json(json!({ "object": "list", "data": data })).into_response()
}

pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let req: openai::ChatRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            return envelope(
                Format::OpenAi,
                &ai_engine_core::error::GatewayError::BadRequest(format!("invalid JSON: {e}")),
            )
        }
    };
    let route = "/v1/chat/completions";
    let raw_len = body.len();
    handle(
        state,
        route,
        headers,
        raw_len,
        RequestBody::OpenAiChat(req),
        Format::OpenAi,
    )
    .await
}

pub async fn messages(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let req: anthropic::MessagesRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            return envelope(
                Format::Anthropic,
                &ai_engine_core::error::GatewayError::BadRequest(format!("invalid JSON: {e}")),
            )
        }
    };
    let route = "/v1/messages";
    let raw_len = body.len();
    handle(
        state,
        route,
        headers,
        raw_len,
        RequestBody::AnthropicMessages(req),
        Format::Anthropic,
    )
    .await
}

pub async fn embeddings(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let req: openai::EmbeddingsRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            return envelope(
                Format::OpenAi,
                &ai_engine_core::error::GatewayError::BadRequest(format!("invalid JSON: {e}")),
            )
        }
    };
    let route = "/v1/embeddings";
    let raw_len = body.len();
    handle(
        state,
        route,
        headers,
        raw_len,
        RequestBody::OpenAiEmbeddings(req),
        Format::OpenAi,
    )
    .await
}

async fn handle(
    state: Arc<AppState>,
    route: &'static str,
    headers: HeaderMap,
    raw_len: usize,
    body: RequestBody,
    format: Format,
) -> Response {
    let Some(pipeline_handle) = state.pipelines.get(route) else {
        return envelope(
            format,
            &ai_engine_core::error::GatewayError::Internal(anyhow::anyhow!(
                "no pipeline configured for route {route}"
            )),
        );
    };
    let pipeline = pipeline_handle.load_full();

    let mut ctx = RequestCtx::new(route, headers, raw_len, body);
    ctx.usage_slot = Some(Arc::new(std::sync::Mutex::new(None)));

    pipeline.execute(&mut ctx).await;

    if let Some(err) = ctx.error.as_ref() {
        return envelope(format, err);
    }

    match std::mem::replace(&mut ctx.response, ResponseSlot::Pending) {
        ResponseSlot::Full(r) => {
            let status = if r.status == 0 {
                StatusCode::OK
            } else {
                StatusCode::from_u16(r.status).unwrap_or(StatusCode::OK)
            };
            let mut response = (status, r.body).into_response();
            for (k, v) in r.headers.iter() {
                response.headers_mut().insert(k, v.clone());
            }
            response
        }
        ResponseSlot::Stream(inner) => match format {
            Format::OpenAi => Sse::new(encode_openai(inner)).into_response(),
            Format::Anthropic => Sse::new(encode_anthropic(inner)).into_response(),
        },
        ResponseSlot::Pending => envelope(
            format,
            &ai_engine_core::error::GatewayError::Internal(anyhow::anyhow!(
                "pipeline produced no response and no error"
            )),
        ),
    }
}

/// `GET /cluster/services` — configured providers and the models routed to each.
pub async fn cluster_services(State(state): State<Arc<AppState>>) -> Response {
    axum::Json(state.services.clone()).into_response()
}

/// `GET /cluster/topology` — node assignments + capabilities. Returns an empty
/// snapshot (200) in gateway-only mode so the UI renders a clean empty state.
pub async fn cluster_topology(State(state): State<Arc<AppState>>) -> Response {
    let snap = match &state.cluster {
        Some(v) => v.topology(),
        None => ai_engine_core::cluster_view::TopologySnapshot::default(),
    };
    axum::Json(snap).into_response()
}

/// `GET /cluster/metrics` — SSE stream emitting cluster-wide and per-node
/// tokens/sec once per second, derived from token-counter deltas.
pub async fn cluster_metrics(State(state): State<Arc<AppState>>) -> Response {
    use std::convert::Infallible;

    let Some(view) = state.cluster.clone() else {
        let s = futures::stream::empty::<Result<axum::response::sse::Event, Infallible>>();
        return Sse::new(s).into_response();
    };

    let mut prev = view.total_tokens();
    let mut prev_t = std::time::Instant::now();
    let stream = async_stream::stream! {
        let mut ticker = tokio::time::interval(std::time::Duration::from_secs(1));
        loop {
            ticker.tick().await;
            let now = view.total_tokens();
            let t = std::time::Instant::now();
            let dt = t.duration_since(prev_t).as_secs_f64().max(1e-6);
            let tps = (now.saturating_sub(prev)) as f64 / dt;
            prev = now;
            prev_t = t;

            let node_ids: Vec<String> =
                view.topology().nodes.into_iter().map(|n| n.node_id).collect();
            let per_node: serde_json::Map<String, serde_json::Value> = node_ids
                .into_iter()
                .map(|id| (id, serde_json::json!(tps)))
                .collect();
            let payload = serde_json::json!({
                "total_tps": tps,
                "per_node": per_node,
            });
            yield Ok::<_, Infallible>(
                axum::response::sse::Event::default().data(payload.to_string()),
            );
        }
    };
    Sse::new(stream).into_response()
}

/// `GET /gateway/metrics` — SSE stream emitting per-provider and total
/// tokens/sec once per second, derived from per-provider completion-token
/// counter deltas. Available in both gateway and leader modes.
pub async fn gateway_metrics(State(state): State<Arc<AppState>>) -> Response {
    use std::collections::{HashMap, VecDeque};
    use std::convert::Infallible;
    use std::time::{Duration, Instant};

    // Report a sliding-window rate (tokens over the last WINDOW seconds) rather
    // than a raw 1-second delta. Non-streaming requests credit all their tokens
    // at completion, so bursty traffic makes the instantaneous rate flicker
    // 0 <-> peak; the window smooths it into a steady, friendly number that
    // eases back to 0 over WINDOW seconds when traffic stops.
    const WINDOW: Duration = Duration::from_secs(5);
    let metrics = state.gateway_metrics.clone();
    let health = state.health.clone();
    let resources = state.resources.clone();
    // Providers running on this host are sampled live; remote ones use the
    // snapshot captured at discovery.
    let local_ids: std::collections::HashSet<String> =
        state.services.iter().filter(|s| s.local).map(|s| s.id.clone()).collect();
    let stream = async_stream::stream! {
        // History of output-token snapshots for the sliding-window tok/s.
        let mut samples: VecDeque<(Instant, HashMap<String, u64>)> = VecDeque::new();
        let mut ticker = tokio::time::interval(Duration::from_secs(1));
        loop {
            ticker.tick().await;
            let now_t = Instant::now();
            let snap = metrics.snapshot();
            let tokens_now: HashMap<String, u64> =
                snap.iter().map(|(id, s)| (id.clone(), s.out_tokens)).collect();
            samples.push_back((now_t, tokens_now));
            while samples.len() > 1 && now_t.duration_since(samples[1].0) >= WINDOW {
                samples.pop_front();
            }
            let (base_t, base) = samples.front().unwrap();
            let dt = now_t.duration_since(*base_t).as_secs_f64().max(1.0);
            let health_snap = health.snapshot();
            // Live sample of this host (shared by all local providers) once per tick.
            let local_res = ai_engine_core::resources::sample();

            let mut per_provider = serde_json::Map::new();      // live tok/s
            let mut per_provider_total = serde_json::Map::new(); // cumulative tokens
            let mut stats = serde_json::Map::new();              // rich per-provider stats
            let mut total_tps = 0.0_f64;
            let mut total_tokens = 0u64;
            for (id, s) in &snap {
                let was = base.get(id).copied().unwrap_or(0);
                let tps = s.out_tokens.saturating_sub(was) as f64 / dt;
                total_tps += tps;
                total_tokens += s.out_tokens;
                per_provider.insert(id.clone(), serde_json::json!(tps));
                per_provider_total.insert(id.clone(), serde_json::json!(s.out_tokens));
                let avg_latency = if s.requests > 0 {
                    s.latency_ms_sum as f64 / s.requests as f64
                } else {
                    0.0
                };
                let h = health_snap.get(id);
                let res = if local_ids.contains(id) {
                    Some(&local_res)
                } else {
                    resources.get(id)
                };
                stats.insert(
                    id.clone(),
                    serde_json::json!({
                        "tps": tps,
                        "tokens": s.out_tokens,
                        "requests": s.requests,
                        "errors": s.errors,
                        "avg_latency_ms": avg_latency,
                        "health": h.map(|h| serde_json::json!({
                            "up": h.up, "latency_ms": h.latency_ms, "checked": h.checked,
                        })).unwrap_or(serde_json::Value::Null),
                        "resources": res.filter(|r| r.is_some()).map(|r| serde_json::json!({
                            "cpu_count": r.cpu_count, "load1": r.load1,
                            "mem_total_mb": r.mem_total_mb, "mem_avail_mb": r.mem_avail_mb,
                            "disk_avail_gb": r.disk_avail_gb,
                        })).unwrap_or(serde_json::Value::Null),
                    }),
                );
            }
            let payload = serde_json::json!({
                "total_tps": total_tps,
                "total_tokens": total_tokens,
                "per_provider": per_provider,
                "per_provider_total": per_provider_total,
                "stats": stats,
                "gpu": ai_engine_core::gpu::sample_cached(),
            });
            yield Ok::<_, Infallible>(
                axum::response::sse::Event::default().data(payload.to_string()),
            );
        }
    };
    Sse::new(stream).into_response()
}

/// `GET /graph` — heterogeneous system knowledge graph (sessions, memories,
/// commands, cluster nodes). Scans local Claude Code data; returns an empty
/// graph when no data dir is found.
pub async fn graph(State(state): State<Arc<AppState>>) -> Response {
    let topo = match &state.cluster {
        Some(v) => v.topology(),
        None => ai_engine_core::cluster_view::TopologySnapshot::default(),
    };
    let snapshot = ai_engine_graph::scan(&topo, &state.activity.recent());
    axum::Json(snapshot).into_response()
}
