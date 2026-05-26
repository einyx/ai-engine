use ai_engine_core::ctx::{RequestBody, RequestCtx};
use ai_engine_core::error::GatewayError;
use ai_engine_core::stage::{Stage, StageOutcome};
use ai_engine_provider::openai::{ChatContent, ChatMessage, ChatRequest};
use ai_engine_stages::model_route::ModelRouteStage;
use http::HeaderMap;

fn chat(model: &str) -> RequestBody {
    RequestBody::OpenAiChat(ChatRequest {
        model: model.into(),
        messages: vec![ChatMessage { role: "user".into(), content: ChatContent::Text("hi".into()), extras: Default::default() }],
        stream: None, temperature: None, max_tokens: None, stream_options: None, extras: Default::default(),
    })
}

fn ctx(body: RequestBody) -> RequestCtx {
    RequestCtx::new("/v1/chat/completions", HeaderMap::new(), 0, body)
}

#[tokio::test]
async fn all_matching_rules_become_candidates_in_order() {
    let s = ModelRouteStage::from_strings(vec![
        ("gpt-4o".into(), "exact".into(), None),
        ("gpt-*".into(),  "glob".into(),  None),
    ]).unwrap();
    let mut c = ctx(chat("gpt-4o"));
    s.process(&mut c).await.unwrap();
    // Both rules match gpt-4o; the pool keeps route order (exact first).
    assert_eq!(c.binding.as_ref().unwrap().candidates, vec!["exact", "glob"]);
}

#[tokio::test]
async fn glob_matches_when_no_exact() {
    let s = ModelRouteStage::from_strings(vec![
        ("gpt-4o".into(), "exact".into(), None),
        ("gpt-*".into(),  "glob".into(),  None),
    ]).unwrap();
    let mut c = ctx(chat("gpt-3.5-turbo"));
    s.process(&mut c).await.unwrap();
    assert_eq!(c.binding.as_ref().unwrap().candidates, vec!["glob"]);
}

#[tokio::test]
async fn upstream_model_override_applied() {
    let s = ModelRouteStage::from_strings(vec![
        ("gpt-4o".into(), "openai".into(), Some("gpt-4o-2024-08-06".into())),
    ]).unwrap();
    let mut c = ctx(chat("gpt-4o"));
    s.process(&mut c).await.unwrap();
    let b = c.binding.as_ref().unwrap();
    assert_eq!(b.candidates, vec!["openai"]);
    assert_eq!(b.upstream_model, "gpt-4o-2024-08-06");
}

#[tokio::test]
async fn falls_back_to_request_model_when_no_override() {
    let s = ModelRouteStage::from_strings(vec![
        ("gpt-*".into(), "openai".into(), None),
    ]).unwrap();
    let mut c = ctx(chat("gpt-3.5"));
    s.process(&mut c).await.unwrap();
    assert_eq!(c.binding.as_ref().unwrap().upstream_model, "gpt-3.5");
}

#[tokio::test]
async fn no_match_returns_no_route_error() {
    let s = ModelRouteStage::from_strings(vec![
        ("gpt-*".into(), "openai".into(), None),
    ]).unwrap();
    let mut c = ctx(chat("claude-3"));
    let err = s.process(&mut c).await.unwrap_err();
    assert!(matches!(err.error, GatewayError::NoRouteForModel { ref model } if model == "claude-3"));
}

#[tokio::test]
async fn empty_body_is_continue_no_binding() {
    let s = ModelRouteStage::from_strings(vec![
        ("gpt-*".into(), "openai".into(), None),
    ]).unwrap();
    let mut c = RequestCtx::new("/v1/models", HeaderMap::new(), 0, RequestBody::Empty);
    let r = s.process(&mut c).await.unwrap();
    assert!(matches!(r, StageOutcome::Continue));
    assert!(c.binding.is_none());
}
