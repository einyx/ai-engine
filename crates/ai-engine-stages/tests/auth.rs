use ai_engine_core::ctx::{Identity, RequestBody, RequestCtx};
use ai_engine_core::error::GatewayError;
use ai_engine_core::stage::{Stage, StageOutcome};
use ai_engine_stages::auth::{AuthMode, AuthStage};
use http::{HeaderMap, HeaderValue};

fn ctx(auth_header: Option<&str>) -> RequestCtx {
    let mut h = HeaderMap::new();
    if let Some(v) = auth_header {
        h.insert("authorization", HeaderValue::from_str(v).unwrap());
    }
    RequestCtx::new("/v1/chat/completions", h, 0, RequestBody::Empty)
}

#[tokio::test]
async fn passthrough_with_bearer_populates_anonymous_raw_bearer() {
    let s = AuthStage { mode: AuthMode::Passthrough };
    let mut c = ctx(Some("Bearer sk-user-abc"));
    let r = s.process(&mut c).await.unwrap();
    assert!(matches!(r, StageOutcome::Continue));
    match c.identity {
        Some(Identity::Anonymous { raw_bearer: Some(b) }) => assert_eq!(b, "sk-user-abc"),
        other => panic!("expected Anonymous with raw_bearer, got {:?}", other.is_some()),
    }
}

#[tokio::test]
async fn passthrough_without_bearer_populates_anonymous_none() {
    let s = AuthStage { mode: AuthMode::Passthrough };
    let mut c = ctx(None);
    s.process(&mut c).await.unwrap();
    assert!(matches!(c.identity, Some(Identity::Anonymous { raw_bearer: None })));
}

#[tokio::test]
async fn shared_key_match_sets_holder_name() {
    let s = AuthStage {
        mode: AuthMode::SharedKey {
            keys: vec![("good-key".into(), "alice".into())],
        },
    };
    let mut c = ctx(Some("Bearer good-key"));
    s.process(&mut c).await.unwrap();
    match c.identity {
        Some(Identity::Holder { name }) => assert_eq!(name, "alice"),
        _ => panic!("expected Holder"),
    }
}

#[tokio::test]
async fn shared_key_mismatch_returns_unauthorized() {
    let s = AuthStage {
        mode: AuthMode::SharedKey {
            keys: vec![("good-key".into(), "alice".into())],
        },
    };
    let mut c = ctx(Some("Bearer wrong-key"));
    let err = s.process(&mut c).await.unwrap_err();
    assert!(matches!(err.error, GatewayError::Unauthorized));
}

#[tokio::test]
async fn shared_key_missing_header_returns_unauthorized() {
    let s = AuthStage {
        mode: AuthMode::SharedKey {
            keys: vec![("good".into(), "a".into())],
        },
    };
    let mut c = ctx(None);
    let err = s.process(&mut c).await.unwrap_err();
    assert!(matches!(err.error, GatewayError::Unauthorized));
}

#[tokio::test]
async fn lowercase_bearer_prefix_accepted() {
    // Some SDKs send `bearer ` (lowercase). Be lenient.
    let s = AuthStage {
        mode: AuthMode::SharedKey {
            keys: vec![("k".into(), "u".into())],
        },
    };
    let mut c = ctx(Some("bearer k"));
    s.process(&mut c).await.unwrap();
    assert!(matches!(c.identity, Some(Identity::Holder { .. })));
}

#[tokio::test]
async fn first_matching_master_key_wins() {
    let s = AuthStage {
        mode: AuthMode::SharedKey {
            keys: vec![
                ("k".into(), "first".into()),
                ("k".into(), "second".into()), // duplicate key, different name
            ],
        },
    };
    let mut c = ctx(Some("Bearer k"));
    s.process(&mut c).await.unwrap();
    match c.identity {
        Some(Identity::Holder { name }) => assert_eq!(name, "first"),
        _ => panic!(),
    }
}
