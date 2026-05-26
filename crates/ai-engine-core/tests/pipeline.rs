use ai_engine_core::ctx::{GatewayResponse, RequestBody, RequestCtx, ResponseSlot};
use ai_engine_core::error::GatewayError;
use ai_engine_core::pipeline::Pipeline;
use ai_engine_core::stage::{Stage, StageError, StageOutcome};
use http::HeaderMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU16, Ordering};

#[derive(Clone)]
struct Marker {
    name: &'static str,
    terminal: bool,
    action: Action,
    counter: Arc<AtomicU16>,
    bit: u16,
}

#[derive(Clone)]
enum Action { Continue, RespondEmpty, Fail }

#[async_trait::async_trait]
impl Stage for Marker {
    fn name(&self) -> &'static str { self.name }
    fn is_terminal(&self) -> bool { self.terminal }
    async fn process(&self, _ctx: &mut RequestCtx) -> Result<StageOutcome, StageError> {
        self.counter.fetch_or(self.bit, Ordering::SeqCst);
        match self.action {
            Action::Continue => Ok(StageOutcome::Continue),
            Action::RespondEmpty => Ok(StageOutcome::Respond(GatewayResponse::default())),
            Action::Fail => Err(StageError {
                stage: self.name,
                error: GatewayError::Internal(anyhow::anyhow!("boom")),
            }),
        }
    }
}

fn empty_ctx() -> RequestCtx {
    RequestCtx::new("/v1/test", HeaderMap::new(), 0, RequestBody::Empty)
}

fn counter() -> Arc<AtomicU16> { Arc::new(AtomicU16::new(0)) }

#[tokio::test]
async fn all_continue_runs_every_stage_then_terminals() {
    let c = counter();
    let a = Arc::new(Marker { name: "a", terminal: false, action: Action::Continue, counter: c.clone(), bit: 0b001 });
    let b = Arc::new(Marker { name: "b", terminal: false, action: Action::Continue, counter: c.clone(), bit: 0b010 });
    let t = Arc::new(Marker { name: "t", terminal: true,  action: Action::Continue, counter: c.clone(), bit: 0b100 });
    let pl = Pipeline::new(vec![a, b, t]);
    let mut ctx = empty_ctx();
    pl.execute(&mut ctx).await;
    assert_eq!(c.load(Ordering::SeqCst), 0b111);
    assert!(ctx.error.is_none());
    assert!(matches!(ctx.response, ResponseSlot::Pending));
}

#[tokio::test]
async fn respond_skips_remaining_non_terminals_but_runs_terminal() {
    let c = counter();
    let a = Arc::new(Marker { name: "a", terminal: false, action: Action::RespondEmpty, counter: c.clone(), bit: 0b001 });
    let b = Arc::new(Marker { name: "b", terminal: false, action: Action::Continue, counter: c.clone(), bit: 0b010 });
    let t = Arc::new(Marker { name: "t", terminal: true,  action: Action::Continue, counter: c.clone(), bit: 0b100 });
    let pl = Pipeline::new(vec![a, b, t]);
    let mut ctx = empty_ctx();
    pl.execute(&mut ctx).await;
    // a + t ran; b skipped
    assert_eq!(c.load(Ordering::SeqCst), 0b101);
    assert!(matches!(ctx.response, ResponseSlot::Full(_)));
}

#[tokio::test]
async fn err_skips_remaining_non_terminals_but_runs_terminal() {
    let c = counter();
    let a = Arc::new(Marker { name: "a", terminal: false, action: Action::Fail, counter: c.clone(), bit: 0b001 });
    let b = Arc::new(Marker { name: "b", terminal: false, action: Action::Continue, counter: c.clone(), bit: 0b010 });
    let t = Arc::new(Marker { name: "t", terminal: true,  action: Action::Continue, counter: c.clone(), bit: 0b100 });
    let pl = Pipeline::new(vec![a, b, t]);
    let mut ctx = empty_ctx();
    pl.execute(&mut ctx).await;
    assert_eq!(c.load(Ordering::SeqCst), 0b101);
    assert!(ctx.error.is_some());
    assert!(matches!(ctx.response, ResponseSlot::Pending));
}

#[tokio::test]
async fn terminal_failures_do_not_clobber_existing_error() {
    let c = counter();
    let a = Arc::new(Marker { name: "a", terminal: false, action: Action::Fail, counter: c.clone(), bit: 0b001 });
    let t = Arc::new(Marker { name: "t", terminal: true,  action: Action::Fail, counter: c.clone(), bit: 0b010 });
    let pl = Pipeline::new(vec![a, t]);
    let mut ctx = empty_ctx();
    pl.execute(&mut ctx).await;
    // Both ran; the FIRST error (from `a`) must still be the ctx.error
    assert_eq!(c.load(Ordering::SeqCst), 0b011);
    assert!(matches!(ctx.error, Some(GatewayError::Internal(_))));
}

#[tokio::test]
async fn terminals_run_in_declared_order_even_with_non_terminals_between() {
    let c = counter();
    // Pipeline interleaves terminals between non-terminals; spec says terminals run AFTER non-terminals, in declared order.
    let t1 = Arc::new(Marker { name: "t1", terminal: true,  action: Action::Continue, counter: c.clone(), bit: 0b001 });
    let a  = Arc::new(Marker { name: "a",  terminal: false, action: Action::Continue, counter: c.clone(), bit: 0b010 });
    let t2 = Arc::new(Marker { name: "t2", terminal: true,  action: Action::Continue, counter: c.clone(), bit: 0b100 });
    let pl = Pipeline::new(vec![t1, a, t2]);
    let mut ctx = empty_ctx();
    pl.execute(&mut ctx).await;
    assert_eq!(c.load(Ordering::SeqCst), 0b111);
}
