use crate::ctx::{GatewayResponse, RequestCtx};
use crate::error::GatewayError;

pub enum StageOutcome {
    Continue,
    Respond(GatewayResponse),
}

pub struct StageError {
    pub stage: &'static str,
    pub error: GatewayError,
}

#[async_trait::async_trait]
pub trait Stage: Send + Sync + 'static {
    fn name(&self) -> &'static str;
    fn is_terminal(&self) -> bool { false }
    async fn process(&self, ctx: &mut RequestCtx) -> Result<StageOutcome, StageError>;
}
