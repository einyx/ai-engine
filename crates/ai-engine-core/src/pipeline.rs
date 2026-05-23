use std::sync::Arc;

use crate::ctx::{RequestCtx, ResponseSlot};
use crate::stage::{Stage, StageOutcome};

pub struct Pipeline {
    pub stages: Vec<Arc<dyn Stage>>,
}

impl Pipeline {
    pub fn new(stages: Vec<Arc<dyn Stage>>) -> Self {
        Self { stages }
    }

    pub async fn execute(&self, ctx: &mut RequestCtx) {
        let mut short_circuit = false;

        // Pass 1: non-terminal stages
        for stage in self.stages.iter().filter(|s| !s.is_terminal()) {
            if short_circuit { break; }
            match stage.process(ctx).await {
                Ok(StageOutcome::Continue) => {}
                Ok(StageOutcome::Respond(resp)) => {
                    ctx.response = ResponseSlot::Full(resp);
                    short_circuit = true;
                }
                Err(e) => {
                    // First non-terminal failure wins; terminal stages still run.
                    ctx.error = Some(e.error);
                    short_circuit = true;
                }
            }
        }

        // Pass 2: terminal stages — always run, in declared order.
        // Terminal-stage errors are swallowed: log/cleanup MUST NOT mask the
        // primary error or response on the way out.
        for stage in self.stages.iter().filter(|s| s.is_terminal()) {
            let _ = stage.process(ctx).await;
        }
    }
}
