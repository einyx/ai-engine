use ai_engine_core::ctx::{RequestBody, RequestCtx};
use ai_engine_core::error::GatewayError;
use ai_engine_core::stage::{Stage, StageError, StageOutcome};
use ai_engine_provider::{anthropic, openai};
use regex::RegexSet;

pub struct ContentPolicyStage {
    pub max_request_bytes: usize,
    pub patterns: RegexSet,
}

impl ContentPolicyStage {
    pub fn new(max_request_bytes: usize, patterns: Vec<String>) -> anyhow::Result<Self> {
        Ok(Self {
            max_request_bytes,
            patterns: RegexSet::new(&patterns)?,
        })
    }

    fn is_match(&self, s: &str) -> bool {
        !self.patterns.is_empty() && self.patterns.is_match(s)
    }
}

#[async_trait::async_trait]
impl Stage for ContentPolicyStage {
    fn name(&self) -> &'static str { "content_policy" }

    async fn process(&self, ctx: &mut RequestCtx) -> Result<StageOutcome, StageError> {
        if ctx.raw_body_len > self.max_request_bytes {
            return Err(StageError {
                stage: self.name(),
                error: GatewayError::PayloadTooLarge,
            });
        }
        let blocked = match &ctx.body {
            RequestBody::OpenAiChat(c) => scan_openai_chat(c, |s| self.is_match(s)),
            RequestBody::AnthropicMessages(m) => scan_anthropic_messages(m, |s| self.is_match(s)),
            RequestBody::OpenAiEmbeddings(e) => scan_openai_embeddings(e, |s| self.is_match(s)),
            RequestBody::Empty => false,
        };
        if blocked {
            Err(StageError {
                stage: self.name(),
                error: GatewayError::BadRequest("prompt blocked by policy".into()),
            })
        } else {
            Ok(StageOutcome::Continue)
        }
    }
}

fn scan_openai_chat<F: FnMut(&str) -> bool>(c: &openai::ChatRequest, mut hit: F) -> bool {
    c.messages.iter().any(|m| match &m.content {
        openai::ChatContent::Text(s) => hit(s),
        openai::ChatContent::Parts(parts) => parts.iter().any(|p|
            p.get("text").and_then(|t| t.as_str()).map(&mut hit).unwrap_or(false)
        ),
    })
}

fn scan_anthropic_messages<F: FnMut(&str) -> bool>(m: &anthropic::MessagesRequest, mut hit: F) -> bool {
    let sys_hit = m.system.as_ref().map(|s| match s {
        anthropic::SystemPrompt::Text(t) => hit(t),
        anthropic::SystemPrompt::Blocks(b) => b.iter().any(|x|
            x.get("text").and_then(|t| t.as_str()).map(&mut hit).unwrap_or(false)
        ),
    }).unwrap_or(false);
    sys_hit || m.messages.iter().any(|msg| match &msg.content {
        anthropic::MessageContent::Text(t) => hit(t),
        anthropic::MessageContent::Blocks(b) => b.iter().any(|x|
            x.get("text").and_then(|t| t.as_str()).map(&mut hit).unwrap_or(false)
        ),
    })
}

fn scan_openai_embeddings<F: FnMut(&str) -> bool>(e: &openai::EmbeddingsRequest, mut hit: F) -> bool {
    match &e.input {
        openai::EmbeddingsInput::Single(s) => hit(s),
        openai::EmbeddingsInput::Many(v) => v.iter().any(|s| hit(s)),
    }
}
