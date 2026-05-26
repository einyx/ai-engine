use ai_engine_core::ctx::{ProviderBinding, RequestBody, RequestCtx};
use ai_engine_core::error::GatewayError;
use ai_engine_core::stage::{Stage, StageError, StageOutcome};
use ai_engine_provider::openai::ChatContent;
use globset::{Glob, GlobMatcher};

pub struct RouteRule {
    pub pattern: String,
    pub matcher: GlobMatcher,
    pub provider_id: String,
    pub upstream_model: Option<String>,
}

pub struct ModelRouteStage {
    pub rules: Vec<RouteRule>,
}

impl ModelRouteStage {
    /// Build from `(model_glob, provider_id, upstream_model_override)` triples.
    pub fn from_strings(
        rules: Vec<(String, String, Option<String>)>,
    ) -> anyhow::Result<Self> {
        let mut out = Vec::with_capacity(rules.len());
        for (pat, prov, up) in rules {
            let matcher = Glob::new(&pat)?.compile_matcher();
            out.push(RouteRule {
                pattern: pat,
                matcher,
                provider_id: prov,
                upstream_model: up,
            });
        }
        Ok(Self { rules: out })
    }
}

#[async_trait::async_trait]
impl Stage for ModelRouteStage {
    fn name(&self) -> &'static str { "model_route" }

    async fn process(&self, ctx: &mut RequestCtx) -> Result<StageOutcome, StageError> {
        let model = match &ctx.body {
            RequestBody::OpenAiChat(c) => &c.model,
            RequestBody::AnthropicMessages(m) => &m.model,
            RequestBody::OpenAiEmbeddings(e) => &e.model,
            RequestBody::Empty => return Ok(StageOutcome::Continue),
        };
        // Collect every provider whose rule matches, in route order, deduped.
        // `forward` load-balances across them and fails over on error.
        let mut candidates: Vec<String> = Vec::new();
        let mut upstream_model: Option<String> = None;
        for rule in &self.rules {
            if rule.matcher.is_match(model) {
                if !candidates.contains(&rule.provider_id) {
                    candidates.push(rule.provider_id.clone());
                }
                // The first matching rule defines the upstream model rewrite.
                upstream_model
                    .get_or_insert_with(|| rule.upstream_model.clone().unwrap_or_else(|| model.clone()));
            }
        }
        if candidates.is_empty() {
            return Err(StageError {
                stage: self.name(),
                error: GatewayError::NoRouteForModel { model: model.clone() },
            });
        }
        // Capture a short prompt preview for the activity graph, while the body
        // is still here (forward consumes it). Last user message, text only.
        let prompt_preview: Option<String> = match &ctx.body {
            RequestBody::OpenAiChat(c) => c
                .messages
                .iter()
                .rev()
                .find(|m| m.role == "user")
                .and_then(|m| match &m.content {
                    ChatContent::Text(t) => Some(t.clone()),
                    _ => None,
                }),
            _ => None,
        };
        ctx.binding = Some(ProviderBinding {
            candidates,
            provider_id: String::new(),
            upstream_model: upstream_model.unwrap_or_else(|| model.clone()),
        });
        if let Some(p) = prompt_preview {
            let snip: String = p.chars().take(200).collect();
            ctx.metadata.insert("prompt_preview", serde_json::Value::String(snip));
        }
        Ok(StageOutcome::Continue)
    }
}
