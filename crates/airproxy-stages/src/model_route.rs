use airproxy_core::ctx::{ProviderBinding, RequestBody, RequestCtx};
use airproxy_core::error::GatewayError;
use airproxy_core::stage::{Stage, StageError, StageOutcome};
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
        for rule in &self.rules {
            if rule.matcher.is_match(model) {
                ctx.binding = Some(ProviderBinding {
                    provider_id: rule.provider_id.clone(),
                    upstream_model: rule.upstream_model.clone().unwrap_or_else(|| model.clone()),
                });
                return Ok(StageOutcome::Continue);
            }
        }
        Err(StageError {
            stage: self.name(),
            error: GatewayError::NoRouteForModel { model: model.clone() },
        })
    }
}
