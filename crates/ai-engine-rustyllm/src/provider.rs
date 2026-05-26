//! `RustyllmProvider`: `ai_engine_provider::Provider` backed by
//! `rustyllm::StreamingLlama` (`kind = "rustyllm"`).
//!
//! Generation runs a token-at-a-time decode loop over the engine's public
//! `forward` + `StreamCache`, so `chat_stream` emits incremental deltas and
//! only generated tokens are decoded (no prompt-stripping guesswork).

use std::sync::{Arc, Mutex};

use ai_engine_provider::{
    error::ProviderError,
    openai,
    provider::{CallCtx, Capabilities, Credentials, EventStream, Provider},
};
use async_trait::async_trait;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::llama::LlamaEosToks;
use rustyllm::inference::LoadOptions;
use rustyllm::streaming::StreamCache;
use rustyllm::StreamingLlama;
use tokenizers::Tokenizer;

/// rustyllm's `LoadOptions::default` loads weights as F16, so the KV cache
/// (and our `StreamCache`) must match.
const MODEL_DTYPE: DType = DType::F16;

/// Fixed RNG seed, matching rustyllm's own `generate` for parity.
const SEED: u64 = 299_792_458;

pub struct RustyllmProvider {
    id: String,
    // rustyllm streams layers with `&self` interior state of unknown
    // thread-safety; the Mutex serialises generation and makes the
    // provider unconditionally Send + Sync across candle device backends.
    model: Arc<Mutex<StreamingLlama>>,
    tokenizer: Arc<Tokenizer>,
    max_seq_len: usize,
    default_max_tokens: usize,
}

impl RustyllmProvider {
    /// Load an HF-safetensors checkpoint. `model_path` must be a local
    /// model directory containing `config.json`, `tokenizer.json`, and the
    /// safetensors weights. `device_spec` is `auto` | `cpu` | `cuda:N` | `metal`.
    pub fn new(
        id: impl Into<String>,
        model_path: &str,
        device_spec: &str,
        max_seq_len: usize,
    ) -> anyhow::Result<Self> {
        let id = id.into();
        let device = resolve_device(device_spec)?;
        let opts = LoadOptions {
            device,
            max_seq_len,
            ..Default::default()
        };
        let mut model = StreamingLlama::from_pretrained(model_path, opts)
            .map_err(|e| anyhow::anyhow!("rustyllm load '{model_path}': {e}"))?;
        // rustyllm streams every layer from disk per token by default — only
        // worth it for models too big for VRAM. When the model fits (the common
        // case for a gateway provider), pin all layers resident on the device so
        // we don't re-stream weights each token. Without this a 1.1B on a 12 GiB
        // GPU crawls at ~1 tok/s; pinned it runs at full GPU speed.
        let n_layers = model.config().num_hidden_layers;
        model
            .pin_resident_layers(n_layers)
            .map_err(|e| anyhow::anyhow!("rustyllm pin resident layers: {e}"))?;

        let tok_path = std::path::Path::new(model_path).join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| {
            anyhow::anyhow!(
                "rustyllm provider '{id}': could not load tokenizer at {}: {e}. \
                 weights_path must be a local model directory containing tokenizer.json",
                tok_path.display()
            )
        })?;

        Ok(Self {
            id,
            model: Arc::new(Mutex::new(model)),
            tokenizer: Arc::new(tokenizer),
            max_seq_len,
            default_max_tokens: 256,
        })
    }

    fn gen_params(&self, req: &openai::ChatRequest) -> GenParams {
        GenParams {
            prompt: render_prompt(req),
            max_new: req
                .max_tokens
                .map(|m| m as usize)
                .unwrap_or(self.default_max_tokens),
            temperature: req.temperature.unwrap_or(0.0) as f64,
            top_p: req.extras.get("top_p").and_then(|v| v.as_f64()),
        }
    }
}

struct GenParams {
    prompt: String,
    max_new: usize,
    temperature: f64,
    top_p: Option<f64>,
}

#[async_trait]
impl Provider for RustyllmProvider {
    fn id(&self) -> &str {
        &self.id
    }

    fn kind(&self) -> &'static str {
        "rustyllm"
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            chat: true,
            streaming: true,
            tools: false,
            vision: false,
            messages: false,
            embeddings: false,
        }
    }

    async fn chat(
        &self,
        req: openai::ChatRequest,
        _creds: &Credentials,
        ctx: &CallCtx,
    ) -> Result<openai::ChatResponse, ProviderError> {
        let params = self.gen_params(&req);
        let model = self.model.clone();
        let tokenizer = self.tokenizer.clone();
        let max_seq_len = self.max_seq_len;

        let (ids, prompt_tokens) = tokio::task::spawn_blocking(move || {
            let m = model.lock().expect("rustyllm model mutex poisoned");
            let mut ids: Vec<u32> = Vec::new();
            let prompt_tokens =
                run_generation(&m, &tokenizer, max_seq_len, &params, |t| ids.push(t))?;
            Ok::<_, anyhow::Error>((ids, prompt_tokens))
        })
        .await
        .map_err(|e| ProviderError::InvalidResponse(format!("rustyllm task join: {e}")))?
        .map_err(|e| ProviderError::InvalidResponse(format!("rustyllm generate: {e}")))?;

        let completion_tokens = ids.len();
        let content = self
            .tokenizer
            .decode(&ids, true)
            .map_err(|e| ProviderError::InvalidResponse(format!("rustyllm decode: {e}")))?;
        Ok(build_chat_response(&req, ctx, content, prompt_tokens, completion_tokens))
    }

    async fn chat_stream(
        &self,
        req: openai::ChatRequest,
        _creds: &Credentials,
        ctx: &CallCtx,
    ) -> Result<EventStream<openai::ChatStreamEvent>, ProviderError> {
        let params = self.gen_params(&req);
        let model = self.model.clone();
        let tokenizer = self.tokenizer.clone();
        let max_seq_len = self.max_seq_len;
        let id = format!("chatcmpl-{}", ctx.request_id);
        let chat_model = req.model.clone();

        // Blocking decode loop on a worker thread; each sampled token id is
        // pushed onto the channel as it is produced. `prompt_tokens` (known
        // only after encoding) comes back via a oneshot so the final chunk
        // can carry an OpenAI-style usage object.
        let (tok_tx, mut tok_rx) = tokio::sync::mpsc::unbounded_channel::<Result<u32, String>>();
        let (prompt_tx, prompt_rx) = tokio::sync::oneshot::channel::<usize>();
        let gen_tokenizer = tokenizer.clone();
        tokio::task::spawn_blocking(move || {
            let m = model.lock().expect("rustyllm model mutex poisoned");
            match run_generation(&m, &gen_tokenizer, max_seq_len, &params, |t| {
                let _ = tok_tx.send(Ok(t));
            }) {
                Ok(prompt_tokens) => {
                    let _ = prompt_tx.send(prompt_tokens);
                }
                Err(e) => {
                    let _ = tok_tx.send(Err(format!("rustyllm generate: {e}")));
                }
            }
        });

        // Decode incrementally: re-decode the full emitted prefix each step
        // and emit only the newly-revealed suffix (handles multi-token
        // unicode the same way the candle provider does).
        let stream = async_stream::stream! {
            let mut emitted: Vec<u32> = Vec::new();
            let mut prev_text = String::new();
            while let Some(item) = tok_rx.recv().await {
                match item {
                    Ok(tok) => {
                        emitted.push(tok);
                        match tokenizer.decode(&emitted, true) {
                            Ok(full) => {
                                if full.len() > prev_text.len() {
                                    let suffix = full[prev_text.len()..].to_string();
                                    prev_text = full;
                                    if !suffix.is_empty() {
                                        let raw = serde_json::json!({
                                            "id": id, "object": "chat.completion.chunk", "model": chat_model,
                                            "choices": [{"index": 0, "delta": {"content": suffix}, "finish_reason": serde_json::Value::Null}],
                                        });
                                        yield Ok(openai::ChatStreamEvent { raw });
                                    }
                                }
                            }
                            Err(e) => {
                                yield Err(ProviderError::Stream(format!("rustyllm decode: {e}")));
                                return;
                            }
                        }
                    }
                    Err(e) => {
                        yield Err(ProviderError::Stream(e));
                        return;
                    }
                }
            }
            // Final chunk carries usage so the gateway's stream tap credits
            // output tokens (tok/s, served counters) — OpenAI's
            // `stream_options.include_usage` convention.
            let completion_tokens = emitted.len();
            let prompt_tokens = prompt_rx.await.unwrap_or(0);
            let done = serde_json::json!({
                "id": id, "object": "chat.completion.chunk", "model": chat_model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            });
            yield Ok(openai::ChatStreamEvent { raw: done });
        };
        Ok(Box::pin(stream))
    }
}

/// Token-at-a-time decode loop mirroring `StreamingLlama::generate`, but
/// invoking `on_token` for each generated (non-EOS) token. Returns the
/// prompt token count. Runs synchronously — call inside `spawn_blocking`.
fn run_generation(
    model: &StreamingLlama,
    tokenizer: &Tokenizer,
    max_seq_len: usize,
    params: &GenParams,
    mut on_token: impl FnMut(u32),
) -> anyhow::Result<usize> {
    let enc = tokenizer
        .encode(params.prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
    let mut tokens: Vec<u32> = enc.get_ids().to_vec();
    if tokens.len() > max_seq_len {
        tokens.truncate(max_seq_len);
    }
    let prompt_tokens = tokens.len();

    let mut lp = LogitsProcessor::new(
        SEED,
        Some(params.temperature).filter(|&t| t > 0.0),
        params.top_p,
    );
    let cfg = model.config();
    let device = model.device();
    let mut cache = StreamCache::new(true, MODEL_DTYPE, cfg, device)?;

    let mut index_pos = 0usize;
    for _ in 0..params.max_new {
        let context_size = if index_pos > 0 { 1 } else { tokens.len() };
        let ctxt = tokens.len().saturating_sub(context_size);
        let input = Tensor::new(&tokens[ctxt..], device)?.unsqueeze(0)?;
        let logits = model.forward(&input, index_pos, &mut cache)?;
        let logits = logits.i(0)?;
        index_pos += context_size;

        let next = lp.sample(&logits)?;
        if is_eos(cfg.eos_token_id.as_ref(), next) {
            break;
        }
        tokens.push(next);
        on_token(next);
    }
    Ok(prompt_tokens)
}

fn is_eos(eos: Option<&LlamaEosToks>, tok: u32) -> bool {
    match eos {
        Some(LlamaEosToks::Single(id)) => tok == *id,
        Some(LlamaEosToks::Multiple(ids)) => ids.contains(&tok),
        None => false,
    }
}

/// Map a device spec to a candle `Device`. `auto` defers to rustyllm's
/// own best-available pick (Metal → CUDA → CPU depending on build).
fn resolve_device(spec: &str) -> anyhow::Result<Device> {
    match spec {
        "auto" | "" => Ok(rustyllm::inference::best_available_device()),
        "cpu" => Ok(Device::Cpu),
        "metal" => Device::new_metal(0).map_err(|e| anyhow::anyhow!("metal device: {e}")),
        s if s.starts_with("cuda") => {
            let idx = s.strip_prefix("cuda:").and_then(|n| n.parse().ok()).unwrap_or(0);
            Device::new_cuda(idx).map_err(|e| anyhow::anyhow!("cuda:{idx} device: {e}"))
        }
        other => anyhow::bail!("unsupported device spec '{other}' (auto|cpu|cuda:N|metal)"),
    }
}

/// Flatten OpenAI chat messages into the Zephyr/TinyLlama chat template
/// (`<|role|>\n{text}</s>\n` per turn). The trailing `</s>` on each turn is
/// what the chat-tuned model is trained on, so it emits its own `</s>` (EOS)
/// at the end of the reply — which `is_eos` catches, preventing the model from
/// hallucinating further `<|user|>`/`<|assistant|>` turns.
fn render_prompt(req: &openai::ChatRequest) -> String {
    let mut out = String::new();
    for m in &req.messages {
        let text = match &m.content {
            openai::ChatContent::Text(s) => s.clone(),
            openai::ChatContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| p.get("text").and_then(|t| t.as_str()).map(String::from))
                .collect::<Vec<_>>()
                .join("\n"),
        };
        out.push_str(&format!("<|{}|>\n{}</s>\n", m.role, text));
    }
    out.push_str("<|assistant|>\n");
    out
}

fn build_chat_response(
    req: &openai::ChatRequest,
    ctx: &CallCtx,
    content: String,
    prompt_tokens: usize,
    completion_tokens: usize,
) -> openai::ChatResponse {
    openai::ChatResponse {
        id: format!("chatcmpl-{}", ctx.request_id),
        model: req.model.clone(),
        choices: vec![openai::ChatChoice {
            index: 0,
            message: openai::ChatMessage {
                role: "assistant".into(),
                content: openai::ChatContent::Text(content),
                extras: Default::default(),
            },
            finish_reason: Some("stop".into()),
            extras: Default::default(),
        }],
        usage: Some(openai::Usage {
            prompt_tokens: prompt_tokens as u32,
            completion_tokens: completion_tokens as u32,
            total_tokens: (prompt_tokens + completion_tokens) as u32,
        }),
        extras: Default::default(),
    }
}
