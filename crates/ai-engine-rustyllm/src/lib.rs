//! ai-engine-rustyllm
//!
//! Wraps the `rustyllm` layer-wise streaming inference engine as an
//! `ai_engine_provider::Provider` under `kind = "rustyllm"`. rustyllm's
//! selling point is running large models on small GPUs by streaming one
//! transformer layer at a time, so this provider targets HF-safetensors
//! checkpoints (a model directory or hub repo id) rather than GGUF.

mod provider;

pub use provider::RustyllmProvider;
