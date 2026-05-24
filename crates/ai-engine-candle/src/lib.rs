//! candle-backed native-quantized local GPU inference provider.
//!
//! Wraps `candle_transformers::models::quantized_llama` to run GGUF Q4/Q6
//! Llama-3 models with native quantized matmul on CUDA/Metal/CPU. Implements
//! the `ai_engine_provider::Provider` trait as `kind = "candle-local"`.

pub mod device;
pub mod model;
pub mod pool;
pub mod provider;

// re-export added in Task 6
// pub use provider::CandleProvider;
