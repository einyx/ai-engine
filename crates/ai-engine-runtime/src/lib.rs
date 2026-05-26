//! ai-engine-runtime
//!
//! Single-node inference primitives. Distributed orchestration lives in
//! ai-engine-cluster (Plan 2).

pub mod arch;
pub mod backend;
pub mod gguf;
pub mod config;
pub mod kv_cache;
pub mod loader;
pub mod name_map;
pub mod quant;
pub mod request;
pub mod sample;
pub mod tokenizer_gguf;

pub use arch::model::Model;
pub use backend::BackendKind;
pub use config::{ModelConfig, ModelFamily};
pub use kv_cache::KvCacheSlot;
pub use loader::{load_gguf, load_range, load_weights, LoadedWeights};
pub use quant::{Q4Tensor, Q4_GROUP_SIZE, QuantizedTensor};
pub use request::RequestState;
pub use sample::{sample, SamplingConfig};
pub use tokenizer_gguf::{load_tokenizer_from_gguf, load_tokenizer_from_gguf_metadata};
