//! ai-engine-runtime
//!
//! Single-node inference primitives. Distributed orchestration lives in
//! ai-engine-cluster (Plan 2).

pub mod arch;
pub mod backend;
pub mod config;
pub mod kv_cache;
pub mod loader;
pub mod name_map;
pub mod sample;

pub use arch::model::Model;
pub use backend::BackendKind;
pub use config::{ModelConfig, ModelFamily};
pub use kv_cache::KvCacheSlot;
pub use loader::{load_range, LoadedWeights};
pub use sample::{sample, SamplingConfig};
