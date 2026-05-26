//! Paged-KV continuous-batching inference engine.
//!
//! One model instance time-shares many sequences via a fixed-block KV pool
//! and a single-threaded scheduler step loop. See
//! `docs/superpowers/specs/2026-05-25-candle-paged-continuous-batching-design.md`.

pub mod arch;
pub mod attention;
pub mod block_table;
pub mod engine;
pub mod rope;
pub mod transformer;
