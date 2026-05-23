//! Backend selection. Filled out in Tasks 15/16.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind { Cpu, Cuda, Metal, Wgpu }
