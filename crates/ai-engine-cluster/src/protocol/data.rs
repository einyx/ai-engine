use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Dtype { F32, F16, Bf16 }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ActivationHeader {
    pub request_id: Uuid,
    pub seq_pos: u32,
    pub shape: [u32; 3],          // [batch, seq, hidden]
    pub dtype: Dtype,
    pub is_terminal: bool,
}
