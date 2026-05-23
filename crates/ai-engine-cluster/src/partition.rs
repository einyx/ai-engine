use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionManifest {
    pub model_id: String,
    // Filled out in Task 6.
}
