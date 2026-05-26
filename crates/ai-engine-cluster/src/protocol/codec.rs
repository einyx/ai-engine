use serde::{de::DeserializeOwned, Serialize};

pub fn encode<T: Serialize>(value: &T) -> anyhow::Result<Vec<u8>> {
    postcard::to_allocvec(value)
        .map_err(|e| anyhow::anyhow!("postcard encode: {e}"))
}

pub fn decode<T: DeserializeOwned>(bytes: &[u8]) -> anyhow::Result<T> {
    postcard::from_bytes(bytes)
        .map_err(|e| anyhow::anyhow!("postcard decode: {e}"))
}
