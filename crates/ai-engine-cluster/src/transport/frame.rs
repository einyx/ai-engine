use quinn::{RecvStream, SendStream};

const MAX_FRAME: u32 = 64 * 1024 * 1024; // 64 MiB cap per frame

pub async fn write_frame(stream: &mut SendStream, payload: &[u8]) -> anyhow::Result<()> {
    if payload.len() as u64 > MAX_FRAME as u64 {
        anyhow::bail!("frame too large: {}", payload.len());
    }
    let len = payload.len() as u32;
    stream.write_all(&len.to_le_bytes()).await?;
    stream.write_all(payload).await?;
    Ok(())
}

pub async fn read_frame(stream: &mut RecvStream) -> anyhow::Result<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    stream
        .read_exact(&mut len_buf)
        .await
        .map_err(|e| anyhow::anyhow!("read frame len: {e}"))?;
    let len = u32::from_le_bytes(len_buf);
    if len > MAX_FRAME {
        anyhow::bail!("oversized frame: {len}");
    }
    let mut payload = vec![0u8; len as usize];
    stream
        .read_exact(&mut payload)
        .await
        .map_err(|e| anyhow::anyhow!("read frame payload: {e}"))?;
    Ok(payload)
}
