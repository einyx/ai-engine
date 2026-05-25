//! Gather-based paged attention.

use candle_core::{Device, Tensor, D};
use crate::paged::block_table::{BlockAllocator, BlockTable};

/// Build an additive attention mask of shape (batch, 1, q_len, kv_len).
/// `seq_lens[b]` = number of valid keys for row b; `q_positions[b]` = global
/// position of the (single) query token for row b (decode: q_len==1).
/// Entry is 0.0 where attention is allowed, -inf where masked (key index >=
/// seq_lens[b], i.e. padding, OR key position > query position, i.e. future).
pub fn build_mask(
    seq_lens: &[usize],
    q_positions: &[usize],
    kv_len: usize,
    device: &Device,
) -> candle_core::Result<Tensor> {
    let batch = seq_lens.len();
    let neg = f32::NEG_INFINITY;
    let mut data = vec![0f32; batch * kv_len];
    for b in 0..batch {
        for k in 0..kv_len {
            if k >= seq_lens[b] || k > q_positions[b] {
                data[b * kv_len + k] = neg;
            }
        }
    }
    Tensor::from_vec(data, (batch, kv_len), device)?.reshape((batch, 1, 1, kv_len))
}

/// Per-layer paged KV storage. K_pool/V_pool: (num_blocks, block_size, n_kv_head, head_dim).
pub struct KvPool {
    k_pool: Tensor,
    v_pool: Tensor,
    block_size: usize,
    n_kv_head: usize,
    head_dim: usize,
}

impl KvPool {
    pub fn new(
        num_blocks: usize,
        block_size: usize,
        n_kv_head: usize,
        head_dim: usize,
        device: &Device,
    ) -> candle_core::Result<Self> {
        let shape = (num_blocks, block_size, n_kv_head, head_dim);
        Ok(Self {
            k_pool: Tensor::zeros(shape, candle_core::DType::F32, device)?,
            v_pool: Tensor::zeros(shape, candle_core::DType::F32, device)?,
            block_size,
            n_kv_head,
            head_dim,
        })
    }

    /// Write `k`/`v` (n_new, n_kv_head, head_dim) for a sequence starting at
    /// token position `start_pos`, using its block table.
    pub fn write(
        &mut self,
        alloc: &BlockAllocator,
        table: &BlockTable,
        start_pos: usize,
        k: &Tensor,
        v: &Tensor,
    ) -> candle_core::Result<()> {
        let n_new = k.dim(0)?;
        for i in 0..n_new {
            let (block_id, slot) = alloc.locate(table, start_pos + i);
            let k_row = k.narrow(0, i, 1)?.reshape((1, 1, self.n_kv_head, self.head_dim))?;
            let v_row = v.narrow(0, i, 1)?.reshape((1, 1, self.n_kv_head, self.head_dim))?;
            let bid = block_id as usize;
            self.k_pool = self.k_pool.slice_assign(
                &[bid..bid + 1, slot..slot + 1, 0..self.n_kv_head, 0..self.head_dim],
                &k_row,
            )?;
            self.v_pool = self.v_pool.slice_assign(
                &[bid..bid + 1, slot..slot + 1, 0..self.n_kv_head, 0..self.head_dim],
                &v_row,
            )?;
        }
        Ok(())
    }

    /// Gather a sequence's KV up to `len` tokens into (len, n_kv_head, head_dim).
    pub fn gather_seq(&self, table: &BlockTable, len: usize) -> candle_core::Result<(Tensor, Tensor)> {
        let mut idx = Vec::with_capacity(len);
        for pos in 0..len {
            let block_id = table.blocks[pos / self.block_size];
            idx.push(block_id * self.block_size as u32 + (pos % self.block_size) as u32);
        }
        let dev = self.k_pool.device();
        let idx = Tensor::new(idx.as_slice(), dev)?;
        let flat_k = self.k_pool.reshape(((), self.n_kv_head, self.head_dim))?;
        let flat_v = self.v_pool.reshape(((), self.n_kv_head, self.head_dim))?;
        Ok((flat_k.index_select(&idx, 0)?, flat_v.index_select(&idx, 0)?))
    }
}

/// Scaled-dot-product attention for a decode batch.
/// `q`: (batch, n_head, q_len, head_dim). `k`,`v`: (batch, n_kv_head, kv_len, head_dim).
/// `mask`: (batch, 1, q_len, kv_len) additive. Returns (batch, q_len, n_head*head_dim).
pub fn sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: &Tensor,
    n_head: usize,
    n_kv_head: usize,
) -> candle_core::Result<Tensor> {
    let head_dim = q.dim(D::Minus1)?;
    let (k, v) = if n_kv_head != n_head {
        let repeat = n_head / n_kv_head;
        (repeat_kv(k, repeat)?, repeat_kv(v, repeat)?)
    } else {
        (k.clone(), v.clone())
    };
    let scale = 1.0 / (head_dim as f64).sqrt();
    let att = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)? * scale)?;
    let att = att.broadcast_add(mask)?;
    let att = candle_nn::ops::softmax_last_dim(&att)?;
    let out = att.matmul(&v.contiguous()?)?;
    let (b, _h, q_len, _d) = out.dims4()?;
    out.transpose(1, 2)?.reshape((b, q_len, n_head * head_dim))
}

fn repeat_kv(x: &Tensor, repeat: usize) -> candle_core::Result<Tensor> {
    if repeat == 1 {
        return Ok(x.clone());
    }
    let (b, n_kv, s, d) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, n_kv, repeat, s, d))?
        .reshape((b, n_kv * repeat, s, d))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_blocks_padding_and_future() {
        let dev = Device::Cpu;
        let m = build_mask(&[3, 1], &[2, 0], 4, &dev).unwrap();
        assert_eq!(m.dims(), &[2, 1, 1, 4]);
        let v: Vec<f32> = m.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(v[0], 0.0);
        assert_eq!(v[1], 0.0);
        assert_eq!(v[2], 0.0);
        assert!(v[3].is_infinite() && v[3] < 0.0);
        assert_eq!(v[4], 0.0);
        assert!(v[5].is_infinite());
        assert!(v[6].is_infinite());
        assert!(v[7].is_infinite());
    }
}
