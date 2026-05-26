//! Fixed-size KV block pool.

/// Tracks physical KV block ownership. Block ids index into the per-layer
/// `K_pool`/`V_pool` tensors owned by the attention layer.
pub struct BlockAllocator {
    block_size: usize,
    free: Vec<u32>,
}

/// One sequence's ordered list of physical block ids.
#[derive(Default, Clone)]
pub struct BlockTable {
    pub blocks: Vec<u32>,
}

impl BlockAllocator {
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        // Hand out high ids last so tests see 0,1,2,... first.
        let free = (0..num_blocks as u32).rev().collect();
        Self { block_size, free }
    }

    pub fn free_count(&self) -> usize {
        self.free.len()
    }

    /// Blocks needed to hold `n_tokens` tokens.
    pub fn blocks_for(&self, n_tokens: usize) -> usize {
        n_tokens.div_ceil(self.block_size)
    }

    /// Ensure `table` has capacity for `n_tokens`. Returns `Err` if the free
    /// list cannot satisfy the request (caller treats this as backpressure).
    pub fn ensure_capacity(&mut self, table: &mut BlockTable, n_tokens: usize) -> Result<(), ()> {
        let need = self.blocks_for(n_tokens);
        if need <= table.blocks.len() {
            return Ok(());
        }
        let extra = need - table.blocks.len();
        if extra > self.free.len() {
            return Err(());
        }
        for _ in 0..extra {
            table.blocks.push(self.free.pop().expect("checked above"));
        }
        Ok(())
    }

    /// Return all of `table`'s blocks to the free list and clear it.
    pub fn release(&mut self, table: &mut BlockTable) {
        self.free.extend(table.blocks.drain(..));
    }

    /// Physical (block_id, slot) for a sequence-local token position.
    pub fn locate(&self, table: &BlockTable, pos: usize) -> (u32, usize) {
        let block_idx = pos / self.block_size;
        (table.blocks[block_idx], pos % self.block_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_for_rounds_up() {
        let a = BlockAllocator::new(10, 16);
        assert_eq!(a.blocks_for(0), 0);
        assert_eq!(a.blocks_for(1), 1);
        assert_eq!(a.blocks_for(16), 1);
        assert_eq!(a.blocks_for(17), 2);
    }

    #[test]
    fn ensure_capacity_allocates_and_reuses() {
        let mut a = BlockAllocator::new(4, 16);
        let mut t = BlockTable::default();
        a.ensure_capacity(&mut t, 1).unwrap();
        assert_eq!(t.blocks, vec![0]);
        a.ensure_capacity(&mut t, 16).unwrap();
        assert_eq!(t.blocks, vec![0]);
        a.ensure_capacity(&mut t, 17).unwrap();
        assert_eq!(t.blocks, vec![0, 1]);
        assert_eq!(a.free_count(), 2);
    }

    #[test]
    fn release_returns_blocks() {
        let mut a = BlockAllocator::new(4, 16);
        let mut t = BlockTable::default();
        a.ensure_capacity(&mut t, 40).unwrap();
        assert_eq!(a.free_count(), 1);
        a.release(&mut t);
        assert_eq!(a.free_count(), 4);
        assert!(t.blocks.is_empty());
    }

    #[test]
    fn oom_returns_err_without_partial_alloc() {
        let mut a = BlockAllocator::new(2, 16);
        let mut t = BlockTable::default();
        assert!(a.ensure_capacity(&mut t, 48).is_err());
        assert!(t.blocks.is_empty());
        assert_eq!(a.free_count(), 2);
    }

    #[test]
    fn locate_maps_position_to_block_and_slot() {
        let mut a = BlockAllocator::new(4, 16);
        let mut t = BlockTable::default();
        a.ensure_capacity(&mut t, 32).unwrap();
        assert_eq!(a.locate(&t, 0), (0, 0));
        assert_eq!(a.locate(&t, 15), (0, 15));
        assert_eq!(a.locate(&t, 16), (1, 0));
        assert_eq!(a.locate(&t, 31), (1, 15));
    }
}
