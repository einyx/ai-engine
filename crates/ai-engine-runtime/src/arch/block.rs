use crate::arch::{attention::Attention, ffn::SwiGluFfn, rmsnorm::RmsNorm};
use crate::kv_cache::KvCacheSlot;
use burn::tensor::{backend::Backend, Tensor};

pub struct DecoderBlock<B: Backend> {
    pub attn_norm: RmsNorm<B>,
    pub attn: Attention<B>,
    pub ffn_norm: RmsNorm<B>,
    pub ffn: SwiGluFfn<B>,
}

impl<B: Backend> DecoderBlock<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        positions: &[i32],
        cache: &mut KvCacheSlot<B>,
    ) -> Tensor<B, 3> {
        // Residual 1: x = x + attn(norm(x))
        let h = self.attn_norm.forward(x.clone());
        let h = self.attn.forward(h, positions, cache);
        let x = x.add(h);
        // Residual 2: x = x + ffn(norm(x))
        let h = self.ffn_norm.forward(x.clone());
        let h = self.ffn.forward(h);
        x.add(h)
    }
}
