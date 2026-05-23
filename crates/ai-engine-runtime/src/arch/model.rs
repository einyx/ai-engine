use crate::arch::{
    block::DecoderBlock,
    embedding::{OutputProjection, TokenEmbedding},
    attention::Attention,
    ffn::SwiGluFfn,
    rmsnorm::RmsNorm,
};
use crate::config::ModelConfig;
use crate::kv_cache::KvCacheSlot;
use burn::tensor::{backend::Backend, Distribution, Int, Tensor};

pub struct Model<B: Backend> {
    pub embedding: TokenEmbedding<B>,
    pub blocks: Vec<DecoderBlock<B>>,
    pub final_norm: RmsNorm<B>,
    pub output: OutputProjection<B>,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq: usize,
}

impl<B: Backend> Model<B> {
    /// Random-weight constructor for shape / smoke tests.
    /// Production path uses `from_loaded` (Task 11).
    pub fn with_random_weights(cfg: &ModelConfig, device: &B::Device) -> Self {
        // Embedding [vocab, hidden]
        let embed_w = Tensor::<B, 2>::random(
            [cfg.vocab_size, cfg.hidden_size],
            Distribution::Default,
            device,
        );
        let embedding = TokenEmbedding::new(embed_w.clone());

        // Stack of decoder blocks
        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            let attn_norm_w = Tensor::<B, 1>::ones([cfg.hidden_size], device);
            let ffn_norm_w = Tensor::<B, 1>::ones([cfg.hidden_size], device);
            let attn_norm = RmsNorm::new(attn_norm_w, cfg.rms_norm_eps);
            let ffn_norm = RmsNorm::new(ffn_norm_w, cfg.rms_norm_eps);

            let attn = Attention::<B>::with_random_weights(
                cfg.hidden_size,
                cfg.n_heads,
                cfg.n_kv_heads,
                cfg.head_dim,
                cfg.max_position_embeddings,
                cfg.rope_theta,
                device,
            );

            let ffn = SwiGluFfn::<B>::with_random_weights(
                cfg.hidden_size,
                cfg.intermediate_size,
                device,
            );

            blocks.push(DecoderBlock {
                attn_norm,
                attn,
                ffn_norm,
                ffn,
            });
        }

        let final_norm = RmsNorm::new(
            Tensor::<B, 1>::ones([cfg.hidden_size], device),
            cfg.rms_norm_eps,
        );

        let output_w = if cfg.tie_word_embeddings {
            // Tied: output projection = embedding^T.
            // [vocab, hidden] -> [hidden, vocab] via swap_dims.
            embed_w.swap_dims(0, 1)
        } else {
            Tensor::<B, 2>::random(
                [cfg.hidden_size, cfg.vocab_size],
                Distribution::Default,
                device,
            )
        };
        let output = OutputProjection::new(output_w);

        Self {
            embedding,
            blocks,
            final_norm,
            output,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            max_seq: cfg.max_position_embeddings,
        }
    }

    /// Used by the shape test: each block gets a fresh KV cache.
    /// Production callers use `forward_with_caches` (added in Task 13).
    pub fn forward(&self, token_ids: Tensor<B, 2, Int>, start_pos: usize) -> Tensor<B, 3> {
        let [batch, seq] = token_ids.dims();
        let positions: Vec<i32> = (start_pos..start_pos + seq).map(|p| p as i32).collect();
        let mut x = self.embedding.forward(token_ids);
        let device = x.device();
        for block in &self.blocks {
            let mut cache = KvCacheSlot::<B>::new(
                batch,
                self.n_kv_heads,
                self.max_seq,
                self.head_dim,
                &device,
            );
            x = block.forward(x, &positions, &mut cache);
        }
        let x = self.final_norm.forward(x);
        self.output.forward(x)
    }
}
