use crate::arch::{
    block::DecoderBlock,
    embedding::{OutputProjection, TokenEmbedding},
    attention::Attention,
    ffn::SwiGluFfn,
    linear::LinearWeight,
    rmsnorm::RmsNorm,
    rope::RotaryEmbedding,
};
use crate::config::ModelConfig;
use crate::kv_cache::KvCacheSlot;
use crate::loader::LoadedWeights;
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
        let output = OutputProjection::new(LinearWeight::Dense(output_w));

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

    /// Production constructor: build a `Model` from `LoadedWeights` (typically
    /// produced by the safetensors loader) and a `ModelConfig`.
    ///
    /// HF safetensors stores `Linear` weights as `[out_features, in_features]`,
    /// but our primitives expect `[in, out]` for the `x @ W` matmul. Every
    /// projection is transposed via `swap_dims(0, 1)` here.
    pub fn from_loaded(
        cfg: &ModelConfig,
        weights: LoadedWeights<B>,
        device: &B::Device,
    ) -> anyhow::Result<Self> {
        let embed_tensor = weights
            .embedding
            .ok_or_else(|| anyhow::anyhow!("embedding required but not loaded"))?;
        let embedding = TokenEmbedding::new(embed_tensor.clone());

        let final_norm = RmsNorm::new(
            weights
                .final_norm
                .ok_or_else(|| anyhow::anyhow!("final_norm required but not loaded"))?,
            cfg.rms_norm_eps,
        );

        // Output projection: tied or untied.
        //
        // Tied case: `lm_head.weight == embed_tokens.weight` in HF Llama; both
        // have shape `[vocab, hidden]`. Our `OutputProjection` expects weight
        // shape `[hidden, vocab]` for the `x @ W` matmul. So when tied,
        // transpose.
        //
        // Untied case: `weights.output_proj` is the lm_head tensor, which HF
        // serializes as `[vocab, hidden]`. Same transpose applies.
        let output_weight = match (cfg.tie_word_embeddings, weights.output_proj) {
            (true, _) => embed_tensor.swap_dims(0, 1),
            (false, Some(w)) => w.swap_dims(0, 1),
            (false, None) => anyhow::bail!("untied output projection missing"),
        };
        let output = OutputProjection::new(LinearWeight::Dense(output_weight));

        if weights.layers.len() != cfg.n_layers {
            anyhow::bail!(
                "expected {} layers, got {}",
                cfg.n_layers,
                weights.layers.len()
            );
        }

        let mut blocks = Vec::with_capacity(weights.layers.len());
        for layer in weights.layers {
            let attn_norm = RmsNorm::new(layer.attn_norm, cfg.rms_norm_eps);
            let ffn_norm = RmsNorm::new(layer.ffn_norm, cfg.rms_norm_eps);
            let rope = RotaryEmbedding::new(
                cfg.head_dim,
                cfg.max_position_embeddings,
                cfg.rope_theta,
                device,
            );
            // HF stores each projection as [out, in]; transpose to [in, out].
            let attn = Attention::new(
                LinearWeight::Dense(layer.q_proj.swap_dims(0, 1)),
                LinearWeight::Dense(layer.k_proj.swap_dims(0, 1)),
                LinearWeight::Dense(layer.v_proj.swap_dims(0, 1)),
                LinearWeight::Dense(layer.o_proj.swap_dims(0, 1)),
                rope,
                cfg.n_heads,
                cfg.n_kv_heads,
                cfg.head_dim,
            );
            let ffn = SwiGluFfn::new(
                LinearWeight::Dense(layer.ffn_gate.swap_dims(0, 1)),
                LinearWeight::Dense(layer.ffn_up.swap_dims(0, 1)),
                LinearWeight::Dense(layer.ffn_down.swap_dims(0, 1)),
            );
            blocks.push(DecoderBlock {
                attn_norm,
                attn,
                ffn_norm,
                ffn,
            });
        }

        Ok(Self {
            embedding,
            blocks,
            final_norm,
            output,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            max_seq: cfg.max_position_embeddings,
        })
    }

    /// Production single-stream forward. Caller owns the cache slots
    /// (one per block) and they persist across calls within one request.
    pub fn forward_with_caches(
        &self,
        token_ids: Tensor<B, 2, Int>,
        start_pos: usize,
        caches: &mut [KvCacheSlot<B>],
    ) -> Tensor<B, 3> {
        assert_eq!(caches.len(), self.blocks.len(), "one cache per block");
        let [_batch, seq] = token_ids.dims();
        let positions: Vec<i32> = (start_pos..start_pos + seq).map(|p| p as i32).collect();
        let mut x = self.embedding.forward(token_ids);
        for (block, cache) in self.blocks.iter().zip(caches.iter_mut()) {
            x = block.forward(x, &positions, cache);
        }
        let x = self.final_norm.forward(x);
        self.output.forward(x)
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
