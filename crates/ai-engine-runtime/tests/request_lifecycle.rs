use ai_engine_runtime::config::{ModelConfig, ModelFamily};
use ai_engine_runtime::request::RequestState;

type B = burn_ndarray::NdArray;

#[test]
fn request_state_constructs_one_cache_per_layer() {
    let cfg = ModelConfig {
        hidden_size: 32, intermediate_size: 64, n_layers: 4,
        n_heads: 4, n_kv_heads: 2, head_dim: 8,
        vocab_size: 100, max_position_embeddings: 64,
        rope_theta: 10000.0, rms_norm_eps: 1e-5,
        tie_word_embeddings: true, family: ModelFamily::Llama3,
    };
    let dev = Default::default();
    let req = RequestState::<B>::new(&cfg, 1, 32, &dev);
    assert_eq!(req.caches.len(), 4);
    assert_eq!(req.current_pos, 0);
}

#[test]
fn request_state_advance_updates_pos() {
    let cfg = ModelConfig {
        hidden_size: 32, intermediate_size: 64, n_layers: 2,
        n_heads: 4, n_kv_heads: 2, head_dim: 8,
        vocab_size: 100, max_position_embeddings: 64,
        rope_theta: 10000.0, rms_norm_eps: 1e-5,
        tie_word_embeddings: true, family: ModelFamily::Llama3,
    };
    let dev = Default::default();
    let mut req = RequestState::<B>::new(&cfg, 1, 32, &dev);
    req.advance(5);
    assert_eq!(req.current_pos, 5);
    req.advance(1);
    assert_eq!(req.current_pos, 6);
}
