use ai_engine_runtime::arch::model::Model;
use ai_engine_runtime::config::{ModelConfig, ModelFamily};
use burn::tensor::{Int, Tensor, TensorData};

type B = burn_ndarray::NdArray;

#[test]
fn model_forward_produces_correct_logit_shape() {
    let dev = Default::default();
    let cfg = ModelConfig {
        hidden_size: 32,
        intermediate_size: 64,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 8,
        vocab_size: 100,
        max_position_embeddings: 32,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-5,
        tie_word_embeddings: true,
        family: ModelFamily::Llama3,
    };
    let model = Model::<B>::with_random_weights(&cfg, &dev);
    let token_ids = Tensor::<B, 2, Int>::from_data(
        TensorData::new(vec![1_i32, 2, 3, 4, 5], [1, 5]),
        &dev,
    );
    let logits = model.forward(token_ids, /*start_pos=*/ 0);
    assert_eq!(logits.dims(), [1, 5, 100]);
}
