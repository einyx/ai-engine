use ai_engine_runtime::arch::attention::Attention;
use ai_engine_runtime::kv_cache::KvCacheSlot;
use burn::tensor::Tensor;

type B = burn_ndarray::NdArray;

#[test]
fn attention_forward_shape_correct_for_gqa() {
    let dev = Default::default();
    // GQA: 4 query heads, 2 KV heads, head_dim 8 -> hidden = 32.
    let attn = Attention::<B>::with_random_weights(
        /*hidden=*/ 32,
        /*n_heads=*/ 4,
        /*n_kv_heads=*/ 2,
        /*head_dim=*/ 8,
        /*max_seq=*/ 16,
        /*rope_theta=*/ 10000.0,
        &dev,
    );
    let mut cache = KvCacheSlot::<B>::new(1, 2, 16, 8, &dev);
    let x = Tensor::<B, 3>::ones([1, 3, 32], &dev);
    let positions = vec![0_i32, 1, 2];
    let out = attn.forward(x, &positions, &mut cache);
    assert_eq!(out.dims(), [1, 3, 32]);
    assert_eq!(cache.current_len(), 3);
}

#[test]
fn attention_second_call_uses_cached_keys() {
    let dev = Default::default();
    let attn = Attention::<B>::with_random_weights(32, 4, 2, 8, 16, 10000.0, &dev);
    let mut cache = KvCacheSlot::<B>::new(1, 2, 16, 8, &dev);
    let first = Tensor::<B, 3>::ones([1, 3, 32], &dev);
    attn.forward(first, &[0, 1, 2], &mut cache);
    assert_eq!(cache.current_len(), 3);
    let next = Tensor::<B, 3>::ones([1, 1, 32], &dev);
    let out = attn.forward(next, &[3], &mut cache);
    assert_eq!(out.dims(), [1, 1, 32]);
    assert_eq!(cache.current_len(), 4);
}
