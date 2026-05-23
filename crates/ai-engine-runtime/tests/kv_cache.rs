use ai_engine_runtime::kv_cache::KvCacheSlot;
use burn::tensor::Tensor;

type B = burn_ndarray::NdArray;

#[test]
fn kv_cache_appends_tokens_and_reads_all() {
    let dev = Default::default();
    let mut slot = KvCacheSlot::<B>::new(1, 2, 16, 64, &dev);
    assert_eq!(slot.current_len(), 0);
    let k_new = Tensor::<B, 4>::ones([1, 2, 3, 64], &dev);
    let v_new = Tensor::<B, 4>::ones([1, 2, 3, 64], &dev);
    slot.append(k_new, v_new);
    assert_eq!(slot.current_len(), 3);
    let (k_all, v_all) = slot.read();
    assert_eq!(k_all.dims(), [1, 2, 3, 64]);
    assert_eq!(v_all.dims(), [1, 2, 3, 64]);
}

#[test]
fn kv_cache_appends_incrementally_for_autoregressive_gen() {
    let dev = Default::default();
    let mut slot = KvCacheSlot::<B>::new(1, 2, 16, 64, &dev);
    let prefill = Tensor::<B, 4>::ones([1, 2, 5, 64], &dev);
    slot.append(prefill.clone(), prefill);
    for _ in 0..3 {
        let one = Tensor::<B, 4>::ones([1, 2, 1, 64], &dev);
        slot.append(one.clone(), one);
    }
    assert_eq!(slot.current_len(), 8);
}
