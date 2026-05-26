use ai_engine_runtime::sample::{sample, SamplingConfig};

#[test]
fn greedy_picks_argmax() {
    let logits = vec![0.1, 5.0, 2.0, -1.0];
    let cfg = SamplingConfig { temperature: 0.0, top_p: None, top_k: None, seed: 42 };
    assert_eq!(sample(&logits, &cfg), 1);
}

#[test]
fn temperature_zero_is_greedy() {
    let logits = vec![1.0, 5.0, 2.0];
    let cfg = SamplingConfig { temperature: 0.0, top_p: None, top_k: None, seed: 0 };
    for _ in 0..20 {
        assert_eq!(sample(&logits, &cfg), 1);
    }
}

#[test]
fn top_k_one_picks_largest() {
    let logits = vec![1.0, 1.0, 1.0, 1.0, 100.0];
    let cfg = SamplingConfig { temperature: 1.0, top_p: None, top_k: Some(1), seed: 0 };
    for _ in 0..20 {
        assert_eq!(sample(&logits, &cfg), 4);
    }
}

#[test]
fn top_p_nucleus_concentrates_mass() {
    let logits = vec![1.0, 1.0, 100.0, 1.0];
    let cfg = SamplingConfig { temperature: 1.0, top_p: Some(0.5), top_k: None, seed: 0 };
    for _ in 0..20 {
        assert_eq!(sample(&logits, &cfg), 2);
    }
}
