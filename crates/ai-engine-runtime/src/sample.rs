use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub seed: u64,
}

pub fn sample(logits: &[f32], cfg: &SamplingConfig) -> u32 {
    if cfg.temperature == 0.0 || logits.len() <= 1 {
        return argmax(logits);
    }
    let mut probs: Vec<(usize, f32)> = logits.iter()
        .map(|x| x / cfg.temperature)
        .enumerate().collect();
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if let Some(k) = cfg.top_k {
        probs.truncate(k);
    }
    let max_x = probs[0].1;
    let mut sum = 0.0;
    for p in probs.iter_mut() {
        p.1 = (p.1 - max_x).exp();
        sum += p.1;
    }
    for p in probs.iter_mut() { p.1 /= sum; }
    if let Some(p_threshold) = cfg.top_p {
        let mut cum = 0.0;
        let mut cutoff = probs.len();
        for (i, p) in probs.iter().enumerate() {
            cum += p.1;
            if cum >= p_threshold { cutoff = i + 1; break; }
        }
        probs.truncate(cutoff);
        let s: f32 = probs.iter().map(|p| p.1).sum();
        for p in probs.iter_mut() { p.1 /= s; }
    }
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let r: f32 = rng.gen();
    let mut acc = 0.0;
    for (idx, prob) in &probs {
        acc += prob;
        if r <= acc { return *idx as u32; }
    }
    probs.last().unwrap().0 as u32
}

fn argmax(logits: &[f32]) -> u32 {
    logits.iter().enumerate().fold((0_usize, f32::NEG_INFINITY), |(bi, bv), (i, v)| {
        if *v > bv { (i, *v) } else { (bi, bv) }
    }).0 as u32
}
