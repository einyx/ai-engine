//! Continuous-batching scheduler.

use std::path::Path;
use std::sync::Arc;
use candle_core::Device;
use tokio::sync::mpsc;

use ai_engine_runtime::sample::{self, SamplingConfig};
use crate::paged::attention::KvPool;
use crate::paged::block_table::{BlockAllocator, BlockTable};
use crate::paged::transformer::Transformer;

pub struct GenRequest {
    pub prompt_ids: Vec<u32>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub tx: mpsc::UnboundedSender<Result<u32, String>>,
}

struct Seq {
    table: BlockTable,
    next_token: u32,
    pos: usize,
    produced: usize,
    max_tokens: usize,
    temperature: f32,
    tx: mpsc::UnboundedSender<Result<u32, String>>,
}

pub struct EngineConfig {
    pub max_num_seqs: usize,
    pub block_size: usize,
    pub kv_cache_blocks: usize,
    pub max_seq: usize,
    pub eos_token_id: u32,
}

pub struct Engine {
    submit_tx: mpsc::UnboundedSender<GenRequest>,
}

impl Engine {
    pub fn spawn(gguf_path: &Path, device: Device, mut cfg: EngineConfig) -> anyhow::Result<Arc<Self>> {
        let model = Transformer::load(gguf_path, device.clone(), cfg.max_seq)?;
        // Override max_seq from the model's actual context_length (GGUF metadata),
        // so RoPE table size and admission guard are consistent.
        cfg.max_seq = model.cfg.context_length;
        let (submit_tx, submit_rx) = mpsc::unbounded_channel::<GenRequest>();
        std::thread::spawn(move || run_loop(model, device, cfg, submit_rx));
        Ok(Arc::new(Self { submit_tx }))
    }

    pub fn submit(&self, req: GenRequest) {
        let _ = self.submit_tx.send(req);
    }
}

fn sample_cfg(temperature: f32) -> SamplingConfig {
    SamplingConfig { temperature, top_p: None, top_k: None, seed: 42 }
}

fn run_loop(
    model: Transformer,
    device: Device,
    cfg: EngineConfig,
    mut submit_rx: mpsc::UnboundedReceiver<GenRequest>,
) {
    let n_layers = model.cfg.block_count;
    let (n_kv, hd) = (model.cfg.head_count_kv, model.cfg.head_dim);
    let mut kv: Vec<KvPool> = (0..n_layers)
        .map(|_| KvPool::new(cfg.kv_cache_blocks, cfg.block_size, n_kv, hd, &device).unwrap())
        .collect();
    let mut alloc = BlockAllocator::new(cfg.kv_cache_blocks, cfg.block_size);
    let mut running: Vec<Seq> = Vec::new();

    loop {
        while running.len() < cfg.max_num_seqs {
            match submit_rx.try_recv() {
                Ok(req) => {
                    if let Some(seq) = prefill(&model, &mut kv, &mut alloc, &cfg, req) {
                        running.push(seq);
                    }
                }
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    if running.is_empty() { return; } else { break; }
                }
            }
        }
        if running.is_empty() {
            match submit_rx.blocking_recv() {
                Some(req) => {
                    if let Some(seq) = prefill(&model, &mut kv, &mut alloc, &cfg, req) {
                        running.push(seq);
                    }
                    continue;
                }
                None => return,
            }
        }

        // HIGH 1: OOM eviction — ensure_capacity before building tables.
        // Sequences that fail get an error, their blocks are released, and they
        // are evicted before we call decode_step (avoiding OOB panic in alloc.locate).
        {
            let mut evicted = false;
            let mut survivors: Vec<Seq> = Vec::with_capacity(running.len());
            for mut s in running.drain(..) {
                // HIGH 3 decode-side guard: evict if already at or past max context.
                if s.pos + 1 > cfg.max_seq {
                    let _ = s.tx.send(Err(format!("sequence exceeded max context ({} > {})", s.pos + 1, cfg.max_seq)));
                    alloc.release(&mut s.table);
                    evicted = true;
                    continue;
                }
                if alloc.ensure_capacity(&mut s.table, s.pos + 1).is_err() {
                    let _ = s.tx.send(Err("KV cache exhausted".into()));
                    alloc.release(&mut s.table);
                    evicted = true;
                    continue;
                }
                survivors.push(s);
            }
            running = survivors;
            if evicted && running.is_empty() {
                continue;
            }
        }

        let token_ids: Vec<u32> = running.iter().map(|s| s.next_token).collect();
        let positions: Vec<usize> = running.iter().map(|s| s.pos).collect();
        let seq_lens: Vec<usize> = running.iter().map(|s| s.pos).collect();
        let tables: Vec<&BlockTable> = running.iter().map(|s| &s.table).collect();
        let logits = match model.decode_step(&token_ids, &positions, &seq_lens, &mut kv, &alloc, &tables) {
            Ok(l) => l,
            Err(e) => {
                for s in &running { let _ = s.tx.send(Err(format!("forward: {e}"))); }
                for s in running.iter_mut() { alloc.release(&mut s.table); }
                running.clear();
                continue;
            }
        };

        // HIGH 2: replace .unwrap() with proper error handling for logit row extraction.
        let mut logit_rows: Vec<Vec<f32>> = Vec::with_capacity(running.len());
        let mut row_err: Option<String> = None;
        for i in 0..running.len() {
            match logits.narrow(0, i, 1).and_then(|t| t.squeeze(0)).and_then(|t| t.to_vec1::<f32>()) {
                Ok(row) => logit_rows.push(row),
                Err(e) => { row_err = Some(format!("logit extract: {e}")); break; }
            }
        }
        if let Some(e) = row_err {
            for s in &running { let _ = s.tx.send(Err(e.clone())); }
            for s in running.iter_mut() { alloc.release(&mut s.table); }
            running.clear();
            continue;
        }

        let mut keep = Vec::with_capacity(running.len());
        for (mut s, row) in running.into_iter().zip(logit_rows) {
            let next = sample::sample(&row, &sample_cfg(s.temperature));
            s.pos += 1;
            s.produced += 1;
            let eos = next == cfg.eos_token_id;
            let at_limit = s.produced >= s.max_tokens;
            // Mirror CandleModel: push token unless it is EOS (EOS triggers stop before push).
            if !eos {
                let _ = s.tx.send(Ok(next));
            }
            if eos || at_limit {
                alloc.release(&mut s.table);
            } else {
                s.next_token = next;
                keep.push(s);
            }
        }
        running = keep;
    }
}

fn prefill(
    model: &Transformer,
    kv: &mut [KvPool],
    alloc: &mut BlockAllocator,
    cfg: &EngineConfig,
    req: GenRequest,
) -> Option<Seq> {
    // MEDIUM 5: zero max_tokens → produce nothing.
    if req.max_tokens == 0 {
        return None;
    }
    // HIGH 3: reject prompts that exceed the model's context window.
    if req.prompt_ids.len() > cfg.max_seq {
        let _ = req.tx.send(Err(format!(
            "prompt exceeds max context ({} > {})",
            req.prompt_ids.len(),
            cfg.max_seq
        )));
        return None;
    }
    let mut table = BlockTable::default();
    if alloc.ensure_capacity(&mut table, req.prompt_ids.len()).is_err() {
        let _ = req.tx.send(Err("KV cache exhausted at admission".into()));
        return None;
    }
    // Batch prefill: process the entire prompt as a causal sequence in one shot,
    // matching candle_transformers' batch forward exactly for numerical parity.
    let logits = match model.prefill_seq(&req.prompt_ids, &table, kv, alloc) {
        Ok(l) => l,
        Err(e) => {
            let _ = req.tx.send(Err(format!("prefill: {e}")));
            alloc.release(&mut table);
            return None;
        }
    };
    // MEDIUM 4: on logit extraction failure, release blocks before returning None.
    let row: Vec<f32> = match logits.squeeze(0).and_then(|t| t.to_vec1::<f32>()) {
        Ok(r) => r,
        Err(e) => {
            let _ = req.tx.send(Err(format!("prefill logit extract: {e}")));
            alloc.release(&mut table);
            return None;
        }
    };
    let next = sample::sample(&row, &sample_cfg(req.temperature));
    let _ = req.tx.send(Ok(next));
    Some(Seq {
        table,
        next_token: next,
        pos: req.prompt_ids.len(),
        produced: 1,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        tx: req.tx,
    })
}
