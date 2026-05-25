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
    pub fn spawn(gguf_path: &Path, device: Device, cfg: EngineConfig) -> anyhow::Result<Arc<Self>> {
        let model = Transformer::load(gguf_path, device.clone(), cfg.max_seq)?;
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
                    if let Some(seq) = prefill(&model, &mut kv, &mut alloc, req) {
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
                    if let Some(seq) = prefill(&model, &mut kv, &mut alloc, req) {
                        running.push(seq);
                    }
                    continue;
                }
                None => return,
            }
        }

        let token_ids: Vec<u32> = running.iter().map(|s| s.next_token).collect();
        let positions: Vec<usize> = running.iter().map(|s| s.pos).collect();
        let seq_lens: Vec<usize> = running.iter().map(|s| s.pos).collect();
        // ensure_capacity must run before we borrow &BlockTable refs
        for s in running.iter_mut() {
            if alloc.ensure_capacity(&mut s.table, s.pos + 1).is_err() {
                let _ = s.tx.send(Err("KV cache exhausted".into()));
            }
        }
        let tables: Vec<&BlockTable> = running.iter().map(|s| &s.table).collect();
        let logits = match model.decode_step(&token_ids, &positions, &seq_lens, &mut kv, &alloc, &tables) {
            Ok(l) => l,
            Err(e) => {
                for s in &running { let _ = s.tx.send(Err(format!("forward: {e}"))); }
                running.clear();
                continue;
            }
        };

        let mut keep = Vec::with_capacity(running.len());
        for (i, mut s) in running.into_iter().enumerate() {
            let row: Vec<f32> = logits.narrow(0, i, 1).and_then(|t| t.squeeze(0)).and_then(|t| t.to_vec1()).unwrap();
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
    req: GenRequest,
) -> Option<Seq> {
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
    let row: Vec<f32> = logits.squeeze(0).and_then(|t| t.to_vec1()).ok()?;
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
