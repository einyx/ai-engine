# Plan 5 — v0.3.0-alpha: Q8 weight quantization

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Q8 (8-bit symmetric per-tensor) weight quantization to `ai-engine-runtime` so realistic-size Llama-family checkpoints take ~4× less RAM/VRAM than bf16 while producing logits matching the bf16 reference within bf16's own noise floor.

**Architecture:** A new `QuantizedTensor<B>` type that wraps a packed `Tensor<B, 2, Int>` of i8 values + a per-tensor `f32` scale. Linear layers (q/k/v/o projections and SwiGLU's gate/up/down) gain a quantized matmul path: `dequantize(weight) → matmul(activations, weight_f32)`. The dequantize step materializes a temporary bf16/f32 tensor per call — slower than a fused GEMM but ~4× memory savings at rest. The safetensors loader detects an `int8` dtype on a weight tensor + a companion `<name>.scale` f32 tensor and constructs a `QuantizedTensor` instead of a regular `Tensor`. A new Python script generates a Q8 version of the toy-llama-3 fixture by quantizing each Linear weight with `scale = max(|w|) / 127`, packing as int8.

**Tech Stack:** No new external crates. Uses the existing burn / safetensors / bytemuck / half stack. Python fixture generator reuses the existing `torch / transformers / safetensors` venv from Plan 1.

**Scope rule:** Plan 5 ships **Q8 only** (per-tensor symmetric, no zero-point). Q4 weights (per-group blockwise, packed nibbles, optional zero-points, AWQ/GPTQ/GGUF format variants) are a substantially harder follow-up — separate Plan 6. mixed-precision activations are also out of scope; activations stay f32.

**Baseline:** Branch `main` at `v0.2.1`. 154 tests + 4 ignored (2 multiproc smokes, 1 backend parity, 1 load smoke). Clippy clean.

---

## File structure

```
crates/ai-engine-runtime/
├── src/
│   ├── lib.rs                       # MODIFY: re-export Quant types
│   ├── quant.rs                     # NEW: QuantizedTensor<B>, dequantize, packed loader
│   ├── arch/
│   │   ├── attention.rs             # MODIFY: q/k/v/o projections accept LinearWeight enum
│   │   ├── ffn.rs                   # MODIFY: gate/up/down projections accept LinearWeight enum
│   │   ├── embedding.rs             # MODIFY: OutputProjection accepts LinearWeight enum
│   │   └── linear.rs                # NEW: LinearWeight<B> enum + matmul()
│   ├── loader.rs                    # MODIFY: detect int8 weights + scale companion
│   └── name_map.rs                  # MODIFY: TensorId::*Scale variants for companion tensors
├── fixtures/
│   └── toy-llama-3-q8/              # NEW: quantized version of toy-llama-3
│       ├── config.json
│       ├── model.safetensors        # int8 weights + f32 scales
│       ├── tokenizer.json
│       ├── reference_logits.bin     # f32 logits from the BF16 model (same as toy-llama-3's)
│       ├── reference_prompt.txt
│       └── README.md
├── scripts/
│   └── generate_q8_fixture.py       # NEW: takes toy-llama-3 bf16 -> outputs toy-llama-3-q8
└── tests/
    └── quant_reference_logits.rs    # NEW: Q8 forward matches bf16 reference within 1e-2 (looser tolerance, Q8 quantization is lossy)
```

File responsibility:
- `quant.rs` owns the `QuantizedTensor<B>` type + a `dequantize()` method returning `Tensor<B, 2>`. Pure ML primitive, no loader concerns.
- `linear.rs` is the new unified API. `LinearWeight<B>::{Dense(Tensor<B, 2>), Quantized(QuantizedTensor<B>)}` + a `matmul(&self, x: Tensor<B, 3>) -> Tensor<B, 3>` method. Every Linear use-site (attention, FFN, embedding's output projection) calls through it.
- `loader.rs` learns to inspect tensor dtypes during load: if int8, look for `<weight_name>.scale` companion and build a Quantized variant.
- `name_map.rs` gets `TensorId::LayerQProjScale(usize)` etc. for the scale companion lookups.

---

## Important pre-flight notes

- **Per-tensor symmetric Q8** is the simplest quantization: one `f32` scale per weight matrix, no zero-point, signed int8 values. `weight_f32 = (i8 as f32) * scale`. The forward pass dequantizes the FULL weight matrix on each call, then runs the existing matmul. Memory at rest is `int8` (1 byte per param); compute peak briefly returns to f32 during the layer's forward.
- **The Q8 fixture's reference_logits.bin is identical to the bf16 fixture's.** We quantize the same model weights, run a forward pass that ALSO uses dequantized f32 math, and the output must match the bf16 reference within the bf16 noise floor + per-tensor Q8 quantization error. Tolerance: 1e-2 (vs 1e-3 for bf16-vs-bf16). For most Linear layers in a randomly-initialized toy, Q8 introduces ~1e-3 per-op error, accumulating to a few times 1e-3 over 4 layers + lm_head. Worst-case 1e-2 is conservative.
- **`burn::tensor::Int` is i64 by default.** We need i8. burn has a `Tensor<B, N, Int>` but the underlying integer type depends on the backend; we store the raw int8 bytes in our own `Vec<i8>` and only construct an f32 Tensor at dequantize time. This avoids fighting burn's int type system.
- **Loader companion lookup**: HF transformers stores `bitsandbytes`-style int8 weights with separate `weight.SCB` (scale per channel) tensors. We use a simpler convention: `<weight>.scale` is a single f32 (per-tensor). Our Python fixture generator writes this convention; real HF int8 dumps will need a translation layer (out of Plan 5 scope — note as limitation).
- **Existing tests must continue to pass.** All bf16 weight paths stay working unchanged. The new code is purely additive.

---

### Task 1: `QuantizedTensor` primitive + dequantize

**Files:**
- Create: `crates/ai-engine-runtime/src/quant.rs`
- Create: `crates/ai-engine-runtime/tests/quant.rs`
- Modify: `crates/ai-engine-runtime/src/lib.rs`

- [ ] **Step 1: Failing test**

`crates/ai-engine-runtime/tests/quant.rs`:

```rust
use ai_engine_runtime::quant::QuantizedTensor;
use burn::tensor::{Tensor, TensorData};

type B = burn_ndarray::NdArray;

#[test]
fn quantize_then_dequantize_recovers_original_within_q8_noise() {
    let dev = Default::default();
    // Build a known dense matrix.
    let original_f32 = vec![
        1.0_f32, -0.5, 0.25, -0.125,
        0.6, -0.6, 0.1, -0.1,
    ];
    let original = Tensor::<B, 2>::from_data(
        TensorData::new(original_f32.clone(), [2, 4]),
        &dev,
    );

    let q = QuantizedTensor::<B>::quantize_from(original.clone());
    let recovered = q.dequantize();

    let original_v: Vec<f32> = original.into_data().to_vec().unwrap();
    let recovered_v: Vec<f32> = recovered.into_data().to_vec().unwrap();

    let max_err = original_v.iter().zip(recovered_v.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    // Q8 with per-tensor scale, signed range -128..127, max abs of inputs = 1.0
    // -> scale = 1.0 / 127 ~ 0.00787. Worst-case quantization error per value
    // is about scale/2 ~ 0.004.
    assert!(max_err < 0.005, "quantization error {max_err} exceeded Q8 bound");
}

#[test]
fn quantized_tensor_stores_int8_bytes_not_f32() {
    let dev = Default::default();
    let original = Tensor::<B, 2>::from_data(
        TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0], [2, 2]),
        &dev,
    );
    let q = QuantizedTensor::<B>::quantize_from(original);
    // Storage: 2x2 = 4 i8 values = 4 bytes, plus an f32 scale.
    assert_eq!(q.packed.len(), 4);
    assert!(q.scale > 0.0);
    assert_eq!(q.shape(), [2, 2]);
}

#[test]
fn quantized_tensor_from_raw_components_roundtrips() {
    let dev = Default::default();
    let packed = vec![127_i8, -127, 0, 64];
    let scale = 0.5_f32;
    let q = QuantizedTensor::<B>::from_packed(packed.clone(), scale, [2, 2], &dev);
    assert_eq!(q.shape(), [2, 2]);
    let d = q.dequantize();
    let v: Vec<f32> = d.into_data().to_vec().unwrap();
    // 127 * 0.5 = 63.5; -127 * 0.5 = -63.5; 0 * 0.5 = 0; 64 * 0.5 = 32
    assert!((v[0] - 63.5).abs() < 1e-4);
    assert!((v[1] - -63.5).abs() < 1e-4);
    assert!(v[2].abs() < 1e-4);
    assert!((v[3] - 32.0).abs() < 1e-4);
}
```

- [ ] **Step 2: Confirm fails**

```bash
cd /home/alessio/aip/airproxy
cargo test -p ai-engine-runtime --test quant 2>&1 | tail -10
# Expected: compile error — QuantizedTensor doesn't exist.
```

- [ ] **Step 3: Implement `quant.rs`**

```rust
use burn::tensor::{backend::Backend, Tensor, TensorData};
use std::marker::PhantomData;

/// Per-tensor symmetric Q8 quantization.
///
/// Storage:
///   - `packed`: raw i8 values, length = product of `shape`.
///   - `scale`: single f32 per tensor.
///   - `shape`: original 2-D shape.
///
/// Reconstruction: `weight_f32[i] = packed[i] as f32 * scale`.
///
/// Quantization: `packed[i] = clamp(round(weight_f32[i] / scale), -127, 127)`
/// where `scale = max(|weight_f32|) / 127`. The clamp prevents -128 (asymmetric
/// edge case) so the negation is always representable.
pub struct QuantizedTensor<B: Backend> {
    pub packed: Vec<i8>,
    pub scale: f32,
    shape: [usize; 2],
    _marker: PhantomData<B>,
    device: B::Device,
}

impl<B: Backend> QuantizedTensor<B> {
    /// Quantize an f32 tensor to Q8 with a per-tensor scale.
    pub fn quantize_from(t: Tensor<B, 2>) -> Self {
        let shape = t.dims();
        let device = t.device();
        let values: Vec<f32> = t.into_data().to_vec()
            .expect("to_vec f32 from Tensor<B, 2>");
        let max_abs = values.iter().copied().fold(0.0_f32, |acc, x| acc.max(x.abs()));
        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
        let packed: Vec<i8> = values.iter()
            .map(|&v| ((v / scale).round().clamp(-127.0, 127.0)) as i8)
            .collect();
        Self { packed, scale, shape, _marker: PhantomData, device }
    }

    /// Construct from raw packed bytes + scale (used by the loader).
    pub fn from_packed(packed: Vec<i8>, scale: f32, shape: [usize; 2], device: &B::Device) -> Self {
        assert_eq!(packed.len(), shape[0] * shape[1], "packed length must match shape product");
        Self { packed, scale, shape, _marker: PhantomData, device: device.clone() }
    }

    pub fn shape(&self) -> [usize; 2] { self.shape }

    /// Dequantize to a regular f32 Tensor<B, 2>. Allocates a new f32 buffer.
    pub fn dequantize(&self) -> Tensor<B, 2> {
        let f32_values: Vec<f32> = self.packed.iter()
            .map(|&q| (q as f32) * self.scale)
            .collect();
        Tensor::<B, 2>::from_data(TensorData::new(f32_values, self.shape), &self.device)
    }
}
```

- [ ] **Step 4: Wire module**

`crates/ai-engine-runtime/src/lib.rs` (append):

```rust
pub mod quant;
pub use quant::QuantizedTensor;
```

- [ ] **Step 5: Verify + commit**

```bash
cargo test -p ai-engine-runtime --test quant
cargo clippy --workspace --all-targets -- -D warnings
git add -A
git commit -m "feat(runtime): QuantizedTensor<B> with per-tensor symmetric Q8 quantization"
```

NO Co-Authored-By footer (global preference).

---

### Task 2: `LinearWeight` enum + matmul dispatch

**Files:**
- Create: `crates/ai-engine-runtime/src/arch/linear.rs`
- Create: `crates/ai-engine-runtime/tests/linear_weight.rs`
- Modify: `crates/ai-engine-runtime/src/arch/mod.rs`

The unified API every Linear in our forward pass goes through.

- [ ] **Step 1: Failing test**

`crates/ai-engine-runtime/tests/linear_weight.rs`:

```rust
use ai_engine_runtime::arch::linear::LinearWeight;
use ai_engine_runtime::quant::QuantizedTensor;
use burn::tensor::{Tensor, TensorData};

type B = burn_ndarray::NdArray;

#[test]
fn dense_linear_matmul_matches_raw_matmul() {
    let dev = Default::default();
    let w = Tensor::<B, 2>::from_data(
        TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]),
        &dev,
    );
    let x = Tensor::<B, 3>::from_data(
        TensorData::new(vec![1.0_f32, 1.0], [1, 1, 2]),
        &dev,
    );
    let lw = LinearWeight::Dense(w.clone());
    let out_via_lw = lw.matmul(x.clone());
    let out_direct = x.matmul(w.unsqueeze());
    let a: Vec<f32> = out_via_lw.into_data().to_vec().unwrap();
    let b: Vec<f32> = out_direct.into_data().to_vec().unwrap();
    assert_eq!(a, b);
}

#[test]
fn quantized_linear_matmul_approximates_dense() {
    let dev = Default::default();
    // Random-ish weight; rounding under Q8 introduces small error.
    let raw: Vec<f32> = (0..6).map(|i| (i as f32 - 3.0) * 0.1).collect();
    let w = Tensor::<B, 2>::from_data(TensorData::new(raw.clone(), [2, 3]), &dev);
    let x = Tensor::<B, 3>::from_data(
        TensorData::new(vec![1.0_f32, -1.0], [1, 1, 2]),
        &dev,
    );

    let dense = LinearWeight::Dense(w.clone());
    let qw = QuantizedTensor::<B>::quantize_from(w);
    let quant = LinearWeight::Quantized(qw);

    let out_dense: Vec<f32> = dense.matmul(x.clone()).into_data().to_vec().unwrap();
    let out_quant: Vec<f32> = quant.matmul(x).into_data().to_vec().unwrap();

    let max_diff = out_dense.iter().zip(out_quant.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    // 2 multiply-adds per output element, each with up to Q8 noise.
    // Empirically well under 1e-2 for tensors of this size.
    assert!(max_diff < 1e-2, "quantized matmul diverged from dense by {max_diff}");
}
```

- [ ] **Step 2: Implement `linear.rs`**

```rust
use crate::quant::QuantizedTensor;
use burn::tensor::{backend::Backend, Tensor};

/// A Linear's weight matrix — either dense or Q8-quantized.
///
/// Both forms produce the same `[in, out]`-shaped weight from the caller's
/// perspective. `matmul(x: [batch, seq, in]) -> [batch, seq, out]` handles
/// the dispatch.
pub enum LinearWeight<B: Backend> {
    Dense(Tensor<B, 2>),
    Quantized(QuantizedTensor<B>),
}

impl<B: Backend> LinearWeight<B> {
    pub fn shape(&self) -> [usize; 2] {
        match self {
            Self::Dense(t) => t.dims(),
            Self::Quantized(q) => q.shape(),
        }
    }

    /// `x: [batch, seq, in]` -> `[batch, seq, out]`. For quantized weights,
    /// dequantizes the weight matrix once before the matmul.
    pub fn matmul(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            Self::Dense(w) => x.matmul(w.clone().unsqueeze()),
            Self::Quantized(q) => x.matmul(q.dequantize().unsqueeze()),
        }
    }
}
```

- [ ] **Step 3: Wire module**

`crates/ai-engine-runtime/src/arch/mod.rs`:
```rust
pub mod attention;
pub mod block;
pub mod embedding;
pub mod ffn;
pub mod linear;
pub mod model;
pub mod rmsnorm;
pub mod rope;
```

- [ ] **Step 4: Verify + commit**

```bash
cargo test -p ai-engine-runtime --test linear_weight
cargo clippy --workspace --all-targets -- -D warnings
git add -A
git commit -m "feat(runtime): LinearWeight<B> enum dispatching dense vs quantized matmul"
```

---

### Task 3: Migrate attention / FFN / embedding to `LinearWeight`

**Files:**
- Modify: `crates/ai-engine-runtime/src/arch/attention.rs`
- Modify: `crates/ai-engine-runtime/src/arch/ffn.rs`
- Modify: `crates/ai-engine-runtime/src/arch/embedding.rs`
- Modify: `crates/ai-engine-runtime/src/arch/model.rs` (constructor wiring)
- Modify: `crates/ai-engine-runtime/src/loader.rs` (`LayerWeights` types — see below)

The existing primitives use `Tensor<B, 2>` directly for weights. Refactor each to take `LinearWeight<B>` instead. All existing bf16-only paths must continue producing identical results.

- [ ] **Step 1: Refactor `Attention`**

In `arch/attention.rs`, change the struct's weight field types:

```rust
use crate::arch::linear::LinearWeight;

pub struct Attention<B: Backend> {
    pub q_proj: LinearWeight<B>,
    pub k_proj: LinearWeight<B>,
    pub v_proj: LinearWeight<B>,
    pub o_proj: LinearWeight<B>,
    pub rope: RotaryEmbedding<B>,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub scale: f32,
}

impl<B: Backend> Attention<B> {
    pub fn new(
        q_proj: LinearWeight<B>,
        k_proj: LinearWeight<B>,
        v_proj: LinearWeight<B>,
        o_proj: LinearWeight<B>,
        rope: RotaryEmbedding<B>,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        assert!(n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads");
        let scale = 1.0 / (head_dim as f32).sqrt();
        Self { q_proj, k_proj, v_proj, o_proj, rope, n_heads, n_kv_heads, head_dim, scale }
    }

    pub fn with_random_weights(...) -> Self {
        // ... existing random tensor construction ...
        // Wrap each weight in LinearWeight::Dense(...).
        Self::new(
            LinearWeight::Dense(q_proj),
            LinearWeight::Dense(k_proj),
            LinearWeight::Dense(v_proj),
            LinearWeight::Dense(o_proj),
            rope, n_heads, n_kv_heads, head_dim,
        )
    }
}
```

In the `forward` method, replace `self.q_proj.clone().unsqueeze()` style calls (which assume `Tensor<B, 2>`) with `self.q_proj.matmul(x.clone())` and adapt. The shape-equivalence is the same; only the call signature changes.

- [ ] **Step 2: Refactor `SwiGluFfn`**

Same treatment in `arch/ffn.rs`:

```rust
pub struct SwiGluFfn<B: Backend> {
    pub gate_proj: LinearWeight<B>,
    pub up_proj: LinearWeight<B>,
    pub down_proj: LinearWeight<B>,
}

impl<B: Backend> SwiGluFfn<B> {
    pub fn new(gate_proj: LinearWeight<B>, up_proj: LinearWeight<B>, down_proj: LinearWeight<B>) -> Self {
        Self { gate_proj, up_proj, down_proj }
    }

    pub fn with_random_weights(hidden: usize, inter: usize, device: &B::Device) -> Self {
        Self {
            gate_proj: LinearWeight::Dense(Tensor::<B, 2>::random([hidden, inter], Distribution::Default, device)),
            up_proj:   LinearWeight::Dense(Tensor::<B, 2>::random([hidden, inter], Distribution::Default, device)),
            down_proj: LinearWeight::Dense(Tensor::<B, 2>::random([inter, hidden], Distribution::Default, device)),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = self.gate_proj.matmul(x.clone());
        let up   = self.up_proj.matmul(x);
        self.down_proj.matmul(silu(gate).mul(up))
    }
}
```

- [ ] **Step 3: Refactor `OutputProjection`**

In `arch/embedding.rs`:

```rust
pub struct OutputProjection<B: Backend> {
    pub weight: LinearWeight<B>,
}

impl<B: Backend> OutputProjection<B> {
    pub fn new(weight: LinearWeight<B>) -> Self { Self { weight } }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.weight.matmul(x)
    }
}
```

`TokenEmbedding` does NOT change — it uses `Tensor::select` for the lookup, not matmul. Embeddings can be quantized too (rare), but Plan 5 leaves them as f32; the safetensors loader keeps them as bf16/f32.

- [ ] **Step 4: Update `Model::from_loaded`**

In `arch/model.rs`, all sites that construct `Attention::new(layer.q_proj.swap_dims(0, 1), ...)` now wrap with `LinearWeight::Dense(...)`:

```rust
let attn = Attention::new(
    LinearWeight::Dense(layer.q_proj.swap_dims(0, 1)),
    LinearWeight::Dense(layer.k_proj.swap_dims(0, 1)),
    LinearWeight::Dense(layer.v_proj.swap_dims(0, 1)),
    LinearWeight::Dense(layer.o_proj.swap_dims(0, 1)),
    rope, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim,
);
let ffn = SwiGluFfn::new(
    LinearWeight::Dense(layer.ffn_gate.swap_dims(0, 1)),
    LinearWeight::Dense(layer.ffn_up.swap_dims(0, 1)),
    LinearWeight::Dense(layer.ffn_down.swap_dims(0, 1)),
);
let output = OutputProjection::new(
    LinearWeight::Dense(output_weight),
);
```

(Note: `Model::with_random_weights` calls per-primitive `with_random_weights` constructors which already wrap in `LinearWeight::Dense`, so no change needed there.)

- [ ] **Step 5: Same for `ai-engine-cluster::session::build_leader_model` + `ai-engine-cluster::worker::run_worker_full`**

The leader and worker both build `DecoderBlock`s manually with `Attention::new(layer.q_proj.swap_dims(0, 1), ...)`. Wrap each in `LinearWeight::Dense(...)`:

In `crates/ai-engine-cluster/src/session.rs`:
```rust
use ai_engine_runtime::arch::linear::LinearWeight;
let attn = Attention::new(
    LinearWeight::Dense(layer.q_proj.swap_dims(0, 1)),
    LinearWeight::Dense(layer.k_proj.swap_dims(0, 1)),
    LinearWeight::Dense(layer.v_proj.swap_dims(0, 1)),
    LinearWeight::Dense(layer.o_proj.swap_dims(0, 1)),
    rope, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim,
);
let ffn = SwiGluFfn::new(
    LinearWeight::Dense(layer.ffn_gate.swap_dims(0, 1)),
    LinearWeight::Dense(layer.ffn_up.swap_dims(0, 1)),
    LinearWeight::Dense(layer.ffn_down.swap_dims(0, 1)),
);
// final_norm is RmsNorm — unchanged.
// output is OutputProjection — wrap embedding.weight.swap_dims in LinearWeight::Dense.
let output = OutputProjection::new(
    LinearWeight::Dense(embedding.weight.clone().swap_dims(0, 1))
);
```

Same in `crates/ai-engine-cluster/src/worker.rs::run_worker_full` for the worker's local DecoderBlock construction.

- [ ] **Step 6: Verify**

```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

ALL 154 existing tests must still pass — the refactor is purely additive for the Quantized variant; the Dense variant produces identical output. If any test fails, the matmul shape/ordering is wrong somewhere.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor(runtime): Linear weights routed through LinearWeight<B>::matmul"
```

NO Co-Authored-By.

---

### Task 4: Name map extension for scale companion tensors

**Files:**
- Modify: `crates/ai-engine-runtime/src/name_map.rs`
- Modify: `crates/ai-engine-runtime/tests/name_map.rs`

For each Linear weight that can be quantized, add a `*Scale` TensorId variant that resolves to `<weight_name>.scale`. This is our convention (not bitsandbytes' SCB — see preflight notes).

- [ ] **Step 1: Failing test**

Add to `tests/name_map.rs`:

```rust
#[test]
fn llama3_q_proj_scale_companion_name() {
    let nm = WeightNameMap::for_family(ModelFamily::Llama3);
    assert_eq!(
        nm.lookup(TensorId::LayerQProjScale(12)),
        "model.layers.12.self_attn.q_proj.weight.scale"
    );
    assert_eq!(
        nm.lookup(TensorId::LayerFfnGateScale(0)),
        "model.layers.0.mlp.gate_proj.weight.scale"
    );
    assert_eq!(
        nm.lookup(TensorId::OutputProjectionScale),
        "lm_head.weight.scale"
    );
}
```

- [ ] **Step 2: Implement — extend `TensorId` enum**

```rust
pub enum TensorId {
    Embedding,
    FinalNorm,
    OutputProjection,
    OutputProjectionScale,
    LayerAttnNorm(usize),
    LayerQProj(usize),
    LayerQProjScale(usize),
    LayerKProj(usize),
    LayerKProjScale(usize),
    LayerVProj(usize),
    LayerVProjScale(usize),
    LayerOProj(usize),
    LayerOProjScale(usize),
    LayerFfnNorm(usize),
    LayerFfnGate(usize),
    LayerFfnGateScale(usize),
    LayerFfnUp(usize),
    LayerFfnUpScale(usize),
    LayerFfnDown(usize),
    LayerFfnDownScale(usize),
}
```

In `llama_style`, add the new arms:

```rust
LayerQProjScale(i) => format!("model.layers.{i}.self_attn.q_proj.weight.scale"),
LayerKProjScale(i) => format!("model.layers.{i}.self_attn.k_proj.weight.scale"),
LayerVProjScale(i) => format!("model.layers.{i}.self_attn.v_proj.weight.scale"),
LayerOProjScale(i) => format!("model.layers.{i}.self_attn.o_proj.weight.scale"),
LayerFfnGateScale(i) => format!("model.layers.{i}.mlp.gate_proj.weight.scale"),
LayerFfnUpScale(i) => format!("model.layers.{i}.mlp.up_proj.weight.scale"),
LayerFfnDownScale(i) => format!("model.layers.{i}.mlp.down_proj.weight.scale"),
OutputProjectionScale => "lm_head.weight.scale".into(),
```

- [ ] **Step 3: Verify + commit**

```bash
cargo test -p ai-engine-runtime --test name_map
cargo clippy --workspace --all-targets -- -D warnings
git add -A
git commit -m "feat(runtime): WeightNameMap scale-companion variants for Q8 weights"
```

---

### Task 5: Q8 fixture generator

**Files:**
- Create: `crates/ai-engine-runtime/scripts/generate_q8_fixture.py`
- Create: `crates/ai-engine-runtime/fixtures/toy-llama-3-q8/` (output files)

The Python script that converts the bf16 toy-llama-3 fixture into a Q8 version. The reference_logits.bin comes from running the **bf16** model — the Q8 path must approximate this within tolerance.

- [ ] **Step 1: Write the script**

`crates/ai-engine-runtime/scripts/generate_q8_fixture.py`:

```python
#!/usr/bin/env python3
"""
Generate the toy-llama-3-q8 fixture from the existing toy-llama-3 (bf16).

For each Linear weight in the bf16 fixture:
    scale = max(|weight|) / 127
    packed = clamp(round(weight / scale), -127, 127).astype(int8)
    write `<name>` as int8 with original shape
    write `<name>.scale` as a single-f32 scalar

Non-Linear tensors (embedding, layernorms, biases if any) stay as bf16.

The reference_prompt.txt, tokenizer.json, and reference_logits.bin are copied
unchanged from the bf16 fixture — Q8 output must match bf16 reference within
the tolerance asserted in tests/quant_reference_logits.rs.

Run once; commit outputs.
"""

import json
import shutil
import struct
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

SRC = Path(__file__).resolve().parent.parent / "fixtures" / "toy-llama-3"
OUT = Path(__file__).resolve().parent.parent / "fixtures" / "toy-llama-3-q8"
OUT.mkdir(parents=True, exist_ok=True)

# Tensors to quantize: any Linear weight in the model.
# In Llama-3 / LlamaForCausalLM these are:
LINEAR_PATTERNS = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    "lm_head.weight",   # tied with embedding — we still quantize the lm_head copy if separately stored
]

def should_quantize(name: str) -> bool:
    return any(p in name for p in LINEAR_PATTERNS)

# Load the bf16 weights.
src_tensors = load_file(SRC / "model.safetensors")
out_tensors = {}

for name, t in src_tensors.items():
    if should_quantize(name):
        # Cast to f32 for precise scale computation.
        w_f32 = t.to(torch.float32)
        max_abs = w_f32.abs().max().item()
        scale = max_abs / 127.0 if max_abs > 0 else 1.0
        packed = (w_f32 / scale).round().clamp(-127, 127).to(torch.int8)
        out_tensors[name] = packed
        # Scale companion: shape [1] for safetensors compatibility (rank-0 not allowed by spec).
        out_tensors[f"{name}.scale"] = torch.tensor([scale], dtype=torch.float32)
    else:
        # Pass through as-is.
        out_tensors[name] = t

save_file(out_tensors, OUT / "model.safetensors")

# Copy config + tokenizer + reference unchanged.
for fname in ["config.json", "tokenizer.json", "reference_prompt.txt", "reference_logits.bin"]:
    shutil.copy(SRC / fname, OUT / fname)

# Updated README.
(OUT / "README.md").write_text(
"""# toy-llama-3-q8 fixture

Generated by `scripts/generate_q8_fixture.py` from `toy-llama-3`. Do not edit by hand.

| File | Purpose |
|---|---|
| config.json | Same as toy-llama-3 (architecture identical) |
| model.safetensors | Q8 weights for Linear layers + f32 scale companions; non-Linear tensors are bf16 passthrough |
| tokenizer.json | Same as toy-llama-3 |
| reference_prompt.txt | Same as toy-llama-3 |
| reference_logits.bin | Same as toy-llama-3 — Q8 forward must match within ~1e-2 |

The Rust Q8 forward pass must produce logits matching `reference_logits.bin`
within `max |a - b| < 1e-2` when run on the same prompt against the same
checkpoint. Tolerance is looser than the bf16 gate (1e-3) because Q8
quantization introduces per-tensor rounding error of ~4e-3 per Linear op,
which accumulates over 4 layers + lm_head.
""")

print(f"Wrote Q8 fixture to {OUT}")

# Sanity print
total_orig = sum(t.numel() * t.element_size() for t in src_tensors.values())
total_q8 = sum(t.numel() * t.element_size() for t in out_tensors.values())
print(f"bf16 fixture size: {total_orig} bytes")
print(f"Q8   fixture size: {total_q8} bytes")
print(f"compression ratio: {total_orig / total_q8:.2f}x")
```

- [ ] **Step 2: Run the script**

```bash
cd /home/alessio/aip/airproxy
source .venv-fixture/bin/activate
python crates/ai-engine-runtime/scripts/generate_q8_fixture.py
deactivate
```

Expected: prints a compression ratio around 1.7x–2x (only Linear weights quantize; bf16 embedding/lm_head and bf16 layernorms stay the same, dragging the ratio below the theoretical 4x).

```bash
ls -la crates/ai-engine-runtime/fixtures/toy-llama-3-q8/
```

Expected: 6 files (config.json, model.safetensors, tokenizer.json, reference_prompt.txt, reference_logits.bin, README.md). model.safetensors is ~half the size of the bf16 fixture.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "test(runtime): add Q8 fixture + generator script for toy-llama-3"
```

NO Co-Authored-By.

---

### Task 6: Loader detects int8 + scale companions

**Files:**
- Modify: `crates/ai-engine-runtime/src/loader.rs`
- Create: `crates/ai-engine-runtime/tests/quant_loader.rs`

The loader currently always casts to f32 via `bytes_to_f32_vec`. Extend it: if a tensor's dtype is `I8`, look for a `<name>.scale` companion, and construct a `LinearWeight::Quantized(QuantizedTensor::from_packed(...))` instead of a `LinearWeight::Dense`.

Each `LayerWeights<B>` field changes from `Tensor<B, 2>` to `LinearWeight<B>` for the seven Linear weights (q/k/v/o, gate/up/down). Tensor<B, 1> stays for norms; Tensor<B, 2> stays for the embedding.

- [ ] **Step 1: Update `LayerWeights<B>`**

```rust
use crate::arch::linear::LinearWeight;

pub struct LayerWeights<B: Backend> {
    pub attn_norm: Tensor<B, 1>,
    pub q_proj: LinearWeight<B>,
    pub k_proj: LinearWeight<B>,
    pub v_proj: LinearWeight<B>,
    pub o_proj: LinearWeight<B>,
    pub ffn_norm: Tensor<B, 1>,
    pub ffn_gate: LinearWeight<B>,
    pub ffn_up: LinearWeight<B>,
    pub ffn_down: LinearWeight<B>,
}

pub struct LoadedWeights<B: Backend> {
    pub embedding: Option<Tensor<B, 2>>,
    pub layers: Vec<LayerWeights<B>>,
    pub final_norm: Option<Tensor<B, 1>>,
    pub output_proj: Option<LinearWeight<B>>,
}
```

- [ ] **Step 2: Add a `load_linear_weight` helper**

```rust
use crate::quant::QuantizedTensor;

fn load_linear_weight<B: Backend>(
    st: &SafeTensors<'_>,
    nm: &WeightNameMap,
    weight_id: TensorId,
    scale_id: TensorId,
    device: &B::Device,
) -> anyhow::Result<LinearWeight<B>> {
    let name = nm.lookup(weight_id);
    let view = st.tensor(&name)
        .with_context(|| format!("missing tensor `{name}`"))?;
    let shape = view.shape();
    if shape.len() != 2 {
        anyhow::bail!("tensor `{name}` expected 2D, got shape {:?}", shape);
    }
    let shape2 = [shape[0], shape[1]];

    match view.dtype() {
        safetensors::Dtype::I8 => {
            // Quantized path: load int8 bytes + companion scale.
            let packed_bytes = view.data();
            let packed: Vec<i8> = bytemuck::cast_slice::<u8, i8>(packed_bytes).to_vec();
            let scale_name = nm.lookup(scale_id);
            let scale_view = st.tensor(&scale_name)
                .with_context(|| format!("quantized weight `{name}` missing scale `{scale_name}`"))?;
            let scale_bytes = scale_view.data();
            let scale_f32: &[f32] = bytemuck::cast_slice(scale_bytes);
            if scale_f32.len() != 1 {
                anyhow::bail!("scale `{scale_name}` must be 1 element, got {}", scale_f32.len());
            }
            Ok(LinearWeight::Quantized(QuantizedTensor::from_packed(
                packed, scale_f32[0], shape2, device,
            )))
        }
        _ => {
            // Dense path: existing f32/f16/bf16 conversion.
            let f32_data = bytes_to_f32_vec(view.data(), view.dtype())?;
            Ok(LinearWeight::Dense(Tensor::<B, 2>::from_data(
                TensorData::new(f32_data, shape2),
                device,
            )))
        }
    }
}
```

Add `safetensors::Dtype::I8` handling. (If `bytes_to_f32_vec` doesn't already error on I8, add an explicit unsupported arm.)

- [ ] **Step 3: Update the layer load loop**

In the existing `load_range` function, replace the seven `load_2d` calls per layer with `load_linear_weight` calls using the appropriate (weight, scale) TensorId pairs:

```rust
layers.push(LayerWeights {
    attn_norm: load_1d(TensorId::LayerAttnNorm(i))?,
    q_proj:    load_linear_weight(&st, &nm, TensorId::LayerQProj(i), TensorId::LayerQProjScale(i), device)?,
    k_proj:    load_linear_weight(&st, &nm, TensorId::LayerKProj(i), TensorId::LayerKProjScale(i), device)?,
    v_proj:    load_linear_weight(&st, &nm, TensorId::LayerVProj(i), TensorId::LayerVProjScale(i), device)?,
    o_proj:    load_linear_weight(&st, &nm, TensorId::LayerOProj(i), TensorId::LayerOProjScale(i), device)?,
    ffn_norm:  load_1d(TensorId::LayerFfnNorm(i))?,
    ffn_gate:  load_linear_weight(&st, &nm, TensorId::LayerFfnGate(i), TensorId::LayerFfnGateScale(i), device)?,
    ffn_up:    load_linear_weight(&st, &nm, TensorId::LayerFfnUp(i), TensorId::LayerFfnUpScale(i), device)?,
    ffn_down:  load_linear_weight(&st, &nm, TensorId::LayerFfnDown(i), TensorId::LayerFfnDownScale(i), device)?,
});
```

Similarly for `output_proj`: replace the existing `load_2d(TensorId::OutputProjection)` with `load_linear_weight(..., OutputProjection, OutputProjectionScale, ...)`.

- [ ] **Step 4: Update `Model::from_loaded` and cluster builders**

Since `LayerWeights` now contains `LinearWeight<B>` not `Tensor<B, 2>`, the `swap_dims(0, 1)` calls in `Model::from_loaded` and cluster builders break. The transpose was applied to the raw bf16 tensor; now it must apply within the Dense or Quantized variant.

Add a `swap_dims_inner` method to `LinearWeight`:

```rust
impl<B: Backend> LinearWeight<B> {
    /// Transpose a 2D linear weight in-place (swap rows/cols). Used to convert
    /// safetensors' [out, in] layout to the [in, out] layout our matmul expects.
    pub fn swap_dims(self, a: usize, b: usize) -> Self {
        match self {
            Self::Dense(t) => Self::Dense(t.swap_dims(a, b)),
            Self::Quantized(q) => {
                // Transposing a quantized tensor: dequantize, transpose, requantize.
                // For Q8 with per-tensor scale, this is lossless because the scale
                // is shape-invariant.
                let dq = q.dequantize().swap_dims(a, b);
                Self::Quantized(QuantizedTensor::quantize_from(dq))
            }
        }
    }
}
```

Then in `Model::from_loaded` (and cluster builders), the call site changes:

```rust
let attn = Attention::new(
    layer.q_proj.swap_dims(0, 1),    // returns LinearWeight<B>
    layer.k_proj.swap_dims(0, 1),
    // ...
```

The `swap_dims` for Dense just transposes the underlying f32 tensor; for Quantized it dequantizes, transposes, requantizes — losing a tiny bit of precision per transpose. This is acceptable; it happens once per load. If precision is critical, the alternative is to write a direct int8 transpose, which is straightforward but adds code.

- [ ] **Step 5: Quantized loader test**

`crates/ai-engine-runtime/tests/quant_loader.rs`:

```rust
use ai_engine_runtime::arch::linear::LinearWeight;
use ai_engine_runtime::config::ModelConfig;
use ai_engine_runtime::loader::load_range;
use std::path::PathBuf;

type B = burn_ndarray::NdArray;

fn q8_fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/toy-llama-3-q8")
}

#[test]
fn load_q8_fixture_produces_quantized_weights() {
    let cfg = ModelConfig::from_file(&q8_fixture_path().join("config.json")).unwrap();
    let dev = Default::default();
    let weights = load_range::<B>(
        &q8_fixture_path().join("model.safetensors"), &cfg,
        0..cfg.n_layers, true, true, &dev,
    ).unwrap();
    // Each layer's Linear weights should be the Quantized variant.
    for layer in &weights.layers {
        assert!(matches!(layer.q_proj, LinearWeight::Quantized(_)));
        assert!(matches!(layer.k_proj, LinearWeight::Quantized(_)));
        assert!(matches!(layer.v_proj, LinearWeight::Quantized(_)));
        assert!(matches!(layer.o_proj, LinearWeight::Quantized(_)));
        assert!(matches!(layer.ffn_gate, LinearWeight::Quantized(_)));
        assert!(matches!(layer.ffn_up, LinearWeight::Quantized(_)));
        assert!(matches!(layer.ffn_down, LinearWeight::Quantized(_)));
    }
    // Embedding stays dense (we didn't quantize it).
    assert!(weights.embedding.is_some());
    // output_proj is None because the toy uses tie_word_embeddings.
    assert!(weights.output_proj.is_none());
}
```

- [ ] **Step 6: Verify + commit**

```bash
cargo test -p ai-engine-runtime
cargo clippy --workspace --all-targets -- -D warnings
git add -A
git commit -m "feat(runtime): safetensors loader detects int8 weights + scale companions"
```

NO Co-Authored-By.

---

### Task 7: End-to-end Q8 correctness gate

**Files:**
- Create: `crates/ai-engine-runtime/tests/quant_reference_logits.rs`

The release gate for Q8: load the toy-llama-3-q8 fixture, run a forward pass, compare against the SAME reference_logits.bin used by the bf16 gate. Tolerance is 1e-2 (looser than bf16's 1e-3 because Q8 introduces ~4e-3 per-op rounding that accumulates).

- [ ] **Step 1: Test**

```rust
use ai_engine_runtime::arch::model::Model;
use ai_engine_runtime::config::ModelConfig;
use ai_engine_runtime::loader::load_range;
use ai_engine_tokenizer::{HfTokenizer, Tokenizer};
use burn::tensor::{Tensor, Int, TensorData};
use std::path::PathBuf;

type B = burn_ndarray::NdArray;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/toy-llama-3-q8")
}

#[test]
fn q8_forward_matches_bf16_reference_within_quantization_tolerance() {
    let fix = fixture();
    let cfg = ModelConfig::from_file(&fix.join("config.json")).unwrap();
    let tok = HfTokenizer::from_path(fix.join("tokenizer.json")).unwrap();
    let prompt = std::fs::read_to_string(fix.join("reference_prompt.txt")).unwrap();
    let ids = tok.encode(prompt.trim()).unwrap();

    let dev = Default::default();
    let weights = load_range::<B>(
        &fix.join("model.safetensors"), &cfg,
        0..cfg.n_layers, true, true, &dev,
    ).unwrap();
    let model = Model::<B>::from_loaded(&cfg, weights, &dev).unwrap();

    let ids_i32: Vec<i32> = ids.iter().map(|x| *x as i32).collect();
    let token_ids = Tensor::<B, 2, Int>::from_data(
        TensorData::new(ids_i32, [1, ids.len()]),
        &dev,
    );
    let logits = model.forward(token_ids, 0);
    let last_pos_logits: Tensor<B, 1> = logits
        .slice([0..1, (ids.len() - 1)..ids.len(), 0..cfg.vocab_size])
        .reshape([cfg.vocab_size]);

    let ref_bytes = std::fs::read(fix.join("reference_logits.bin")).unwrap();
    let ref_f32: &[f32] = bytemuck::cast_slice(&ref_bytes);
    assert_eq!(ref_f32.len(), cfg.vocab_size);

    let got: Vec<f32> = last_pos_logits.to_data().to_vec().unwrap();

    let mut max_abs_diff = 0.0_f32;
    let mut argmax_us = (0usize, f32::NEG_INFINITY);
    let mut argmax_ref = (0usize, f32::NEG_INFINITY);
    for (i, (a, b)) in got.iter().zip(ref_f32.iter()).enumerate() {
        let d = (a - b).abs();
        if d > max_abs_diff { max_abs_diff = d; }
        if *a > argmax_us.1 { argmax_us = (i, *a); }
        if *b > argmax_ref.1 { argmax_ref = (i, *b); }
    }
    eprintln!("Q8 vs bf16-reference max |a-b| = {max_abs_diff}");
    eprintln!("argmax  ours = {} ({})", argmax_us.0, argmax_us.1);
    eprintln!("argmax  ref  = {} ({})", argmax_ref.0, argmax_ref.1);

    // Q8 tolerance: 1e-2 worst-case.
    assert!(
        max_abs_diff < 1e-2,
        "Q8 correctness gate failed: max |a-b| = {max_abs_diff} (tolerance 1e-2)"
    );
}
```

- [ ] **Step 2: Run + iterate**

```bash
cd /home/alessio/aip/airproxy
cargo test -p ai-engine-runtime --test quant_reference_logits -- --nocapture
```

Expected behaviors:
- **`max_abs_diff < 1e-3`**: Excellent — Q8 is essentially indistinguishable from bf16. May happen if the toy's weights are well-behaved.
- **`max_abs_diff` between 1e-3 and 1e-2**: Expected. Q8 quantization introduces this magnitude of error per Linear op; over 4 layers + lm_head it's well within 1e-2.
- **`max_abs_diff` > 1e-2**: Real bug. Likely candidates:
  - Quantize-then-dequantize round-trip is wrong (test in Task 1 catches this in isolation; if isolated quant works but full-model doesn't, the issue is integration)
  - `swap_dims` on a Quantized LinearWeight loses too much precision (Task 6 dequantize-transpose-requantize). To debug: temporarily make `swap_dims` direct on the int8 buffer.
  - Loader is reading the scale companion incorrectly (e.g., wrong endianness, wrong dtype). Print out a few scale values and compare to what the Python script wrote.
- **Argmax mismatch**: If argmax_us != argmax_ref, the Q8 output is producing a different top token than bf16. For a toy model with random weights this can happen if quantization error pushes a borderline logit past another — assert `argmax_us == argmax_ref` only if `max_abs_diff < 5e-3`, otherwise just check tolerance.

For the test in this task, **don't assert argmax equality** — the toy's random weights make a few logits very close, and Q8 noise can swap them. The tolerance assertion is the gate.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "test(runtime): Q8 forward matches bf16 reference within 1e-2 tolerance"
```

NO Co-Authored-By.

---

### Task 8: Cluster + Q8 (integration test)

**Files:**
- Create: `crates/ai-engine-cluster/tests/q8_cluster.rs`

Prove Q8 works through the distributed path: leader + 2 workers load a Q8 fixture, run a forward pass, output matches the single-node Q8 baseline.

- [ ] **Step 1: Test**

```rust
//! Q8 cluster generation matches Q8 single-node baseline.

use std::path::PathBuf;

type B = burn_ndarray::NdArray;

fn q8_fixture() -> PathBuf {
    // The fixture lives in ai-engine-runtime; reach it from ai-engine-cluster.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .join("ai-engine-runtime/fixtures/toy-llama-3-q8")
}

fn single_node_q8_greedy_5(fix: &std::path::Path, cfg: &ai_engine_runtime::config::ModelConfig, prompt_ids: &[i32]) -> Vec<u32> {
    use ai_engine_runtime::{
        arch::model::Model, kv_cache::KvCacheSlot, loader::load_range,
        sample::{sample, SamplingConfig},
    };
    use burn::tensor::{Tensor, TensorData, Int};

    let dev = Default::default();
    let weights = load_range::<B>(&fix.join("model.safetensors"), cfg, 0..cfg.n_layers, true, true, &dev).unwrap();
    let model = Model::<B>::from_loaded(cfg, weights, &dev).unwrap();
    let mut caches: Vec<KvCacheSlot<B>> = (0..cfg.n_layers).map(|_| {
        KvCacheSlot::<B>::new(1, cfg.n_kv_heads, cfg.max_position_embeddings, cfg.head_dim, &dev)
    }).collect();
    let prompt = Tensor::<B, 2, Int>::from_data(TensorData::new(prompt_ids.to_vec(), [1, prompt_ids.len()]), &dev);
    let logits = model.forward_with_caches(prompt, 0, &mut caches);
    let last: Vec<f32> = logits.slice([0..1, (prompt_ids.len()-1)..prompt_ids.len(), 0..cfg.vocab_size])
        .reshape([cfg.vocab_size]).to_data().to_vec().unwrap();
    let scfg = SamplingConfig { temperature: 0.0, top_p: None, top_k: None, seed: 0 };
    let mut tokens = vec![sample(&last, &scfg)];
    for i in 1..5 {
        let next = Tensor::<B, 2, Int>::from_data(
            TensorData::new(vec![*tokens.last().unwrap() as i32], [1, 1]), &dev,
        );
        let logits = model.forward_with_caches(next, prompt_ids.len() + i - 1, &mut caches);
        let v: Vec<f32> = logits.reshape([cfg.vocab_size]).to_data().to_vec().unwrap();
        tokens.push(sample(&v, &scfg));
    }
    tokens
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn q8_cluster_generation_matches_q8_single_node() {
    use ai_engine_cluster::{
        capability::BackendKind, leader::{ClusterLeader, LeaderConfig, WorkerEndpoint},
        tls::generate_node_identity, transport::quic::server_endpoint,
        worker::run_worker_full,
    };

    let fix = q8_fixture();
    let cfg = ai_engine_runtime::config::ModelConfig::from_file(&fix.join("config.json")).unwrap();
    let tok = ai_engine_tokenizer::HfTokenizer::from_path(fix.join("tokenizer.json")).unwrap();
    let prompt = "The quick brown fox";
    let ids: Vec<u32> = ai_engine_tokenizer::Tokenizer::encode(&tok, prompt).unwrap();
    let ids_i32: Vec<i32> = ids.iter().map(|x| *x as i32).collect();

    let baseline = single_node_q8_greedy_5(&fix, &cfg, &ids_i32);

    let w1_id = generate_node_identity("w1").unwrap();
    let w1_ep = server_endpoint(&w1_id, "127.0.0.1:0".parse().unwrap()).unwrap();
    let w1_addr = w1_ep.local_addr().unwrap();
    let w2_id = generate_node_identity("w2").unwrap();
    let w2_ep = server_endpoint(&w2_id, "127.0.0.1:0".parse().unwrap()).unwrap();
    let w2_addr = w2_ep.local_addr().unwrap();

    let model_path = fix.join("model.safetensors");
    let cfg_w1 = cfg.clone();
    let mp1 = model_path.clone();
    tokio::spawn(async move {
        run_worker_full::<B>(w1_ep, "w1".into(), BackendKind::Cpu, mp1, cfg_w1).await
    });
    let cfg_w2 = cfg.clone();
    let mp2 = model_path.clone();
    tokio::spawn(async move {
        run_worker_full::<B>(w2_ep, "w2".into(), BackendKind::Cpu, mp2, cfg_w2).await
    });

    let leader_id = generate_node_identity("leader").unwrap();
    let lcfg = LeaderConfig {
        cluster_id: "q8-test".into(),
        leader_node_id: "leader".into(),
        model_id: "toy-q8".into(),
        n_layers: cfg.n_layers,
        layer_bytes: 256 * 1024,
        embed_output_bytes: 256 * 1024,
        per_node_overhead: 64 * 1024,
        workers: vec![
            WorkerEndpoint { node_id: "w1".into(), addr: w1_addr, fingerprint: w1_id.fingerprint.clone() },
            WorkerEndpoint { node_id: "w2".into(), addr: w2_addr, fingerprint: w2_id.fingerprint.clone() },
        ],
        partition_override: Some(vec![("w1".into(), 0..2), ("w2".into(), 2..4)]),
    };
    let leader = ClusterLeader::start(leader_id, lcfg).await.unwrap();

    let cluster_tokens = leader.generate::<B>(
        &model_path, &cfg, 0..0, &ids_i32, 5,
        ai_engine_runtime::sample::SamplingConfig { temperature: 0.0, top_p: None, top_k: None, seed: 0 },
    ).await.unwrap();

    assert_eq!(
        cluster_tokens, baseline,
        "Q8 cluster generation must match Q8 single-node baseline"
    );
}
```

- [ ] **Step 2: Run + commit**

```bash
cargo test -p ai-engine-cluster --test q8_cluster -- --nocapture
cargo clippy --workspace --all-targets -- -D warnings
git add -A
git commit -m "test(cluster): Q8 distributed generation matches Q8 single-node baseline"
```

NO Co-Authored-By.

---

### Task 9: README + tag v0.3.0-alpha

**Files:**
- Modify: `README.md`
- Tag: `v0.3.0-alpha`

- [ ] **Step 1: Final verification**

```bash
cd /home/alessio/aip/airproxy
cargo test --workspace 2>&1 | grep "test result" | awk '{passed += $4; ignored += $8} END {print "PASSED=" passed " IGNORED=" ignored}'
cargo clippy --workspace --all-targets -- -D warnings
cargo build --workspace --release
```

Expected: ~160 tests passing (154 + 6 new = quant + linear_weight + name_map + quant_loader + quant_reference_logits + q8_cluster). Clippy + release clean.

- [ ] **Step 2: Update README**

Add a section under "Release history" (or top of file):

```markdown
### v0.3.0-alpha — Q8 weight quantization

ai-engine v0.3.0-alpha adds Q8 (8-bit symmetric per-tensor) weight
quantization to `ai-engine-runtime`. Each Linear weight is stored as
int8 + an f32 scale; the forward pass dequantizes per call. Memory at
rest is ~2× smaller for typical Llama-family checkpoints (more if the
embedding+lm_head dominate; less if they don't — they stay bf16).

Correctness:
- Q8 forward matches bf16 reference within 1e-2 on the toy-llama-3 fixture.
- 3-node Q8 cluster generation matches single-node Q8 generation exactly.

Generate a Q8 checkpoint from any bf16 safetensors model using the
`scripts/generate_q8_fixture.py` template — adapt the input/output paths
for real models like Llama-3-8B.

Known limitations:
- Q4 (4-bit packed) not supported — see Plan 6.
- Dequantize-on-forward is unfused; specialized int8 GEMM kernels would
  be substantially faster on GPU backends.
- Loader recognizes only our `<name>.scale` convention, not bitsandbytes
  `<name>.SCB` per-channel scales or AWQ/GPTQ packed layouts.
- Activations stay f32.
```

- [ ] **Step 3: Commit + tag**

```bash
git add README.md
git commit -m "docs: announce v0.3.0-alpha Q8 quantization release"
git tag v0.3.0-alpha
git log --oneline -5
git tag
```

NO Co-Authored-By.

## Report
- Status
- Test count
- Tag listing
- `git log --oneline -5`

---

## Self-review

**Spec coverage:**
- "Q8 weight format" → Task 1 (primitive) + Task 6 (loader)
- "Load quantized safetensors checkpoints" → Task 6
- "Dequantize-on-forward" → Tasks 1, 2 (matmul dispatches via LinearWeight)
- "Test against quantized version of toy fixture" → Tasks 5 (fixture), 7 (correctness gate), 8 (cluster)
- "Document perf tradeoff" → Task 9 README
- "Plain int8 (Q8) only — defer Q4" → scope rule, restated in Task 9 README

**Placeholder scan:**
No `TBD` / `implement later` / `add error handling` strings. Task 3 has prose-level direction for `with_random_weights` adapters but the code shown is complete. Task 5's Python script is checked-in code.

**Type consistency:**
- `QuantizedTensor<B>` (Task 1) → field of `LinearWeight::Quantized` (Task 2). Same backend B. ✓
- `LinearWeight<B>` (Task 2) → consumed by Attention / SwiGluFfn / OutputProjection (Task 3) and produced by loader (Task 6). Field shapes consistent: 2D linear weight. ✓
- `LayerWeights<B>` (Task 6) → consumed by `Model::from_loaded` and cluster builders (Task 3 + Task 6 cross-references). Migration done in Task 3 (cluster) and Task 6 (loader). ✓
- `TensorId::*Scale` (Task 4) → consumed by `load_linear_weight` (Task 6). Variants match. ✓
- Q8 fixture (Task 5) → consumed by `tests/quant_loader.rs` (Task 6), `tests/quant_reference_logits.rs` (Task 7), `tests/q8_cluster.rs` (Task 8). Same path. ✓

**Acknowledged risks:**

1. **`swap_dims` precision loss for Quantized (Task 6 Step 4)**: dequantize-transpose-requantize loses a tiny bit of precision per call. Happens once per load. The correctness gate in Task 7 will detect if it pushes past tolerance — the toy fixture is small enough that the accumulated loss should be well under 1e-3. If real-model loaders see issues, write a direct int8 transpose; not in scope for Plan 5.

2. **HF compatibility**: Real HF checkpoints quantized via `bitsandbytes` store per-channel int8 with `<name>.SCB` (column-wise scales), not per-tensor with `<name>.scale`. Our convention is simpler but isn't compatible with downloaded HF int8 dumps. Real-model validation will need a converter script. Documented as a limitation.

3. **Loader changes are invasive** — every LayerWeights field type changes. This ripples through every consumer. Task 3 (in-tree consumers) + Task 6 (loader itself) cover all sites I can identify in the current codebase. If a test fails at compile-time after these tasks, the missing site needs the same `LinearWeight::Dense(...)` wrap.

---

## Execution Handoff

Plan 5 saved to `docs/superpowers/plans/2026-05-24-plan-5-quantization-q8.md`. 9 tasks total.

Two execution options:

**1. Subagent-Driven (recommended)** — Tasks 1, 2 are bounded primitives. Task 3 is a wide refactor — touches every primitive constructor. Tasks 4–6 are loader + fixture. Tasks 7–8 are correctness gates. Task 9 is the release.

**2. Inline Execution** — also reasonable; Plan 5 is similar size to Plan 4.

After v0.3.0-alpha ships, Plan 6 (Q4 + bit-packing + AWQ/GPTQ loader) is the natural follow-up.
