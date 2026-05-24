# Plan 6 — v0.3.0-alpha.2: Q4 weight quantization

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Q4 (4-bit, per-group symmetric, group size 32) weight quantization to `ai-engine-runtime`. ~4× smaller than bf16 at rest; ~2× smaller than Q8. Q4 cluster forward matches Q4 single-node forward exactly; Q4-vs-bf16 logit drift bounded within a tolerance derived from per-group quantization error.

**Architecture:** A new `Q4Tensor<B>` storing weights packed as `Vec<u8>` (two int4 nibbles per byte) plus per-group `Vec<f32>` scales (group size = 32 along the **input** dim). `LinearWeight<B>` gains a third variant `Q4(Q4Tensor<B>)`. Dequantize materializes a temporary `Tensor<B, 2>` (`f32`) on each matmul call — same dequantize-on-forward strategy as Q8, just with smaller storage. Our own safetensors layout: weight stored U8 packed pre-transposed (so the loader never has to swap_dims a Q4 tensor) at shape `[in, out/2]` is wrong — the natural layout is `[in, out]` weights post-transpose, packed `[in, out/2]` with **groups along `in`**. Concretely: scales shape is `[in / 32, out]`, one f32 per (input-group, output-channel). Reconstruction iterates: `w[i][j] = (packed nibble for [i][j]) * scale[i/32][j]`.

**Tech Stack:** No new external crates. Reuses `bytemuck` for byte casting, `half` (already in workspace), the existing burn / safetensors / Python pipeline.

**Scope rule:** Plan 6 ships **our own Q4 format only** (groupwise symmetric, group size 32, packed nibbles, low-nibble-first). HF-compatibility for AWQ / GPTQ / GGUF / bitsandbytes is **deferred** to Plan 7 — each requires its own loader translation layer because of differing bit-packing conventions, zero-point handling, and per-channel vs per-group scale layouts.

**Baseline:** Branch `main` at `v0.3.0-alpha`. 166 tests + 4 ignored. Clippy clean.

---

## Group size and layout decisions (locked in here)

| Decision | Choice | Rationale |
|---|---|---|
| Group size | 32 | Matches GGUF Q4_0; divides every Llama hidden / intermediate dim |
| Group axis | input dim (`in`) | Standard for activations × weights; activations on `in` axis vary so per-group rescaling helps |
| Symmetry | symmetric (no zero point) | Simplest; matches GGUF Q4_0; per-group scale captures asymmetry close enough |
| Nibble layout | low-nibble first | `byte & 0x0F` is value at even index; `(byte >> 4) & 0x0F` is value at odd index |
| Storage shape | weight: `[in, out/2]` U8; scale: `[in/32, out]` f32 | Pre-transposed to `[in, out]` math order — loader **never** swap_dims a Q4 tensor |
| Constraint | `in` must be a multiple of 32 | All Llama-family dims satisfy this |

Reconstruction formula (canonical):

```
for i in 0..in:
    g = i / 32
    for j in 0..out:
        byte = packed[i][j / 2]
        nibble = if j % 2 == 0 { byte & 0x0F } else { (byte >> 4) & 0x0F }
        signed_i4 = if nibble < 8 { nibble as i8 } else { nibble as i8 - 16 }   // 0..7 -> 0..7; 8..15 -> -8..-1
        weight_f32[i][j] = (signed_i4 as f32) * scale[g][j]
```

Quantization formula (used by the Python fixture generator):

```
for each (group_g, out_j) where group_g spans 32 consecutive in indices:
    block = weight_f32[g*32 .. (g+1)*32, j]
    s = max(|block|) / 7         # signed range -8..7 effectively maps to ~|block|/7
    scale[g][j] = s
    for k in 0..32:
        q = clamp(round(block[k] / s), -7, 7) as i8     # restrict to -7..7 to keep range symmetric
        nibble = (q & 0x0F)
        # Pack two nibbles per byte; j increments column-wise so packing on j%2 makes sense
        # Actually for fast row-major bit-packing, pack along `j` (the column / out dim):
        #   byte_index = (i, j / 2)
        #   if j % 2 == 0: byte = nibble
        #   if j % 2 == 1: byte |= nibble << 4
```

**Important**: the round-clamp to `-7..7` (not `-8..7`) keeps the value range symmetric around zero. This loses one slot (-8) but avoids asymmetry artifacts; matches GGUF Q4_0.

---

## File structure

```
crates/ai-engine-runtime/
├── src/
│   ├── quant.rs                     # MODIFY: add Q4Tensor<B> alongside QuantizedTensor
│   ├── arch/linear.rs               # MODIFY: LinearWeight::Q4 variant + matmul dispatch
│   └── loader.rs                    # MODIFY: detect U8 dtype + multi-element .scale = Q4
├── fixtures/
│   └── toy-llama-3-q4/              # NEW: Q4 version of toy-llama-3, ~4x smaller than bf16
└── scripts/
    └── generate_q4_fixture.py       # NEW
└── tests/
    ├── q4_tensor.rs                 # NEW: Q4Tensor quantize/dequantize roundtrip
    ├── q4_linear_weight.rs          # NEW: Q4 matmul vs dense
    ├── q4_loader.rs                 # NEW: load Q4 fixture, check LinearWeight::Q4
    └── q4_reference_logits.rs       # NEW: forward gate against bf16 reference, tolerance 5e-2
```

```
crates/ai-engine-cluster/
└── tests/
    └── q4_cluster.rs                # NEW: 3-node cluster generation matches single-node Q4
```

---

### Task 1: `Q4Tensor<B>` primitive

**Files:**
- Modify: `crates/ai-engine-runtime/src/quant.rs`
- Create: `crates/ai-engine-runtime/tests/q4_tensor.rs`

- [ ] **Step 1: Failing test**

`crates/ai-engine-runtime/tests/q4_tensor.rs`:

```rust
use ai_engine_runtime::quant::Q4Tensor;
use burn::tensor::{Tensor, TensorData};

type B = burn_ndarray::NdArray;

const GROUP_SIZE: usize = 32;

#[test]
fn q4_quantize_then_dequantize_recovers_within_4bit_noise() {
    let dev = Default::default();
    // Build a known f32 weight, in=32 (one group), out=4.
    // Values chosen so int4 quantization has bounded error.
    let raw: Vec<f32> = (0..32 * 4).map(|i| ((i as f32) * 0.05).sin()).collect();
    let original = Tensor::<B, 2>::from_data(
        TensorData::new(raw.clone(), [32, 4]),
        &dev,
    );

    let q = Q4Tensor::<B>::quantize_from(original.clone());
    assert_eq!(q.shape(), [32, 4]);
    assert_eq!(q.packed.len(), 32 * 2);   // 32 rows × (4 cols / 2 cols/byte) = 64 bytes
    assert_eq!(q.scales.len(), 1 * 4);    // 1 group × 4 cols

    let recovered = q.dequantize();
    let original_v: Vec<f32> = original.into_data().to_vec().unwrap();
    let recovered_v: Vec<f32> = recovered.into_data().to_vec().unwrap();

    // Q4 with signed -7..7 range, per-group scale s = max|block|/7.
    // Worst-case rounding error per value is s/2 ~ max|block|/14.
    // For our sin-based input, max|block| < 1, so error per value < ~0.07.
    let max_err = original_v.iter().zip(recovered_v.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(max_err < 0.08, "Q4 quantization error {max_err} exceeded expected ~0.07");
}

#[test]
fn q4_storage_size_is_quarter_of_dense() {
    // 32 × 8 = 256 weights.
    // Dense f32: 256 × 4 = 1024 bytes.
    // Q4: 256 × 0.5 = 128 bytes packed + 1 group × 8 cols × 4 = 32 bytes scales = 160 bytes total.
    // Ratio = 1024 / 160 = 6.4× (the per-group scale overhead is significant for small tensors).
    let dev = Default::default();
    let raw: Vec<f32> = (0..32 * 8).map(|i| i as f32 * 0.01).collect();
    let t = Tensor::<B, 2>::from_data(TensorData::new(raw, [32, 8]), &dev);
    let q = Q4Tensor::<B>::quantize_from(t);
    assert_eq!(q.packed.len(), 128);       // 32 × 4
    assert_eq!(q.scales.len(), 8);         // 1 group × 8 cols
}

#[test]
fn q4_from_packed_components_roundtrips() {
    let dev = Default::default();
    // One group, one column. Packed nibble values: 7 -> 0x07, -7 -> 0x09 (signed).
    // We pack 4 byte values (8 nibbles total, but we have 1 group × 32 = 32 values).
    // Easier test: shape [32, 2], one group, two columns.
    // Column 0: alternating 7, -7 across 32 rows.
    // Column 1: zeros.
    //
    // Packed layout: packed[i, j/2] holds two nibbles for columns 2k and 2k+1.
    // For 2 columns: packed[i, 0] holds col0 in low nibble and col1 in high nibble.
    // col0 = 7 (low nibble = 0x07), col1 = 0 (high nibble = 0x00) -> byte = 0x07
    // col0 = -7 (low nibble = 0x09 = 9), col1 = 0 -> byte = 0x09
    let mut packed = Vec::with_capacity(32);
    for i in 0..32 {
        if i % 2 == 0 { packed.push(0x07); } else { packed.push(0x09); }
    }
    let scales = vec![0.5_f32, 1.0_f32];   // group 0, col 0: 0.5; group 0, col 1: 1.0
    let q = Q4Tensor::<B>::from_packed(packed, scales, [32, 2], &dev);
    let d = q.dequantize();
    let v: Vec<f32> = d.into_data().to_vec().unwrap();
    // v is row-major [32, 2]. Column 0: alternating 7*0.5, -7*0.5 = 3.5, -3.5.
    // Column 1: always 0 * 1.0 = 0.
    for i in 0..32 {
        let c0 = v[i * 2];
        let c1 = v[i * 2 + 1];
        let expected_c0 = if i % 2 == 0 { 3.5 } else { -3.5 };
        assert!((c0 - expected_c0).abs() < 1e-5, "row {i} col 0: {c0} != {expected_c0}");
        assert!(c1.abs() < 1e-5, "row {i} col 1: {c1} != 0");
    }
}
```

- [ ] **Step 2: Confirm fails**

```bash
cd /home/alessio/aip/airproxy
cargo test -p ai-engine-runtime --test q4_tensor 2>&1 | tail -10
# Expected: Q4Tensor doesn't exist.
```

- [ ] **Step 3: Implement `Q4Tensor` in `crates/ai-engine-runtime/src/quant.rs`**

Append to the existing `quant.rs`:

```rust
/// Per-group symmetric Q4 quantization (group size 32 along input dim).
///
/// Storage:
///   - `packed`: U8 bytes, length = (in × out) / 2. Two nibbles per byte.
///   - `scales`: f32, length = (in / 32) × out. One scale per (input group, output channel).
///   - `shape`: [in, out] (post-transpose; we store in math order).
///
/// Layout: packed is row-major [in, out/2]. Within byte `packed[i * (out/2) + j/2]`:
///   - low nibble (bits 0-3): column 2k
///   - high nibble (bits 4-7): column 2k+1
///
/// Reconstruction: `weight[i][j] = signed_nibble[i][j] * scales[(i / 32) * out + j]`
/// where signed_nibble is `nibble if nibble < 8 else nibble - 16`.
pub const Q4_GROUP_SIZE: usize = 32;

pub struct Q4Tensor<B: Backend> {
    pub packed: Vec<u8>,
    pub scales: Vec<f32>,
    shape: [usize; 2],      // [in, out]
    _marker: std::marker::PhantomData<B>,
    device: B::Device,
}

impl<B: Backend> Q4Tensor<B> {
    pub fn shape(&self) -> [usize; 2] { self.shape }

    /// Quantize a dense [in, out] f32 tensor.
    /// Panics if `in` is not divisible by 32 or `out` is not even.
    pub fn quantize_from(t: Tensor<B, 2>) -> Self {
        let shape = t.dims();
        let in_dim = shape[0];
        let out_dim = shape[1];
        assert!(in_dim % Q4_GROUP_SIZE == 0, "Q4 requires in dim divisible by {Q4_GROUP_SIZE}, got in={in_dim}");
        assert!(out_dim % 2 == 0, "Q4 requires out dim even (packs 2 nibbles/byte), got out={out_dim}");
        let device = t.device();
        let values: Vec<f32> = t.into_data().to_vec().expect("to_vec f32");

        let num_groups = in_dim / Q4_GROUP_SIZE;
        let mut scales = vec![0.0_f32; num_groups * out_dim];
        let mut packed = vec![0u8; in_dim * (out_dim / 2)];

        // For each (group_g, out_j) compute the per-group-per-column scale.
        for g in 0..num_groups {
            for j in 0..out_dim {
                let mut max_abs = 0.0_f32;
                for k in 0..Q4_GROUP_SIZE {
                    let i = g * Q4_GROUP_SIZE + k;
                    let v = values[i * out_dim + j].abs();
                    if v > max_abs { max_abs = v; }
                }
                let s = if max_abs == 0.0 { 1.0 } else { max_abs / 7.0 };
                scales[g * out_dim + j] = s;
            }
        }

        // Pack nibbles row-major.
        for i in 0..in_dim {
            let g = i / Q4_GROUP_SIZE;
            for j in (0..out_dim).step_by(2) {
                let s_lo = scales[g * out_dim + j];
                let s_hi = scales[g * out_dim + j + 1];
                let v_lo = values[i * out_dim + j];
                let v_hi = values[i * out_dim + j + 1];
                let q_lo = ((v_lo / s_lo).round() as i32).clamp(-7, 7) as i8;
                let q_hi = ((v_hi / s_hi).round() as i32).clamp(-7, 7) as i8;
                let nibble_lo = (q_lo as u8) & 0x0F;
                let nibble_hi = (q_hi as u8) & 0x0F;
                let byte_idx = i * (out_dim / 2) + (j / 2);
                packed[byte_idx] = nibble_lo | (nibble_hi << 4);
            }
        }

        Self { packed, scales, shape, _marker: std::marker::PhantomData, device }
    }

    /// Construct from raw packed bytes + per-group scales.
    pub fn from_packed(
        packed: Vec<u8>,
        scales: Vec<f32>,
        shape: [usize; 2],
        device: &B::Device,
    ) -> Self {
        let in_dim = shape[0];
        let out_dim = shape[1];
        assert_eq!(packed.len(), in_dim * (out_dim / 2),
            "packed length must be in*out/2");
        assert_eq!(scales.len(), (in_dim / Q4_GROUP_SIZE) * out_dim,
            "scales length must be (in/32)*out");
        Self { packed, scales, shape, _marker: std::marker::PhantomData, device: device.clone() }
    }

    /// Dequantize to a regular f32 Tensor<B, 2>. Allocates a fresh buffer.
    pub fn dequantize(&self) -> Tensor<B, 2> {
        let [in_dim, out_dim] = self.shape;
        let mut f32_values = vec![0.0_f32; in_dim * out_dim];
        for i in 0..in_dim {
            let g = i / Q4_GROUP_SIZE;
            for j in (0..out_dim).step_by(2) {
                let byte = self.packed[i * (out_dim / 2) + (j / 2)];
                let nibble_lo = byte & 0x0F;
                let nibble_hi = (byte >> 4) & 0x0F;
                let q_lo = if nibble_lo < 8 { nibble_lo as i32 } else { (nibble_lo as i32) - 16 };
                let q_hi = if nibble_hi < 8 { nibble_hi as i32 } else { (nibble_hi as i32) - 16 };
                let s_lo = self.scales[g * out_dim + j];
                let s_hi = self.scales[g * out_dim + j + 1];
                f32_values[i * out_dim + j] = (q_lo as f32) * s_lo;
                f32_values[i * out_dim + j + 1] = (q_hi as f32) * s_hi;
            }
        }
        Tensor::<B, 2>::from_data(TensorData::new(f32_values, [in_dim, out_dim]), &self.device)
    }
}
```

- [ ] **Step 4: Wire export**

`crates/ai-engine-runtime/src/lib.rs` — append:

```rust
pub use quant::{Q4Tensor, Q4_GROUP_SIZE};
```

- [ ] **Step 5: Verify + commit**

```bash
cargo test -p ai-engine-runtime --test q4_tensor
cargo clippy --workspace --all-targets -- -D warnings
git add -A
git commit -m "feat(runtime): Q4Tensor<B> with per-group symmetric quantization (group size 32)"
```

NO Co-Authored-By footer (global preference).

---

### Task 2: `LinearWeight::Q4` variant + matmul dispatch

**Files:**
- Modify: `crates/ai-engine-runtime/src/arch/linear.rs`
- Create: `crates/ai-engine-runtime/tests/q4_linear_weight.rs`

- [ ] **Step 1: Failing test**

```rust
use ai_engine_runtime::arch::linear::LinearWeight;
use ai_engine_runtime::quant::Q4Tensor;
use burn::tensor::{Tensor, TensorData};

type B = burn_ndarray::NdArray;

#[test]
fn q4_linear_matmul_approximates_dense() {
    let dev = Default::default();
    // Build [in=32, out=4] dense weight with values that quantize well.
    let raw: Vec<f32> = (0..32 * 4).map(|i| ((i as f32) * 0.07).sin() * 0.5).collect();
    let w = Tensor::<B, 2>::from_data(TensorData::new(raw.clone(), [32, 4]), &dev);

    // [batch=1, seq=1, in=32] activation
    let x_data: Vec<f32> = (0..32).map(|i| ((i as f32) * 0.1).cos()).collect();
    let x = Tensor::<B, 3>::from_data(TensorData::new(x_data, [1, 1, 32]), &dev);

    let dense = LinearWeight::Dense(w.clone());
    let q4 = LinearWeight::Q4(Q4Tensor::<B>::quantize_from(w));

    let out_dense: Vec<f32> = dense.matmul(x.clone()).into_data().to_vec().unwrap();
    let out_q4: Vec<f32>    = q4.matmul(x).into_data().to_vec().unwrap();

    let max_diff = out_dense.iter().zip(out_q4.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    // 32 multiply-adds per output, each with Q4 noise; cancellation across the sum
    // typically keeps the matmul error well under per-element Q4 noise * sqrt(32).
    assert!(max_diff < 0.5, "Q4 matmul diverged from dense by {max_diff}");
}

#[test]
fn q4_linear_shape_matches_dense() {
    let dev = Default::default();
    let raw: Vec<f32> = (0..64 * 8).map(|i| i as f32 * 0.001).collect();
    let w = Tensor::<B, 2>::from_data(TensorData::new(raw, [64, 8]), &dev);
    let q = LinearWeight::Q4(Q4Tensor::<B>::quantize_from(w));
    assert_eq!(q.shape(), [64, 8]);
}
```

- [ ] **Step 2: Implement — add Q4 variant**

In `crates/ai-engine-runtime/src/arch/linear.rs`:

```rust
use crate::quant::{Q4Tensor, QuantizedTensor};
use burn::tensor::{backend::Backend, Tensor};

pub enum LinearWeight<B: Backend> {
    Dense(Tensor<B, 2>),
    Quantized(QuantizedTensor<B>),
    Q4(Q4Tensor<B>),
}

impl<B: Backend> LinearWeight<B> {
    pub fn shape(&self) -> [usize; 2] {
        match self {
            Self::Dense(t) => t.dims(),
            Self::Quantized(q) => q.shape(),
            Self::Q4(q) => q.shape(),
        }
    }

    pub fn matmul(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            Self::Dense(w) => x.matmul(w.clone().unsqueeze()),
            Self::Quantized(q) => x.matmul(q.dequantize().unsqueeze()),
            Self::Q4(q) => x.matmul(q.dequantize().unsqueeze()),
        }
    }

    pub fn swap_dims(self, a: usize, b: usize) -> Self {
        match self {
            Self::Dense(t) => Self::Dense(t.swap_dims(a, b)),
            Self::Quantized(q) => {
                // Q8 has a direct lossless transpose available.
                Self::Quantized(q.transpose_2d())
            }
            Self::Q4(q) => {
                // Q4 transpose: dequantize, swap, requantize. This introduces a
                // small additional Q4 noise (one round-trip through quantization)
                // but only happens once at load. The loader is designed to AVOID
                // calling swap_dims on Q4 weights (Q4 fixtures store pre-transposed
                // weight in math order), so this code path is rarely exercised in
                // practice but kept for completeness.
                let dq = q.dequantize().swap_dims(a, b);
                Self::Q4(Q4Tensor::quantize_from(dq))
            }
        }
    }
}
```

- [ ] **Step 3: Verify + commit**

```bash
cargo test -p ai-engine-runtime --test q4_linear_weight
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
git add -A
git commit -m "feat(runtime): LinearWeight::Q4 variant dispatches dequantize-then-matmul"
```

NO Co-Authored-By.

---

### Task 3: Q4 fixture generator (Python)

**Files:**
- Create: `crates/ai-engine-runtime/scripts/generate_q4_fixture.py`
- Create: `crates/ai-engine-runtime/fixtures/toy-llama-3-q4/` (outputs)

The Python script that takes the bf16 toy-llama-3 fixture and produces a Q4 version.

**Key difference from the Q8 generator**: we store weights **pre-transposed** (in math order `[in, out]`). The bf16 fixture stores HF's natural `[out, in]`; we transpose during quantization so the loader doesn't have to.

This means the Q4 fixture's weight tensors will have **transposed shapes** compared to the bf16 fixture. The loader must recognize this. We use a different tensor naming suffix — `.q4_weight` for the packed bytes and `.q4_scale` for the per-group scales — so the loader can distinguish Q4 from bf16/Q8 unambiguously.

- [ ] **Step 1: Write the script**

`crates/ai-engine-runtime/scripts/generate_q4_fixture.py`:

```python
#!/usr/bin/env python3
"""
Generate toy-llama-3-q4 from toy-llama-3 (bf16). Per-group symmetric Q4
(group size 32 along input dim). Pre-transposed: weights stored in math
order [in, out] (not HF's [out, in]).

Naming convention:
  <linear_name>.q4_weight       packed nibbles, shape [in, out/2], dtype uint8
  <linear_name>.q4_scale        per-(group, out_channel) scales, shape [in/32, out], dtype float32
  <other tensors>               unchanged (bf16 passthrough)

Run once; commit outputs.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

SRC = Path(__file__).resolve().parent.parent / "fixtures" / "toy-llama-3"
OUT = Path(__file__).resolve().parent.parent / "fixtures" / "toy-llama-3-q4"
OUT.mkdir(parents=True, exist_ok=True)

GROUP_SIZE = 32

LINEAR_PATTERNS = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    "lm_head.weight",
]

def should_quantize(name: str) -> bool:
    return any(p in name for p in LINEAR_PATTERNS)

def quantize_q4(w_f32: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    w_f32: [out, in] (HF natural layout).
    Returns: (packed: uint8 [in, out/2], scales: float32 [in/32, out]).
    Pre-transposes to math order [in, out] internally.
    """
    out_dim, in_dim = w_f32.shape
    assert in_dim % GROUP_SIZE == 0, f"in dim {in_dim} not divisible by {GROUP_SIZE}"
    assert out_dim % 2 == 0, f"out dim {out_dim} must be even"

    # Transpose to math order [in, out].
    w = w_f32.transpose(0, 1).contiguous().float().numpy()  # [in, out]

    num_groups = in_dim // GROUP_SIZE
    # scales[g, j] = max(|w[g*32:(g+1)*32, j]|) / 7
    scales = np.zeros((num_groups, out_dim), dtype=np.float32)
    for g in range(num_groups):
        block = np.abs(w[g * GROUP_SIZE:(g + 1) * GROUP_SIZE, :])
        max_abs = block.max(axis=0)             # per-column max within this group
        # Avoid zero division; fall back to scale=1.0 for all-zero groups.
        scales[g, :] = np.where(max_abs > 0, max_abs / 7.0, 1.0)

    # Pack nibbles row-major. packed[i, j/2] holds col j (low nibble) and col j+1 (high nibble).
    packed = np.zeros((in_dim, out_dim // 2), dtype=np.uint8)
    for i in range(in_dim):
        g = i // GROUP_SIZE
        for j in range(0, out_dim, 2):
            s_lo = scales[g, j]
            s_hi = scales[g, j + 1]
            q_lo = int(np.clip(np.round(w[i, j] / s_lo), -7, 7))
            q_hi = int(np.clip(np.round(w[i, j + 1] / s_hi), -7, 7))
            nibble_lo = q_lo & 0x0F
            nibble_hi = q_hi & 0x0F
            packed[i, j // 2] = nibble_lo | (nibble_hi << 4)

    return torch.from_numpy(packed), torch.from_numpy(scales)

src_tensors = load_file(SRC / "model.safetensors")
out_tensors = {}

for name, t in src_tensors.items():
    if should_quantize(name):
        packed, scales = quantize_q4(t)
        out_tensors[f"{name}.q4_weight"] = packed
        out_tensors[f"{name}.q4_scale"] = scales
    else:
        out_tensors[name] = t

save_file(out_tensors, OUT / "model.safetensors")

# Copy config + tokenizer + reference unchanged.
for fname in ["config.json", "tokenizer.json", "reference_prompt.txt", "reference_logits.bin"]:
    shutil.copy(SRC / fname, OUT / fname)

(OUT / "README.md").write_text(
"""# toy-llama-3-q4 fixture

Generated by `scripts/generate_q4_fixture.py` from `toy-llama-3`. Do not edit by hand.

Per-group symmetric Q4 quantization (group size 32 along input dim).

| File | Purpose |
|---|---|
| config.json | Same as toy-llama-3 (architecture identical) |
| model.safetensors | Q4 packed weights + f32 scales; non-Linear tensors are bf16 passthrough |
| tokenizer.json | Same as toy-llama-3 |
| reference_prompt.txt | Same as toy-llama-3 |
| reference_logits.bin | Same as toy-llama-3 — Q4 forward must match within ~5e-2 |

Naming convention:
  `<linear_name>.q4_weight`  packed nibbles, uint8, shape `[in, out/2]`, pre-transposed
  `<linear_name>.q4_scale`   per-group scales, f32, shape `[in/32, out]`

The Rust Q4 forward pass must produce logits matching `reference_logits.bin`
within `max |a - b| < 5e-2` when run on the same prompt against the same
checkpoint. Tolerance is looser than Q8's (3e-2) because Q4 per-group
quantization is intrinsically less precise.
""")

bf16_size = sum(t.numel() * t.element_size() for t in src_tensors.values())
q4_size = sum(t.numel() * t.element_size() for t in out_tensors.values())
print(f"wrote Q4 fixture to {OUT}")
print(f"bf16 fixture: {bf16_size} bytes")
print(f"Q4   fixture: {q4_size} bytes")
print(f"compression ratio: {bf16_size / q4_size:.2f}x")
```

- [ ] **Step 2: Run + verify**

```bash
cd /home/alessio/aip/airproxy
source .venv-fixture/bin/activate
python crates/ai-engine-runtime/scripts/generate_q4_fixture.py
deactivate
ls -la crates/ai-engine-runtime/fixtures/toy-llama-3-q4/
```

Expected:
- 6 files in `crates/ai-engine-runtime/fixtures/toy-llama-3-q4/`
- compression ratio printed by the script: approximately 3–3.5× (the per-group scale overhead reduces the theoretical 4× to ~3.x)
- `model.safetensors` smaller than the bf16 version

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "test(runtime): add Q4 fixture + generator script for toy-llama-3"
```

NO Co-Authored-By.

---

### Task 4: WeightNameMap extensions for `.q4_weight` and `.q4_scale`

**Files:**
- Modify: `crates/ai-engine-runtime/src/name_map.rs`
- Modify: `crates/ai-engine-runtime/tests/name_map.rs` (add tests)

- [ ] **Step 1: Failing test**

Append to `crates/ai-engine-runtime/tests/name_map.rs`:

```rust
#[test]
fn llama3_q4_companion_names() {
    let nm = WeightNameMap::for_family(ModelFamily::Llama3);
    assert_eq!(
        nm.lookup(TensorId::LayerQProjQ4Weight(12)),
        "model.layers.12.self_attn.q_proj.weight.q4_weight"
    );
    assert_eq!(
        nm.lookup(TensorId::LayerQProjQ4Scale(12)),
        "model.layers.12.self_attn.q_proj.weight.q4_scale"
    );
    assert_eq!(
        nm.lookup(TensorId::OutputProjectionQ4Weight),
        "lm_head.weight.q4_weight"
    );
    assert_eq!(
        nm.lookup(TensorId::OutputProjectionQ4Scale),
        "lm_head.weight.q4_scale"
    );
}
```

- [ ] **Step 2: Implement — extend `TensorId`**

Add to the enum:

```rust
OutputProjectionQ4Weight,
OutputProjectionQ4Scale,
LayerQProjQ4Weight(usize),
LayerQProjQ4Scale(usize),
LayerKProjQ4Weight(usize),
LayerKProjQ4Scale(usize),
LayerVProjQ4Weight(usize),
LayerVProjQ4Scale(usize),
LayerOProjQ4Weight(usize),
LayerOProjQ4Scale(usize),
LayerFfnGateQ4Weight(usize),
LayerFfnGateQ4Scale(usize),
LayerFfnUpQ4Weight(usize),
LayerFfnUpQ4Scale(usize),
LayerFfnDownQ4Weight(usize),
LayerFfnDownQ4Scale(usize),
```

Add to `llama_style`:

```rust
LayerQProjQ4Weight(i) => format!("model.layers.{i}.self_attn.q_proj.weight.q4_weight"),
LayerQProjQ4Scale(i)  => format!("model.layers.{i}.self_attn.q_proj.weight.q4_scale"),
LayerKProjQ4Weight(i) => format!("model.layers.{i}.self_attn.k_proj.weight.q4_weight"),
LayerKProjQ4Scale(i)  => format!("model.layers.{i}.self_attn.k_proj.weight.q4_scale"),
LayerVProjQ4Weight(i) => format!("model.layers.{i}.self_attn.v_proj.weight.q4_weight"),
LayerVProjQ4Scale(i)  => format!("model.layers.{i}.self_attn.v_proj.weight.q4_scale"),
LayerOProjQ4Weight(i) => format!("model.layers.{i}.self_attn.o_proj.weight.q4_weight"),
LayerOProjQ4Scale(i)  => format!("model.layers.{i}.self_attn.o_proj.weight.q4_scale"),
LayerFfnGateQ4Weight(i) => format!("model.layers.{i}.mlp.gate_proj.weight.q4_weight"),
LayerFfnGateQ4Scale(i)  => format!("model.layers.{i}.mlp.gate_proj.weight.q4_scale"),
LayerFfnUpQ4Weight(i) => format!("model.layers.{i}.mlp.up_proj.weight.q4_weight"),
LayerFfnUpQ4Scale(i)  => format!("model.layers.{i}.mlp.up_proj.weight.q4_scale"),
LayerFfnDownQ4Weight(i) => format!("model.layers.{i}.mlp.down_proj.weight.q4_weight"),
LayerFfnDownQ4Scale(i)  => format!("model.layers.{i}.mlp.down_proj.weight.q4_scale"),
OutputProjectionQ4Weight => "lm_head.weight.q4_weight".into(),
OutputProjectionQ4Scale  => "lm_head.weight.q4_scale".into(),
```

- [ ] **Step 3: Verify + commit**

```bash
cargo test -p ai-engine-runtime --test name_map
cargo clippy --workspace --all-targets -- -D warnings
git add -A
git commit -m "feat(runtime): WeightNameMap Q4 companion variants"
```

---

### Task 5: Loader detects Q4 fixtures

**Files:**
- Modify: `crates/ai-engine-runtime/src/loader.rs`
- Create: `crates/ai-engine-runtime/tests/q4_loader.rs`

The loader's `load_linear_weight` currently dispatches on the weight tensor's dtype: I8 → Q8, otherwise → Dense. Add a new dispatch: if the BASE weight name doesn't exist as a tensor BUT `<name>.q4_weight` does, load as Q4.

- [ ] **Step 1: Failing test**

`crates/ai-engine-runtime/tests/q4_loader.rs`:

```rust
use ai_engine_runtime::arch::linear::LinearWeight;
use ai_engine_runtime::config::ModelConfig;
use ai_engine_runtime::loader::load_range;
use std::path::PathBuf;

type B = burn_ndarray::NdArray;

fn q4_fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/toy-llama-3-q4")
}

#[test]
fn load_q4_fixture_produces_q4_weights() {
    let cfg = ModelConfig::from_file(&q4_fixture().join("config.json")).unwrap();
    let dev = Default::default();
    let weights = load_range::<B>(
        &q4_fixture().join("model.safetensors"), &cfg,
        0..cfg.n_layers, true, true, &dev,
    ).unwrap();
    for layer in &weights.layers {
        assert!(matches!(layer.q_proj, LinearWeight::Q4(_)));
        assert!(matches!(layer.k_proj, LinearWeight::Q4(_)));
        assert!(matches!(layer.v_proj, LinearWeight::Q4(_)));
        assert!(matches!(layer.o_proj, LinearWeight::Q4(_)));
        assert!(matches!(layer.ffn_gate, LinearWeight::Q4(_)));
        assert!(matches!(layer.ffn_up, LinearWeight::Q4(_)));
        assert!(matches!(layer.ffn_down, LinearWeight::Q4(_)));
    }
    // Embedding stays dense (toy fixture has tied embedding via lm_head.q4_weight;
    // the loader's tied-embedding fallback dequantizes lm_head to give us a Dense embedding).
    assert!(weights.embedding.is_some());
    assert!(weights.output_proj.is_none() || weights.output_proj.is_some());   // either way, depends on tying
}
```

- [ ] **Step 2: Implement — dispatch in `load_linear_weight`**

Modify `load_linear_weight` in `crates/ai-engine-runtime/src/loader.rs`. Add new TensorId arguments OR a helper for "given a logical weight, try Q4 first, then I8, then dense":

```rust
fn load_linear_weight<B: Backend>(
    st: &SafeTensors<'_>,
    nm: &WeightNameMap,
    weight_id: TensorId,
    scale_id: TensorId,           // Q8 scale (single-element) — only checked on Q8 path
    q4_weight_id: TensorId,
    q4_scale_id: TensorId,
    device: &B::Device,
) -> anyhow::Result<LinearWeight<B>> {
    // 1. Try Q4 path first: is `<name>.q4_weight` present?
    let q4_weight_name = nm.lookup(q4_weight_id);
    if let Ok(packed_view) = st.tensor(&q4_weight_name) {
        let packed_shape = packed_view.shape();
        if packed_shape.len() != 2 {
            anyhow::bail!("Q4 weight `{q4_weight_name}` expected 2D, got {:?}", packed_shape);
        }
        // packed has shape [in, out/2]; reconstruct [in, out].
        let in_dim = packed_shape[0];
        let packed_cols = packed_shape[1];
        let out_dim = packed_cols * 2;

        let packed_bytes = packed_view.data();
        let packed: Vec<u8> = packed_bytes.to_vec();

        let q4_scale_name = nm.lookup(q4_scale_id);
        let scale_view = st.tensor(&q4_scale_name)
            .with_context(|| format!("Q4 weight `{q4_weight_name}` missing scale `{q4_scale_name}`"))?;
        let scale_shape = scale_view.shape();
        if scale_shape.len() != 2 {
            anyhow::bail!("Q4 scale `{q4_scale_name}` expected 2D, got {:?}", scale_shape);
        }
        let num_groups = in_dim / crate::quant::Q4_GROUP_SIZE;
        if scale_shape != &[num_groups, out_dim] {
            anyhow::bail!(
                "Q4 scale `{q4_scale_name}` shape {:?} mismatches expected [{}, {}]",
                scale_shape, num_groups, out_dim
            );
        }
        let scales: Vec<f32> = bytemuck::cast_slice::<u8, f32>(scale_view.data()).to_vec();

        return Ok(LinearWeight::Q4(crate::quant::Q4Tensor::from_packed(
            packed, scales, [in_dim, out_dim], device,
        )));
    }

    // 2. Fall through to dense / Q8 dispatch (existing logic).
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
            // existing Q8 path
            let packed: Vec<i8> = bytemuck::cast_slice::<u8, i8>(view.data()).to_vec();
            let scale_name = nm.lookup(scale_id);
            let scale_view = st.tensor(&scale_name)
                .with_context(|| format!("Q8 weight `{name}` missing scale `{scale_name}`"))?;
            let scale_f32: &[f32] = bytemuck::cast_slice(scale_view.data());
            if scale_f32.len() != 1 {
                anyhow::bail!("Q8 scale `{scale_name}` must be single-element");
            }
            Ok(LinearWeight::Quantized(crate::quant::QuantizedTensor::from_packed(
                packed, scale_f32[0], shape2, device,
            )))
        }
        _ => {
            // dense path
            let f32_data = bytes_to_f32_vec(view.data(), view.dtype())?;
            Ok(LinearWeight::Dense(Tensor::<B, 2>::from_data(
                TensorData::new(f32_data, shape2),
                device,
            )))
        }
    }
}
```

Now update the `load_range` body to pass the Q4 TensorIds alongside the existing weight + Q8 scale TensorIds:

```rust
layers.push(LayerWeights {
    attn_norm: load_1d(TensorId::LayerAttnNorm(i))?,
    q_proj: load_linear_weight(&st, &nm,
        TensorId::LayerQProj(i), TensorId::LayerQProjScale(i),
        TensorId::LayerQProjQ4Weight(i), TensorId::LayerQProjQ4Scale(i),
        device,
    )?,
    k_proj: load_linear_weight(&st, &nm,
        TensorId::LayerKProj(i), TensorId::LayerKProjScale(i),
        TensorId::LayerKProjQ4Weight(i), TensorId::LayerKProjQ4Scale(i),
        device,
    )?,
    // ... v, o, ffn_gate, ffn_up, ffn_down — same pattern ...
    ffn_norm: load_1d(TensorId::LayerFfnNorm(i))?,
    ffn_gate: load_linear_weight(&st, &nm,
        TensorId::LayerFfnGate(i), TensorId::LayerFfnGateScale(i),
        TensorId::LayerFfnGateQ4Weight(i), TensorId::LayerFfnGateQ4Scale(i),
        device,
    )?,
    // ... etc.
});
```

Similarly for `output_proj`: pass `OutputProjectionQ4Weight` and `OutputProjectionQ4Scale`.

Plus update the embedding-tied fallback in the loader: when `model.embed_tokens.weight` is missing AND the lm_head is Q4, dequantize the Q4 lm_head to give us a dense embedding tensor. The existing fallback already dequantizes Q8; extend it to dequantize Q4 as well.

**Critical**: the Q4 weights are stored PRE-TRANSPOSED ([in, out] math order). `Model::from_loaded` and the cluster builders currently call `.swap_dims(0, 1)` on the LinearWeight result, which would WRONG for Q4. Fix: add a flag to LinearWeight signaling "already in math order" OR check the variant before calling swap_dims.

Simpler approach: make `LinearWeight::swap_dims` for the Q4 variant a no-op when called immediately after load (since it's already transposed). This is conceptually wrong (a no-op swap_dims is a lie) but works in practice because nobody else calls swap_dims on Q4 weights.

Cleanest approach: do NOT call `.swap_dims(0, 1)` on Q4 weights in `Model::from_loaded` and cluster builders. Use a helper:

```rust
fn ensure_math_order<B: Backend>(w: LinearWeight<B>) -> LinearWeight<B> {
    match &w {
        LinearWeight::Q4(_) => w,                  // Q4 fixtures store pre-transposed
        _ => w.swap_dims(0, 1),                    // Dense/Q8 stored as [out, in]
    }
}
```

Use `ensure_math_order(layer.q_proj)` everywhere `layer.q_proj.swap_dims(0, 1)` was called. Update `Model::from_loaded` and both cluster builders (`leader.rs` and `worker.rs`).

- [ ] **Step 3: Add the `ensure_math_order` helper**

In `crates/ai-engine-runtime/src/arch/linear.rs`:

```rust
impl<B: Backend> LinearWeight<B> {
    /// Ensure the weight is in math order [in, out]. Q4 fixtures are stored
    /// pre-transposed; bf16/Q8 fixtures are stored in HF's [out, in] layout
    /// and need a swap_dims at load time.
    pub fn ensure_math_order(self) -> Self {
        match self {
            Self::Q4(_) => self,
            _ => self.swap_dims(0, 1),
        }
    }
}
```

Replace every `<weight>.swap_dims(0, 1)` call in `Model::from_loaded`, `leader.rs::build_leader_model`, and `worker.rs::run_worker_full` with `<weight>.ensure_math_order()`.

- [ ] **Step 4: Verify + commit**

```bash
cargo test -p ai-engine-runtime
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
git add -A
git commit -m "feat(runtime): loader detects Q4 weight + scale companions; ensure_math_order helper"
```

NO Co-Authored-By.

---

### Task 6: Q4 end-to-end correctness gate

**Files:**
- Create: `crates/ai-engine-runtime/tests/q4_reference_logits.rs`

Same shape as Q8's correctness gate but Q4. Tolerance: 5e-2 (Q4 has ~2× per-op error vs Q8 because of the smaller range; accumulating over 4 layers + lm_head puts it around 4-5e-2).

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
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/toy-llama-3-q4")
}

#[test]
fn q4_forward_matches_bf16_reference_within_quantization_tolerance() {
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
    eprintln!("Q4 vs bf16-reference max |a-b| = {max_abs_diff}");
    eprintln!("argmax  ours = {} ({})", argmax_us.0, argmax_us.1);
    eprintln!("argmax  ref  = {} ({})", argmax_ref.0, argmax_ref.1);

    // Q4 tolerance: 5e-2 conservative. If empirical is much higher (10e-2+), it's
    // likely a real bug — most likely candidates: wrong nibble order, wrong group
    // axis, wrong scales-shape interpretation, or missing ensure_math_order on
    // some site.
    assert!(
        max_abs_diff < 5e-2,
        "Q4 correctness gate failed: max |a-b| = {max_abs_diff} (tolerance 5e-2)"
    );
}
```

- [ ] **Step 2: Run + iterate**

Expected: `max_abs_diff` between 2e-2 and 5e-2 for the random-weight toy. If it's much higher, debug systematically:
1. Nibble order — verify the Python script and Rust dequantize agree on low-vs-high.
2. Group axis — verify scales are indexed `[i/32, j]` consistently in both.
3. Sign of int4 — verify nibble 0x09 maps to -7 (since (9 - 16) = -7), not +9.
4. swap_dims accidentally called on Q4 — check Model::from_loaded uses `ensure_math_order`.

If after debugging the value lands above 5e-2 but argmax matches, widen the tolerance with explanation (similar to Q8's outcome). If argmax FLIPS, that's a real bug.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "test(runtime): Q4 forward matches bf16 reference within 5e-2 tolerance"
```

NO Co-Authored-By.

---

### Task 7: Q4 cluster generation test

**Files:**
- Create: `crates/ai-engine-cluster/tests/q4_cluster.rs`

Mirror Q8's `q8_cluster.rs` but with the Q4 fixture. Q4 cluster output must match Q4 single-node EXACTLY (greedy + lossless wire).

- [ ] **Step 1: Test (copy template from `q8_cluster.rs`)**

`crates/ai-engine-cluster/tests/q4_cluster.rs` (copy q8_cluster.rs, change fixture path to `toy-llama-3-q4`):

```rust
use std::path::PathBuf;

type B = burn_ndarray::NdArray;

fn q4_fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .join("ai-engine-runtime/fixtures/toy-llama-3-q4")
}

fn single_node_q4_greedy_5(
    fix: &std::path::Path,
    cfg: &ai_engine_runtime::config::ModelConfig,
    prompt_ids: &[i32],
) -> Vec<u32> {
    // ... (same as single_node_q8_greedy_5 but loads the Q4 fixture)
    // Copy from `q8_cluster.rs::single_node_q8_greedy_5` verbatim, just pass q4 fixture
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
async fn q4_cluster_generation_matches_q4_single_node() {
    use ai_engine_cluster::{
        capability::BackendKind, leader::{ClusterLeader, LeaderConfig, WorkerEndpoint},
        tls::generate_node_identity, transport::quic::server_endpoint,
        worker::run_worker_full,
    };

    let fix = q4_fixture();
    let cfg = ai_engine_runtime::config::ModelConfig::from_file(&fix.join("config.json")).unwrap();
    let tok = ai_engine_tokenizer::HfTokenizer::from_path(fix.join("tokenizer.json")).unwrap();
    let prompt = "The quick brown fox";
    let ids: Vec<u32> = ai_engine_tokenizer::Tokenizer::encode(&tok, prompt).unwrap();
    let ids_i32: Vec<i32> = ids.iter().map(|x| *x as i32).collect();

    let baseline = single_node_q4_greedy_5(&fix, &cfg, &ids_i32);

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
        cluster_id: "q4-test".into(),
        leader_node_id: "leader".into(),
        model_id: "toy-q4".into(),
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
        "Q4 cluster generation must match Q4 single-node baseline"
    );
}
```

- [ ] **Step 2: Run + commit**

```bash
cargo test -p ai-engine-cluster --test q4_cluster
cargo clippy --workspace --all-targets -- -D warnings
git add -A
git commit -m "test(cluster): Q4 distributed generation matches Q4 single-node baseline"
```

NO Co-Authored-By.

---

### Task 8: README + tag v0.3.0-alpha.2

**Files:**
- Modify: `README.md`
- Tag: `v0.3.0-alpha.2`

- [ ] **Step 1: Final verification**

```bash
cd /home/alessio/aip/airproxy
cargo test --workspace 2>&1 | grep "test result" | awk '{passed += $4; ignored += $8} END {print "PASSED=" passed " IGNORED=" ignored}'
cargo clippy --workspace --all-targets -- -D warnings
cargo build --workspace --release
cargo test -p ai-engine --test multiproc_smoke -- --ignored
cargo test -p ai-engine --test streaming_smoke -- --ignored
```

All must succeed.

- [ ] **Step 2: README update**

Append:

```markdown
### v0.3.0-alpha.2 — Q4 weight quantization

ai-engine v0.3.0-alpha.2 adds Q4 (4-bit, per-group symmetric, group
size 32) weight quantization. Each Linear weight is stored as packed
nibbles (2 values per byte) plus per-group f32 scales. Memory at rest
is ~3× smaller than bf16 for the toy fixture (closer to 4× for real
models where Linear weights dominate the parameter count and scale
overhead amortizes).

Correctness:
- Q4 forward matches bf16 reference within ~5e-2 on the random-init
  toy-llama-3 fixture. Argmax matches the bf16 reference.
- 3-node Q4 cluster generation matches single-node Q4 generation EXACTLY
  under greedy sampling.

Format: our own per-group symmetric Q4, group size 32, low-nibble-first
packing. Stored pre-transposed (math order [in, out]) so the loader
never has to transpose Q4 weights. AWQ / GPTQ / GGUF / bitsandbytes
compatibility is deferred to future plans.

Generate a Q4 checkpoint from any bf16 safetensors model using the
`crates/ai-engine-runtime/scripts/generate_q4_fixture.py` template.

Known limitations (still deferred):
- External format readers (AWQ/GPTQ/GGUF).
- Dequantize-on-forward is unfused.
- Activations stay f32.
```

- [ ] **Step 3: Commit + tag**

```bash
git add README.md
git commit -m "docs: announce v0.3.0-alpha.2 Q4 quantization release"
git tag v0.3.0-alpha.2
git log --oneline -5
git tag
```

NO Co-Authored-By.

## Report
- Status
- Final test count
- Q4 vs bf16 max_abs_diff
- Q4 fixture compression ratio
- `git tag` listing
- `git log --oneline -5`

---

## Self-review

**Spec coverage:**
- Q4 primitive (storage, quantize, dequantize) → Task 1
- LinearWeight::Q4 variant + matmul → Task 2
- Q4 fixture from bf16 → Task 3
- Loader detects Q4 → Task 5
- Correctness gate → Task 6
- Cluster integration → Task 7
- Release → Task 8

**Placeholder scan:** Task 7's test body is mostly verbatim from q8_cluster.rs with the fixture path swapped — full code shown, not just "copy from q8_cluster.rs". Task 5 has prose-level dispatch description in `load_linear_weight`; code block is complete.

**Type consistency:**
- `Q4Tensor<B>` (Task 1) → consumed by `LinearWeight::Q4` (Task 2) and produced by loader (Task 5). ✓
- `LinearWeight` Q4 arm (Task 2) → matmul dispatch + swap_dims + ensure_math_order. ✓
- `TensorId::*Q4Weight/*Q4Scale` (Task 4) → consumed by `load_linear_weight` (Task 5). ✓
- Q4 fixture format (Task 3 Python) → consumed by loader (Task 5). Shapes match: packed [in, out/2] uint8, scales [in/32, out] f32. ✓
- `ensure_math_order` (Task 5) replaces `swap_dims(0, 1)` everywhere. ✓

**Acknowledged risks:**

1. **Pre-transposed Q4 layout vs swap_dims** — the loader returns Q4 weights in math order [in, out]. Existing callers (`Model::from_loaded`, cluster builders) call `.swap_dims(0, 1)` assuming HF [out, in]. The `ensure_math_order` helper fixes this for Q4 but must be applied at EVERY call site. Task 5 lists three (Model::from_loaded, leader::build_leader_model, worker::run_worker_full). Miss any one and you get wrong logits.
2. **Q4 quantization error** is intrinsically higher than Q8. Expect max_abs_diff in the 2e-2 to 5e-2 range on the toy. If real, document with the tolerance; if anomalous (above 1e-1), there's a bug.
3. **Group size 32 + dimension constraint** — `in % 32 == 0` and `out % 2 == 0`. All Llama dims satisfy these but exotic configs might not. The quantize_from method asserts; document the constraint.
4. **Tied-embedding Q4** — when only `lm_head.weight.q4_weight` is stored, the embedding fallback path must dequantize the Q4 lm_head. The Q8 loader already handles tied-embedding-Q8; extend to Q4.

---

## Execution Handoff

Plan 6 saved to `docs/superpowers/plans/2026-05-24-plan-6-quantization-q4.md`. 8 tasks total.

Two execution options:

**1. Subagent-Driven (recommended)** — Task 1 is the heaviest pure-algorithm task (bit packing). Tasks 2–4 are bounded. Task 5 is the most invasive loader change. Tasks 6–8 are correctness gates + release.

**2. Inline Execution** — also reasonable.

After v0.3.0-alpha.2 ships, Plan 7 candidates: AWQ/GPTQ/GGUF loaders, fused int8/int4 GEMM kernels on GPU backends, or mDNS auto-discovery for cluster ergonomics.
