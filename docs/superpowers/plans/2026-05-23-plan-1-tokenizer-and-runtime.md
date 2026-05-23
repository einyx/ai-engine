# Plan 1 — `ai-engine-tokenizer` + `ai-engine-runtime`

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship two new crates that, together, allow `ai-engine` to load a Hugging Face safetensors checkpoint and run a forward pass that produces logits bytes-tolerant-identical to `transformers` on a toy Llama-3-style model. No cluster yet; this is single-node inference correctness.

**Architecture:** `ai-engine-tokenizer` wraps the `tokenizers` crate to encode/decode strings. `ai-engine-runtime` defines a parameterized transformer (RMSNorm + GQA + RoPE + SwiGLU) generic over `burn::Backend`, plus a safetensors loader with per-layer range support so future cluster workers can load only their assigned layers. The crucial gate is a bytes-tolerant correctness test against a precomputed `transformers` reference on a tiny in-tree fixture (~32M params, 4 layers, hidden 256, GQA 4→2). If that test passes, the math is right.

**Tech Stack:** `burn` (multi-backend ML library — ndarray + cuda + metal + wgpu backends), `safetensors` (mmap-friendly weight format), `tokenizers` (HF BPE/SentencePiece), `bytemuck` (raw-bytes deserialization of f32 reference logits), Python (one-time fixture generation only; not a runtime or CI dep).

**Scope rule:** This plan ships single-node inference. NO cluster, NO QUIC, NO partitioner. Plan 2 owns those.

**Baseline:** Branch `chore/rename-to-ai-engine` at `v0.1.1` (8 crates, 78 tests, clippy clean).

---

## File structure (locked in here)

```
crates/
├── ai-engine-tokenizer/                # NEW (Tasks 1–2)
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                      # pub trait Tokenizer + pub use HfTokenizer
│       └── hf.rs                       # HfTokenizer (wraps `tokenizers` crate)
│
└── ai-engine-runtime/                  # NEW (Tasks 3–18)
    ├── Cargo.toml
    ├── fixtures/
    │   └── toy-llama-3/
    │       ├── config.json             # ModelConfig in HF JSON layout
    │       ├── model.safetensors       # ~40 MB at bf16
    │       ├── tokenizer.json          # HF tokenizer dump
    │       ├── reference_logits.bin    # [vocab_size] f32, precomputed by transformers
    │       ├── reference_prompt.txt    # the input prompt used to generate the reference
    │       └── README.md               # how to regenerate (one-time script)
    ├── scripts/
    │   └── generate_toy_fixture.py     # one-time script; NOT a build dep
    └── src/
        ├── lib.rs                      # public surface
        ├── config.rs                   # ModelConfig + HF config.json parsing
        ├── name_map.rs                 # weight-name registry per ModelFamily
        ├── backend.rs                  # BackendKind + device factory functions
        ├── kv_cache.rs                 # KvCacheSlot<B>
        ├── loader.rs                   # safetensors → LoadedWeights<B>; load_range
        ├── arch/
        │   ├── mod.rs                  # module declarations
        │   ├── rope.rs                 # precomputed cos/sin, apply_rope
        │   ├── rmsnorm.rs              # RMSNorm
        │   ├── ffn.rs                  # SwiGLU FFN
        │   ├── attention.rs            # GQA attention with KV cache
        │   ├── embedding.rs            # token embedding + output projection
        │   ├── block.rs                # one decoder block (norm → attn → res → norm → ffn → res)
        │   └── model.rs                # Model<B>: stack of blocks + final norm + output proj
        └── sample.rs                   # greedy / temperature / top-p / top-k
```

File responsibility principle: each `arch/*.rs` file owns one transformer primitive, exposes a small constructor + forward method, and is independently testable. The biggest file (`model.rs`) is ~150 lines because it's just wiring.

---

## Burn version pinning

The plan assumes a recent stable `burn` (likely 0.18+ at the time of execution). The implementer **must verify the actual API** for each operation via:

1. `cargo doc --open -p burn` after the first dep add, OR
2. Context7: `mcp__plugin_context7_context7__resolve-library-id` → `burn-rs/burn`, then `query-docs` for specific operations, OR
3. The burn book at https://burn.dev/book/

The plan documents *what* each function does and the test that verifies it; the implementer writes the burn-specific calls. If burn's API has shifted, the plan's algorithm is still correct — only the call syntax changes.

---

### Task 1: `ai-engine-tokenizer` — crate scaffold + trait + test

**Files:**
- Create: `crates/ai-engine-tokenizer/Cargo.toml`
- Create: `crates/ai-engine-tokenizer/src/lib.rs`
- Create: `crates/ai-engine-tokenizer/src/hf.rs`
- Create: `crates/ai-engine-tokenizer/tests/roundtrip.rs`
- Modify: root `Cargo.toml` (add to `[workspace.dependencies]`)

- [ ] **Step 1: Add workspace deps**

Append to root `Cargo.toml` `[workspace.dependencies]`:

```toml
tokenizers = { version = "0.20", default-features = false, features = ["onig"] }
ai-engine-tokenizer = { path = "crates/ai-engine-tokenizer" }
```

Verify with `cargo metadata --no-deps 2>&1 | head -3` — should succeed.

- [ ] **Step 2: Crate skeleton**

`crates/ai-engine-tokenizer/Cargo.toml`:

```toml
[package]
name = "ai-engine-tokenizer"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true

[dependencies]
anyhow.workspace = true
tokenizers.workspace = true
```

- [ ] **Step 3: Write failing test FIRST**

`crates/ai-engine-tokenizer/tests/roundtrip.rs`:

```rust
use ai_engine_tokenizer::{HfTokenizer, Tokenizer};

#[test]
fn encode_then_decode_roundtrips_text() {
    // Uses a checked-in tests/fixtures/tokenizer_tiny.json — see Step 3 below
    // for how it gets created. The same tokenizer.json gets reused as the toy
    // fixture's tokenizer in Task 4.
    let tok = HfTokenizer::from_path("tests/fixtures/tokenizer_tiny.json")
        .expect("load tokenizer");
    let text = "Hello, world!";
    let ids = tok.encode(text).unwrap();
    assert!(!ids.is_empty(), "encode produced tokens");
    let back = tok.decode(&ids).unwrap();
    // Trim because HF tokenizers can emit a leading space.
    assert_eq!(back.trim(), text);
}

#[test]
fn encode_handles_unicode() {
    let tok = HfTokenizer::from_path("tests/fixtures/tokenizer_tiny.json").unwrap();
    let _ids = tok.encode("café 日本").unwrap();   // just doesn't panic
}
```

To produce `tests/fixtures/tokenizer_tiny.json`, the implementer either:

(a) Generates it with `tokenizers` in Python:

```bash
python -c "
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
tok = Tokenizer(BPE())
tok.pre_tokenizer = ByteLevel()
tok.save('crates/ai-engine-tokenizer/tests/fixtures/tokenizer_tiny.json')
"
```

(b) Downloads a minimal BPE tokenizer.json from a public HF model (e.g., `gpt2`) — the JSON config is data, not code.

- [ ] **Step 4: Confirm test fails**

```bash
cd /home/alessio/aip/airproxy
cargo test -p ai-engine-tokenizer
# Expected: compile error — HfTokenizer doesn't exist yet
```

- [ ] **Step 5: Implement `lib.rs` + `hf.rs`**

`crates/ai-engine-tokenizer/src/lib.rs`:

```rust
//! ai-engine-tokenizer

mod hf;

pub use hf::HfTokenizer;

/// Minimal tokenizer surface: just encode / decode. No special-token handling
/// in v0.2 — that's the caller's job. Add traits like `bos_token_id()` etc.
/// when v0.3 needs them.
pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>>;
    fn decode(&self, ids: &[u32]) -> anyhow::Result<String>;
    fn vocab_size(&self) -> usize;
}
```

`crates/ai-engine-tokenizer/src/hf.rs`:

```rust
use crate::Tokenizer;
use std::path::Path;

pub struct HfTokenizer {
    inner: tokenizers::Tokenizer,
}

impl HfTokenizer {
    pub fn from_path<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path.as_ref())
            .map_err(|e| anyhow::anyhow!("load tokenizer.json: {e}"))?;
        Ok(Self { inner })
    }
}

impl Tokenizer for HfTokenizer {
    fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        let enc = self.inner.encode(text, /*add_special_tokens=*/false)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    fn decode(&self, ids: &[u32]) -> anyhow::Result<String> {
        self.inner.decode(ids, /*skip_special_tokens=*/false)
            .map_err(|e| anyhow::anyhow!("decode: {e}"))
    }

    fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(/*with_added_tokens=*/true)
    }
}
```

- [ ] **Step 6: Verify**

```bash
cargo test -p ai-engine-tokenizer
cargo clippy --workspace --all-targets -- -D warnings
```

Expect: 2 tests pass; clippy clean.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat(tokenizer): HfTokenizer wrapping the tokenizers crate"
```

NO Co-Authored-By footer (global preference).

---

### Task 2: Tokenizer family detection helper

**Files:**
- Modify: `crates/ai-engine-tokenizer/src/lib.rs`
- Modify: `crates/ai-engine-tokenizer/src/hf.rs`
- Add: `crates/ai-engine-tokenizer/tests/specials.rs`

The runtime needs to know BOS / EOS token ids per family for generation. v0.2 supports four families (Llama3, Mistral, Qwen25, DeepSeekV2). Each has known special-token ids; we hard-code them via a family enum rather than try to auto-detect from the tokenizer config (HF tokenizers don't always store this consistently).

- [ ] **Step 1: Write failing test**

```rust
use ai_engine_tokenizer::{ModelFamily, SpecialTokens};

#[test]
fn llama3_specials() {
    let s = SpecialTokens::for_family(ModelFamily::Llama3);
    assert_eq!(s.bos_token_id, 128000);   // Llama-3 <|begin_of_text|>
    assert_eq!(s.eos_token_id, 128001);   // Llama-3 <|end_of_text|>
}

#[test]
fn qwen25_specials() {
    let s = SpecialTokens::for_family(ModelFamily::Qwen25);
    assert_eq!(s.bos_token_id, 151643);   // Qwen 2.5 <|endoftext|> used as BOS
    assert_eq!(s.eos_token_id, 151645);   // Qwen 2.5 <|im_end|>
}
```

- [ ] **Step 2: Implement**

`crates/ai-engine-tokenizer/src/lib.rs` (additions):

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    Llama3,
    Mistral,
    Qwen25,
    DeepSeekV2,
}

#[derive(Debug, Clone, Copy)]
pub struct SpecialTokens {
    pub bos_token_id: u32,
    pub eos_token_id: u32,
}

impl SpecialTokens {
    pub fn for_family(family: ModelFamily) -> Self {
        match family {
            ModelFamily::Llama3 => Self { bos_token_id: 128000, eos_token_id: 128001 },
            // Mistral 7B v0.1: BOS=1, EOS=2 (sentencepiece default)
            ModelFamily::Mistral => Self { bos_token_id: 1, eos_token_id: 2 },
            // Qwen 2.5: uses <|endoftext|> (151643) for BOS, <|im_end|> (151645) for EOS in chat
            ModelFamily::Qwen25 => Self { bos_token_id: 151643, eos_token_id: 151645 },
            // DeepSeek V2: matches Llama tokenizer family with custom ids
            ModelFamily::DeepSeekV2 => Self { bos_token_id: 100000, eos_token_id: 100001 },
        }
    }
}
```

- [ ] **Step 3: Verify + commit**

```bash
cargo test -p ai-engine-tokenizer
cargo clippy --workspace --all-targets -- -D warnings
git add -A
git commit -m "feat(tokenizer): ModelFamily + SpecialTokens for the four supported families"
```

The DeepSeek-V2 / Qwen-2.5 / Mistral token IDs above are best-effort defaults; the implementer should sanity-check against current HF model cards before committing. Note as a concern if any disagree.

---

### Task 3: `ai-engine-runtime` scaffold + ModelConfig + name_map

**Files:**
- Create: `crates/ai-engine-runtime/Cargo.toml`
- Create: `crates/ai-engine-runtime/src/{lib.rs, config.rs, name_map.rs, backend.rs}`
- Create: `crates/ai-engine-runtime/tests/config.rs`
- Modify: root `Cargo.toml` (add workspace deps)

- [ ] **Step 1: Root workspace deps**

Append to root `Cargo.toml` `[workspace.dependencies]`:

```toml
burn = { version = "0.18", default-features = false }
burn-ndarray = "0.18"
burn-cuda = "0.18"
burn-wgpu = "0.18"
# Metal backend: as of burn 0.18, Metal goes through burn-wgpu's Metal target.
# If a dedicated burn-metal crate exists at execution time, prefer it.
safetensors = "0.4"
bytemuck = { version = "1", features = ["derive"] }
half = "2"
memmap2 = "0.9"
ai-engine-runtime = { path = "crates/ai-engine-runtime" }
ai-engine-tokenizer = { path = "crates/ai-engine-tokenizer" }
```

The implementer must verify burn version + crate names against current crates.io. If burn 0.18 has been replaced by a newer version with breaking changes, pin to the version that compiles; document the choice in a commit message.

- [ ] **Step 2: Crate Cargo.toml**

`crates/ai-engine-runtime/Cargo.toml`:

```toml
[package]
name = "ai-engine-runtime"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true

[features]
default = ["backend-cpu", "backend-wgpu"]
# Multi-backend story — enable subset as desired.
backend-cpu  = ["dep:burn-ndarray"]
backend-cuda = ["dep:burn-cuda"]
backend-wgpu = ["dep:burn-wgpu"]
backend-metal = ["backend-wgpu"]  # Metal is exposed via wgpu in burn 0.18

[dependencies]
ai-engine-tokenizer.workspace = true
anyhow.workspace = true
bytemuck.workspace = true
burn = { workspace = true, features = ["std"] }
burn-ndarray = { workspace = true, optional = true }
burn-cuda    = { workspace = true, optional = true }
burn-wgpu    = { workspace = true, optional = true }
half.workspace = true
memmap2.workspace = true
safetensors.workspace = true
serde = { workspace = true }
serde_json.workspace = true
thiserror.workspace = true

[dev-dependencies]
tempfile = "3"
```

`backend-cuda` and `backend-metal` are NOT in `default` so CI on a generic Linux runner doesn't fail compiling them. CUDA needs a CUDA toolkit; the implementer enables it locally with `cargo build --features backend-cuda`.

- [ ] **Step 3: Failing test**

`crates/ai-engine-runtime/tests/config.rs`:

```rust
use ai_engine_runtime::config::{ModelConfig, ModelFamily};

const LLAMA3_8B_CONFIG: &str = r#"{
  "architectures": ["LlamaForCausalLM"],
  "hidden_size": 4096,
  "intermediate_size": 14336,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "vocab_size": 128256,
  "max_position_embeddings": 8192,
  "rope_theta": 500000.0,
  "rms_norm_eps": 1e-5,
  "tie_word_embeddings": false
}"#;

#[test]
fn parses_llama3_hf_config() {
    let cfg = ModelConfig::from_str(LLAMA3_8B_CONFIG).unwrap();
    assert_eq!(cfg.hidden_size, 4096);
    assert_eq!(cfg.n_layers, 32);
    assert_eq!(cfg.n_heads, 32);
    assert_eq!(cfg.n_kv_heads, 8);          // GQA: 32 → 8
    assert_eq!(cfg.head_dim, 128);          // 4096 / 32
    assert_eq!(cfg.family, ModelFamily::Llama3);
}

#[test]
fn rejects_mixtral_with_clear_message() {
    let mixtral = r#"{
      "architectures": ["MixtralForCausalLM"],
      "hidden_size": 4096,
      "num_hidden_layers": 32,
      "num_attention_heads": 32,
      "num_key_value_heads": 8,
      "vocab_size": 32000
    }"#;
    let err = ModelConfig::from_str(mixtral).unwrap_err();
    assert!(err.to_string().to_lowercase().contains("moe not supported"));
}

#[test]
fn computes_head_dim_when_not_in_json() {
    // HF configs sometimes omit head_dim; we compute hidden_size / n_heads.
    let cfg = ModelConfig::from_str(LLAMA3_8B_CONFIG).unwrap();
    assert_eq!(cfg.head_dim, 4096 / 32);
}
```

- [ ] **Step 4: Implement `config.rs`**

`crates/ai-engine-runtime/src/config.rs`:

```rust
use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily { Llama3, Mistral, Qwen25, DeepSeekV2 }

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub tie_word_embeddings: bool,
    pub family: ModelFamily,
}

#[derive(Deserialize)]
struct HfConfigJson {
    architectures: Vec<String>,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    #[serde(default)]
    num_key_value_heads: Option<usize>,
    #[serde(default)]
    head_dim: Option<usize>,
    vocab_size: usize,
    #[serde(default = "default_max_pos")]
    max_position_embeddings: usize,
    #[serde(default = "default_rope_theta")]
    rope_theta: f32,
    #[serde(default = "default_rms_eps")]
    rms_norm_eps: f32,
    #[serde(default)]
    tie_word_embeddings: bool,
}

fn default_max_pos() -> usize { 8192 }
fn default_rope_theta() -> f32 { 10000.0 }
fn default_rms_eps() -> f32 { 1e-6 }

impl ModelConfig {
    pub fn from_str(s: &str) -> anyhow::Result<Self> {
        let raw: HfConfigJson = serde_json::from_str(s)
            .map_err(|e| anyhow::anyhow!("config.json parse: {e}"))?;
        let family = detect_family(&raw.architectures)?;
        let n_kv_heads = raw.num_key_value_heads.unwrap_or(raw.num_attention_heads);
        let head_dim = raw.head_dim.unwrap_or(raw.hidden_size / raw.num_attention_heads);
        Ok(Self {
            hidden_size: raw.hidden_size,
            intermediate_size: raw.intermediate_size,
            n_layers: raw.num_hidden_layers,
            n_heads: raw.num_attention_heads,
            n_kv_heads,
            head_dim,
            vocab_size: raw.vocab_size,
            max_position_embeddings: raw.max_position_embeddings,
            rope_theta: raw.rope_theta,
            rms_norm_eps: raw.rms_norm_eps,
            tie_word_embeddings: raw.tie_word_embeddings,
            family,
        })
    }

    pub fn from_file(path: &std::path::Path) -> anyhow::Result<Self> {
        Self::from_str(&std::fs::read_to_string(path)?)
    }
}

fn detect_family(architectures: &[String]) -> anyhow::Result<ModelFamily> {
    for arch in architectures {
        let lc = arch.to_lowercase();
        if lc.contains("llama") { return Ok(ModelFamily::Llama3); }
        if lc.contains("mistral") { return Ok(ModelFamily::Mistral); }
        if lc.contains("qwen") { return Ok(ModelFamily::Qwen25); }
        if lc.contains("deepseek") { return Ok(ModelFamily::DeepSeekV2); }
        if lc.contains("mixtral") {
            anyhow::bail!("Mixtral / MoE not supported in v0.2 (architecture: {arch})");
        }
    }
    anyhow::bail!("unknown model architecture: {:?}", architectures)
}
```

- [ ] **Step 5: Empty stubs for the other modules** (so lib.rs compiles)

`crates/ai-engine-runtime/src/lib.rs`:

```rust
//! ai-engine-runtime

pub mod backend;
pub mod config;
pub mod name_map;

pub use config::{ModelConfig, ModelFamily};
```

`crates/ai-engine-runtime/src/backend.rs`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind { Cpu, Cuda, Metal, Wgpu }
```

`crates/ai-engine-runtime/src/name_map.rs`:

```rust
//! Maps logical weight tensor identifiers to HF safetensors names per family.
//! Filled in by Task 9.
```

- [ ] **Step 6: Verify + commit**

```bash
cargo test -p ai-engine-runtime
cargo clippy --workspace --all-targets -- -D warnings
git add -A
git commit -m "feat(runtime): ModelConfig + HF config.json loader with family detection"
```

---

### Task 4: Generate toy-llama-3 fixture (one-time Python script + checked-in artifacts)

**Files:**
- Create: `crates/ai-engine-runtime/scripts/generate_toy_fixture.py`
- Create: `crates/ai-engine-runtime/fixtures/toy-llama-3/{config.json, model.safetensors, tokenizer.json, reference_logits.bin, reference_prompt.txt, README.md}`
- Modify: `.gitignore`

This is the foundation of the entire bytes-tolerant correctness gate. Done once; outputs are checked into git.

**What we need to produce:**

| File | Format | What it is |
|---|---|---|
| `config.json` | HF config.json shape | A toy ModelConfig: hidden=256, n_layers=4, n_heads=4, n_kv_heads=2 (GQA), vocab=512, intermediate=512, max_pos=128, rope_theta=10000, rms_eps=1e-5, tied embeddings, family Llama3 |
| `model.safetensors` | safetensors bf16 | Random-but-seeded weights matching the config. Total ~3 MB at bf16. |
| `tokenizer.json` | HF tokenizer.json | A trivial byte-level BPE with 512 entries (chars + a few common merges). Same one used in Task 1. |
| `reference_prompt.txt` | UTF-8 | A fixed prompt string: `"The quick brown fox"` |
| `reference_logits.bin` | raw f32 LE | Logits for the FINAL token after one forward pass: [vocab_size]=512 f32 values. ~2 KB. |
| `README.md` | markdown | Notes on what the fixture is and how to regenerate. |

- [ ] **Step 1: Write the generation script**

`crates/ai-engine-runtime/scripts/generate_toy_fixture.py`:

```python
#!/usr/bin/env python3
"""
One-time fixture generator for ai-engine-runtime tests.

Produces a tiny Llama-3-style model and a reference forward-pass result
that the Rust implementation must match within tolerance.

Run once; commit the outputs. NOT a runtime/build dependency.

Requirements (one-time, in a Python venv):
    pip install torch==2.4 transformers==4.45 safetensors==0.4 tokenizers==0.20

Usage:
    cd crates/ai-engine-runtime
    python scripts/generate_toy_fixture.py

Output:
    fixtures/toy-llama-3/
"""

import json
import struct
from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer
from transformers import LlamaConfig, LlamaForCausalLM

OUT = Path(__file__).resolve().parent.parent / "fixtures" / "toy-llama-3"
OUT.mkdir(parents=True, exist_ok=True)

# Seed determinism — the reference must be reproducible.
torch.manual_seed(20260523)

# Toy config — small enough to ship in git, large enough to exercise GQA + RoPE.
cfg = LlamaConfig(
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,        # GQA: 4 -> 2
    vocab_size=512,
    max_position_embeddings=128,
    rope_theta=10000.0,
    rms_norm_eps=1e-5,
    tie_word_embeddings=True,
    torch_dtype="bfloat16",
)
cfg.architectures = ["LlamaForCausalLM"]

# Save config.json
with open(OUT / "config.json", "w") as f:
    json.dump(cfg.to_dict(), f, indent=2)

# Build the model with random weights at the seeded torch state.
# `.train(False)` sets PyTorch into inference mode (disables dropout etc.)
model = LlamaForCausalLM(cfg).to(torch.bfloat16)
model.train(False)

# Save weights in safetensors with standard HF names so our name_map can find them.
state = {k: v.contiguous() for k, v in model.state_dict().items()}
save_file(state, OUT / "model.safetensors")

# Build a minimal byte-level BPE tokenizer with 512 token vocab.
tok = Tokenizer(BPE(unk_token="<unk>"))
tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
tok.decoder = ByteLevelDecoder()
trainer = BpeTrainer(vocab_size=512, special_tokens=["<unk>"], min_frequency=1)
training_text = [
    "The quick brown fox jumps over the lazy dog.",
    "Hello, world! Foo bar baz.",
    "ai-engine runtime test fixture.",
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
]
tok.train_from_iterator(training_text, trainer)
tok.save(str(OUT / "tokenizer.json"))

# Pick the fixed reference prompt.
prompt = "The quick brown fox"
(OUT / "reference_prompt.txt").write_text(prompt)

# Tokenize and run the forward pass.
ids = tok.encode(prompt).ids
input_ids = torch.tensor([ids], dtype=torch.long)

with torch.no_grad():
    out = model(input_ids=input_ids, use_cache=False)

# Logits at the FINAL position. Shape: [1, seq_len, vocab_size]; take [:, -1, :].
logits = out.logits[0, -1, :].to(torch.float32)  # f32 for reference precision
arr = logits.numpy().tobytes(order="C")          # little-endian f32

with open(OUT / "reference_logits.bin", "wb") as f:
    f.write(arr)

# README
(OUT / "README.md").write_text(
"""# toy-llama-3 fixture

Generated by `scripts/generate_toy_fixture.py`. Do not edit by hand.

| File | Purpose |
|---|---|
| config.json | HF-format model config (4 layers, hidden 256, GQA 4->2, vocab 512) |
| model.safetensors | bf16 weights, seeded so regeneration is bit-identical |
| tokenizer.json | minimal byte-level BPE, 512 vocab |
| reference_prompt.txt | the input prompt used to compute reference_logits.bin |
| reference_logits.bin | raw little-endian f32 logits at the FINAL token position, length=vocab_size |

The Rust forward pass must produce logits matching `reference_logits.bin`
within `max |a - b| < 1e-3` when run on the same prompt against the same
safetensors. This is the bytes-tolerant correctness gate.
""")

print(f"wrote fixtures to {OUT}")
```

- [ ] **Step 2: Run the script once**

If a Python venv with the right deps isn't already set up, the implementer creates one:

```bash
cd /home/alessio/aip/airproxy
python3 -m venv .venv-fixture
source .venv-fixture/bin/activate
pip install torch==2.4 transformers==4.45 safetensors==0.4 tokenizers==0.20
python crates/ai-engine-runtime/scripts/generate_toy_fixture.py
deactivate
```

This generates the fixture files. The venv is local — add `.venv-fixture/` to `.gitignore`.

- [ ] **Step 3: Confirm fixture files exist + are sane**

```bash
ls -la crates/ai-engine-runtime/fixtures/toy-llama-3/
# Expected: config.json (~1 KB), model.safetensors (~3 MB), tokenizer.json (~50 KB),
#           reference_logits.bin (2048 bytes = 512 floats × 4), reference_prompt.txt, README.md
python3 -c "
import struct
with open('crates/ai-engine-runtime/fixtures/toy-llama-3/reference_logits.bin', 'rb') as f:
    data = f.read()
print(f'reference_logits.bin: {len(data)} bytes ({len(data)//4} floats)')
print(f'first 4 floats: {struct.unpack(\"4f\", data[:16])}')
"
```

Expected: 2048 bytes = 512 floats; finite (not NaN, not Inf) first values.

- [ ] **Step 4: Update .gitignore**

```bash
echo "" >> .gitignore
echo "# Python fixture-generation venv (not a runtime dep)" >> .gitignore
echo ".venv-fixture/" >> .gitignore
```

The fixture files themselves are committed to git (~3 MB total — acceptable for tests).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "test(runtime): add toy-llama-3 fixture + Python regeneration script

3 MB safetensors checkpoint + reference logits, generated deterministically
by scripts/generate_toy_fixture.py (one-time; not a build dep). The fixture
is the bytes-tolerant correctness gate for the burn implementation in
Tasks 6-12."
```

---

### Task 5: KV cache + RoPE primitives

**Files:**
- Create: `crates/ai-engine-runtime/src/kv_cache.rs`
- Create: `crates/ai-engine-runtime/src/arch/mod.rs`
- Create: `crates/ai-engine-runtime/src/arch/rope.rs`
- Create: `crates/ai-engine-runtime/tests/rope.rs`
- Create: `crates/ai-engine-runtime/tests/kv_cache.rs`

These are the two primitives that are independently testable without any model wiring.

- [ ] **Step 1: arch/mod.rs**

```rust
pub mod rope;
// Subsequent tasks add: rmsnorm, ffn, attention, embedding, block, model
```

- [ ] **Step 2: Failing test for RoPE shape + idempotence**

`crates/ai-engine-runtime/tests/rope.rs`:

```rust
use ai_engine_runtime::arch::rope::RotaryEmbedding;
use burn::tensor::Tensor;

type B = burn_ndarray::NdArray;

#[test]
fn rope_precomputes_correct_table_shape() {
    let dev = Default::default();
    let rope = RotaryEmbedding::<B>::new(
        /*head_dim=*/64, /*max_seq=*/128, /*theta=*/10000.0, &dev,
    );
    // Internal cos / sin tables: shape [max_seq, head_dim/2]
    assert_eq!(rope.cos_table_shape(), [128, 32]);
    assert_eq!(rope.sin_table_shape(), [128, 32]);
}

#[test]
fn rope_at_position_zero_is_identity() {
    let dev = Default::default();
    let rope = RotaryEmbedding::<B>::new(64, 128, 10000.0, &dev);
    // Random input: [batch=1, n_heads=4, seq=1, head_dim=64]
    let x = Tensor::<B, 4>::random(
        [1, 4, 1, 64],
        burn::tensor::Distribution::Default,
        &dev,
    );
    let positions = vec![0_i32];
    let out = rope.apply(x.clone(), &positions);
    let diff: f32 = (out - x).abs().max().into_scalar();
    assert!(diff < 1e-5, "RoPE@0 should be identity; max diff = {diff}");
}

#[test]
fn rope_at_different_positions_differs() {
    let dev = Default::default();
    let rope = RotaryEmbedding::<B>::new(64, 128, 10000.0, &dev);
    let x = Tensor::<B, 4>::ones([1, 4, 1, 64], &dev);
    let out_a = rope.apply(x.clone(), &[5]);
    let out_b = rope.apply(x, &[37]);
    let diff: f32 = (out_a - out_b).abs().max().into_scalar();
    assert!(diff > 1e-3, "RoPE at different positions should differ; max diff = {diff}");
}
```

- [ ] **Step 3: Implement RoPE**

`crates/ai-engine-runtime/src/arch/rope.rs`:

```rust
use burn::tensor::{backend::Backend, Tensor};

/// Rotary Positional Embeddings.
///
/// Precomputes cos/sin tables of shape `[max_seq, head_dim/2]` at construction.
/// `apply(x, positions)` rotates the last dim of `x` per-position. `x` has
/// shape `[batch, n_heads, seq_len, head_dim]`; `positions[i]` is the absolute
/// sequence position of the i-th token in `seq_len`.
///
/// **Critical implementation note**: HF Llama uses the "split-halves" RoPE
/// convention (rotate the FIRST half against the SECOND half), NOT the
/// interleaved evens/odds convention from the original RoFormer paper. If
/// the bytes-tolerant correctness gate in Task 12 fails with huge logit
/// differences, the split-halves vs interleaved convention is the most
/// likely culprit — verify against:
///   transformers/src/transformers/models/llama/modeling_llama.py::apply_rotary_pos_emb
pub struct RotaryEmbedding<B: Backend> {
    pub cos: Tensor<B, 2>,
    pub sin: Tensor<B, 2>,
    pub head_dim: usize,
    pub max_seq: usize,
}

impl<B: Backend> RotaryEmbedding<B> {
    pub fn new(head_dim: usize, max_seq: usize, theta: f32, device: &B::Device) -> Self {
        let half = head_dim / 2;
        let freqs: Vec<f32> = (0..half)
            .map(|k| 1.0 / theta.powf((2.0 * k as f32) / head_dim as f32))
            .collect();
        let mut cos_data = Vec::with_capacity(max_seq * half);
        let mut sin_data = Vec::with_capacity(max_seq * half);
        for t in 0..max_seq {
            for k in 0..half {
                let angle = t as f32 * freqs[k];
                cos_data.push(angle.cos());
                sin_data.push(angle.sin());
            }
        }
        let cos = Tensor::<B, 2>::from_floats(&cos_data[..], device).reshape([max_seq, half]);
        let sin = Tensor::<B, 2>::from_floats(&sin_data[..], device).reshape([max_seq, half]);
        Self { cos, sin, head_dim, max_seq }
    }

    pub fn cos_table_shape(&self) -> [usize; 2] { self.cos.dims() }
    pub fn sin_table_shape(&self) -> [usize; 2] { self.sin.dims() }

    /// Rotate `x` by RoPE. `x: [batch, n_heads, seq, head_dim]`,
    /// `positions[i]` = absolute seq position of token i within `seq`.
    ///
    /// Algorithm (split-halves, HF Llama convention):
    ///   first_half  = x[..., :head_dim/2]
    ///   second_half = x[..., head_dim/2:]
    ///   cos = cos_table[positions, :]  -> [seq, head_dim/2]
    ///   sin = sin_table[positions, :]  -> [seq, head_dim/2]
    ///   out_first  = first_half  * cos - second_half * sin
    ///   out_second = first_half  * sin + second_half * cos
    ///   concat(out_first, out_second) along last dim -> [batch, n_heads, seq, head_dim]
    ///
    /// The implementer fills in the burn tensor ops (slice, unsqueeze for
    /// broadcast, mul, sub, add, cat).
    pub fn apply(&self, x: Tensor<B, 4>, positions: &[i32]) -> Tensor<B, 4> {
        todo!("implementer: split-halves rotation using burn slice/cat/mul ops")
    }
}
```

The position-zero identity test catches the case where the convention is "interleaved" instead of "split-halves" — both should be identity at position 0, but the implementation will differ. The cross-position test catches missing or wrong RoPE application entirely. The bytes-tolerant gate in Task 12 is the ultimate verification.

- [ ] **Step 4: Failing test for KV cache**

`crates/ai-engine-runtime/tests/kv_cache.rs`:

```rust
use ai_engine_runtime::kv_cache::KvCacheSlot;
use burn::tensor::Tensor;

type B = burn_ndarray::NdArray;

#[test]
fn kv_cache_appends_tokens_and_reads_all() {
    let dev = Default::default();
    let mut slot = KvCacheSlot::<B>::new(
        /*batch=*/1, /*n_kv_heads=*/2, /*max_seq=*/16, /*head_dim=*/64, &dev,
    );
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
```

- [ ] **Step 5: Implement KV cache**

`crates/ai-engine-runtime/src/kv_cache.rs`:

```rust
use burn::tensor::{backend::Backend, Tensor};

/// Per-layer KV cache for a single request. Holds k/v tensors with
/// `seq` dimension growing as tokens are produced.
///
/// Memory: 2 * batch * n_kv_heads * max_seq * head_dim * sizeof(elem)
/// At bf16: 2 * 1 * 8 * 4096 * 128 * 2 = 16 MiB per layer for Llama-3-8B
/// at max_seq=4096, batch=1.
pub struct KvCacheSlot<B: Backend> {
    pub k: Tensor<B, 4>,    // [batch, n_kv_heads, max_seq, head_dim], unfilled positions = 0
    pub v: Tensor<B, 4>,
    pub max_seq: usize,
    current_len: usize,
}

impl<B: Backend> KvCacheSlot<B> {
    pub fn new(batch: usize, n_kv_heads: usize, max_seq: usize, head_dim: usize, device: &B::Device) -> Self {
        let k = Tensor::<B, 4>::zeros([batch, n_kv_heads, max_seq, head_dim], device);
        let v = Tensor::<B, 4>::zeros([batch, n_kv_heads, max_seq, head_dim], device);
        Self { k, v, max_seq, current_len: 0 }
    }

    pub fn current_len(&self) -> usize { self.current_len }

    /// Append `k_new`, `v_new` of shape `[batch, n_kv_heads, new_tokens, head_dim]`.
    /// Panics if appending would exceed `max_seq`.
    ///
    /// Algorithm: write k_new and v_new into self.k and self.v at positions
    /// [current_len..current_len+new_tokens] along seq dim (dim 2), then
    /// update current_len. burn 0.18: use Tensor::slice_assign on dim 2.
    pub fn append(&mut self, k_new: Tensor<B, 4>, v_new: Tensor<B, 4>) {
        let n = k_new.dims()[2];
        assert!(
            self.current_len + n <= self.max_seq,
            "KV cache overflow: have {}, adding {}, max {}",
            self.current_len, n, self.max_seq,
        );
        todo!("implementer: slice_assign self.k and self.v on dim 2 at [current_len..current_len+n]")
    }

    /// Returns slices of k and v covering [0..current_len] on the seq dim.
    pub fn read(&self) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let k = self.k.clone().slice([0..self.k.dims()[0], 0..self.k.dims()[1], 0..self.current_len, 0..self.k.dims()[3]]);
        let v = self.v.clone().slice([0..self.v.dims()[0], 0..self.v.dims()[1], 0..self.current_len, 0..self.v.dims()[3]]);
        (k, v)
    }
}
```

- [ ] **Step 6: Verify + commit**

```bash
cargo test -p ai-engine-runtime --tests rope kv_cache
cargo clippy --workspace --all-targets -- -D warnings
git add -A
git commit -m "feat(runtime): RoPE + KV cache primitives with isolated tests"
```

---

### Task 6: RMSNorm + SwiGLU FFN

**Files:**
- Create: `crates/ai-engine-runtime/src/arch/rmsnorm.rs`
- Create: `crates/ai-engine-runtime/src/arch/ffn.rs`
- Create: `crates/ai-engine-runtime/tests/norm_and_ffn.rs`
- Modify: `crates/ai-engine-runtime/src/arch/mod.rs`

Pure-math primitives — easy to test in isolation.

- [ ] **Step 1: Failing tests**

```rust
use ai_engine_runtime::arch::ffn::SwiGluFfn;
use ai_engine_runtime::arch::rmsnorm::RmsNorm;
use burn::tensor::Tensor;

type B = burn_ndarray::NdArray;

#[test]
fn rmsnorm_unit_weights_normalizes_to_unit_rms() {
    let dev = Default::default();
    let norm = RmsNorm::<B>::with_weights(
        /*hidden=*/4, /*weight=*/[1.0, 1.0, 1.0, 1.0], /*eps=*/1e-6, &dev,
    );
    let x = Tensor::<B, 2>::from_floats([[2.0, 2.0, 2.0, 2.0]], &dev);
    let out = norm.forward(x);
    // RMS of [2,2,2,2] is 2. Output should be [1,1,1,1].
    let v: Vec<f32> = out.into_data().to_vec().unwrap();
    for x in &v { assert!((x - 1.0).abs() < 1e-5, "{x} != 1"); }
}

#[test]
fn swiglu_ffn_runs_with_expected_output_shape() {
    let dev = Default::default();
    let ffn = SwiGluFfn::<B>::with_random_weights(/*hidden=*/8, /*inter=*/16, &dev);
    let x = Tensor::<B, 3>::ones([1, 2, 8], &dev);  // [batch=1, seq=2, hidden=8]
    let out = ffn.forward(x);
    assert_eq!(out.dims(), [1, 2, 8]);
}
```

- [ ] **Step 2: Implement RMSNorm**

`crates/ai-engine-runtime/src/arch/rmsnorm.rs`:

```rust
use burn::tensor::{backend::Backend, Tensor};

/// RMSNorm: `out = x * rsqrt(mean(x^2, dim=-1) + eps) * weight`.
pub struct RmsNorm<B: Backend> {
    pub weight: Tensor<B, 1>,   // [hidden]
    pub eps: f32,
}

impl<B: Backend> RmsNorm<B> {
    pub fn new(weight: Tensor<B, 1>, eps: f32) -> Self {
        Self { weight, eps }
    }

    pub fn with_weights(hidden: usize, weights: impl AsRef<[f32]>, eps: f32, device: &B::Device) -> Self {
        let weight = Tensor::<B, 1>::from_floats(weights.as_ref(), device);
        assert_eq!(weight.dims()[0], hidden);
        Self { weight, eps }
    }

    /// `x: [..., hidden]` — operates on the last dim.
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        // sq_mean = mean(x*x, dim=-1, keepdim=true)
        // rsqrt   = 1.0 / sqrt(sq_mean + eps)
        // out     = x * rsqrt * weight
        let sq = x.clone().powf_scalar(2.0);
        let mean = sq.mean_dim(D - 1);
        let rsqrt = mean.add_scalar(self.eps).sqrt().recip();
        // Broadcast self.weight to [..., hidden]; burn's broadcast rules handle this
        // when self.weight is [hidden] and the tensor has hidden as its last dim.
        x.mul(rsqrt).mul(self.weight.clone().unsqueeze())
    }
}
```

- [ ] **Step 3: Implement SwiGLU FFN**

`crates/ai-engine-runtime/src/arch/ffn.rs`:

```rust
use burn::tensor::{activation::silu, backend::Backend, Tensor};

/// SwiGLU FFN: `down(silu(gate(x)) * up(x))`.
pub struct SwiGluFfn<B: Backend> {
    pub gate_proj: Tensor<B, 2>,   // [hidden, inter]
    pub up_proj: Tensor<B, 2>,     // [hidden, inter]
    pub down_proj: Tensor<B, 2>,   // [inter, hidden]
}

impl<B: Backend> SwiGluFfn<B> {
    pub fn new(gate_proj: Tensor<B, 2>, up_proj: Tensor<B, 2>, down_proj: Tensor<B, 2>) -> Self {
        Self { gate_proj, up_proj, down_proj }
    }

    pub fn with_random_weights(hidden: usize, inter: usize, device: &B::Device) -> Self {
        let dist = burn::tensor::Distribution::Default;
        Self {
            gate_proj: Tensor::<B, 2>::random([hidden, inter], dist, device),
            up_proj:   Tensor::<B, 2>::random([hidden, inter], dist, device),
            down_proj: Tensor::<B, 2>::random([inter, hidden], dist, device),
        }
    }

    /// `x: [batch, seq, hidden]` -> `[batch, seq, hidden]`.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = x.clone().matmul(self.gate_proj.clone().unsqueeze());
        let up   = x.matmul(self.up_proj.clone().unsqueeze());
        silu(gate).mul(up).matmul(self.down_proj.clone().unsqueeze())
    }
}
```

(Implementer: verify `matmul` broadcasts a 2-d weight to a 3-d input correctly in burn 0.18; if not, use explicit reshape/unsqueeze.)

- [ ] **Step 4: Add modules to arch/mod.rs**

```rust
pub mod ffn;
pub mod rmsnorm;
pub mod rope;
```

- [ ] **Step 5: Verify + commit**

```bash
cargo test -p ai-engine-runtime --tests norm_and_ffn
cargo clippy --workspace --all-targets -- -D warnings
git add -A
git commit -m "feat(runtime): RMSNorm + SwiGLU FFN primitives"
```

---

### Task 7: Attention (GQA + RoPE + KV cache)

**Files:**
- Create: `crates/ai-engine-runtime/src/arch/attention.rs`
- Create: `crates/ai-engine-runtime/tests/attention.rs`
- Modify: `crates/ai-engine-runtime/src/arch/mod.rs`

This is the most complex single primitive. Splitting Q/K/V projections, applying RoPE, doing GQA broadcast, scaled dot-product, KV-cache integration.

- [ ] **Step 1: Failing test**

```rust
use ai_engine_runtime::arch::attention::Attention;
use ai_engine_runtime::kv_cache::KvCacheSlot;
use burn::tensor::Tensor;

type B = burn_ndarray::NdArray;

#[test]
fn attention_forward_shape_correct_for_gqa() {
    let dev = Default::default();
    // GQA: 4 query heads, 2 KV heads, head_dim 8 -> hidden = 32.
    let attn = Attention::<B>::with_random_weights(
        /*hidden=*/32, /*n_heads=*/4, /*n_kv_heads=*/2, /*head_dim=*/8,
        /*max_seq=*/16, /*rope_theta=*/10000.0, &dev,
    );
    let mut cache = KvCacheSlot::<B>::new(1, 2, 16, 8, &dev);
    let x = Tensor::<B, 3>::ones([1, 3, 32], &dev);   // [batch=1, seq=3, hidden=32]
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
```

- [ ] **Step 2: Implement attention**

`crates/ai-engine-runtime/src/arch/attention.rs`:

```rust
use crate::arch::rope::RotaryEmbedding;
use crate::kv_cache::KvCacheSlot;
use burn::tensor::{activation::softmax, backend::Backend, Tensor};

pub struct Attention<B: Backend> {
    pub q_proj: Tensor<B, 2>,        // [hidden, n_heads * head_dim]
    pub k_proj: Tensor<B, 2>,        // [hidden, n_kv_heads * head_dim]
    pub v_proj: Tensor<B, 2>,        // [hidden, n_kv_heads * head_dim]
    pub o_proj: Tensor<B, 2>,        // [n_heads * head_dim, hidden]
    pub rope: RotaryEmbedding<B>,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub scale: f32,                  // 1.0 / sqrt(head_dim)
}

impl<B: Backend> Attention<B> {
    pub fn new(
        q_proj: Tensor<B, 2>, k_proj: Tensor<B, 2>,
        v_proj: Tensor<B, 2>, o_proj: Tensor<B, 2>,
        rope: RotaryEmbedding<B>,
        n_heads: usize, n_kv_heads: usize, head_dim: usize,
    ) -> Self {
        let scale = 1.0 / (head_dim as f32).sqrt();
        Self { q_proj, k_proj, v_proj, o_proj, rope, n_heads, n_kv_heads, head_dim, scale }
    }

    pub fn with_random_weights(
        hidden: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize,
        max_seq: usize, rope_theta: f32, device: &B::Device,
    ) -> Self {
        let dist = burn::tensor::Distribution::Default;
        let q_proj = Tensor::<B, 2>::random([hidden, n_heads * head_dim], dist, device);
        let k_proj = Tensor::<B, 2>::random([hidden, n_kv_heads * head_dim], dist, device);
        let v_proj = Tensor::<B, 2>::random([hidden, n_kv_heads * head_dim], dist, device);
        let o_proj = Tensor::<B, 2>::random([n_heads * head_dim, hidden], dist, device);
        let rope = RotaryEmbedding::<B>::new(head_dim, max_seq, rope_theta, device);
        Self::new(q_proj, k_proj, v_proj, o_proj, rope, n_heads, n_kv_heads, head_dim)
    }

    /// `x: [batch, seq, hidden]`, `positions[i]` = absolute pos of input token i.
    /// Mutates `cache` by appending the new K, V derived from `x`.
    /// Returns `[batch, seq, hidden]`.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        positions: &[i32],
        cache: &mut KvCacheSlot<B>,
    ) -> Tensor<B, 3> {
        let [batch, seq, _hidden] = x.dims();

        // 1. Linear projections — Q, K, V
        let q = x.clone().matmul(self.q_proj.clone().unsqueeze());
        let k = x.clone().matmul(self.k_proj.clone().unsqueeze());
        let v = x.matmul(self.v_proj.clone().unsqueeze());

        // 2. Reshape to [batch, n_heads | n_kv_heads, seq, head_dim].
        let q = q.reshape([batch, seq, self.n_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = k.reshape([batch, seq, self.n_kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = v.reshape([batch, seq, self.n_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        // 3. Apply RoPE to q and k.
        let q = self.rope.apply(q, positions);
        let k = self.rope.apply(k, positions);

        // 4. Append k, v to cache; then read the full cached k, v.
        cache.append(k, v);
        let (k_all, v_all) = cache.read();

        // 5. Broadcast k_all, v_all from n_kv_heads to n_heads (GQA).
        //    Repeat each KV head `n_heads / n_kv_heads` times along the head dim.
        //    Llama convention: CONSECUTIVE repetition — for 2 KV heads -> 4 query heads,
        //    the layout is [kv0, kv0, kv1, kv1], NOT [kv0, kv1, kv0, kv1].
        let repeat = self.n_heads / self.n_kv_heads;
        let k_all = repeat_heads(k_all, repeat);
        let v_all = repeat_heads(v_all, repeat);

        // 6. Scaled dot-product attention.
        let scores = q.matmul(k_all.swap_dims(2, 3)).mul_scalar(self.scale);
        let scores = apply_causal_mask::<B>(scores, positions);
        let probs = softmax(scores, /*dim=*/3);
        let ctx = probs.matmul(v_all);    // [batch, n_heads, seq, head_dim]

        // 7. Merge heads back: [batch, seq, n_heads * head_dim].
        let ctx = ctx.swap_dims(1, 2).reshape([batch, seq, self.n_heads * self.head_dim]);

        // 8. Output projection.
        ctx.matmul(self.o_proj.clone().unsqueeze())
    }
}

/// Repeat each KV head `n` times consecutively along dim 1 to match n_heads.
/// Llama convention: [kv0, kv0, kv1, kv1] for repeat=2, n_kv_heads=2.
fn repeat_heads<B: Backend>(x: Tensor<B, 4>, n: usize) -> Tensor<B, 4> {
    if n == 1 { return x; }
    // burn API: implementer uses Tensor::repeat_dim, or
    //   x.unsqueeze_dim(2) -> [batch, n_kv_heads, 1, seq, head_dim]
    //   .expand([batch, n_kv_heads, n, seq, head_dim])
    //   .reshape([batch, n_kv_heads * n, seq, head_dim])
    todo!("implementer: KV head broadcast (GQA), n={n}")
}

/// `scores: [batch, n_heads, q_seq, k_seq]`. Causal mask: position j > positions[i] -> -inf.
fn apply_causal_mask<B: Backend>(scores: Tensor<B, 4>, positions: &[i32]) -> Tensor<B, 4> {
    // Build a mask tensor [q_seq, k_seq] with -inf above the diagonal aligned to positions,
    // then broadcast to [1, 1, q_seq, k_seq] and add to scores.
    todo!("implementer: causal mask construction")
}
```

- [ ] **Step 3: Add module + verify + commit**

```rust
// arch/mod.rs adds: pub mod attention;
```

```bash
cargo test -p ai-engine-runtime --tests attention
cargo clippy --workspace --all-targets -- -D warnings
git add -A
git commit -m "feat(runtime): GQA attention with RoPE and KV cache integration"
```

---

### Task 8: Decoder block + Model wiring

**Files:**
- Create: `crates/ai-engine-runtime/src/arch/block.rs`
- Create: `crates/ai-engine-runtime/src/arch/embedding.rs`
- Create: `crates/ai-engine-runtime/src/arch/model.rs`
- Create: `crates/ai-engine-runtime/tests/model_shape.rs`
- Modify: `crates/ai-engine-runtime/src/arch/mod.rs`

Wire the primitives into a complete decoder.

- [ ] **Step 1: Failing test for end-to-end shape**

```rust
use ai_engine_runtime::config::{ModelConfig, ModelFamily};
use ai_engine_runtime::arch::model::Model;
use burn::tensor::Tensor;

type B = burn_ndarray::NdArray;

#[test]
fn model_forward_produces_correct_logit_shape() {
    let dev = Default::default();
    let cfg = ModelConfig {
        hidden_size: 32, intermediate_size: 64, n_layers: 2,
        n_heads: 4, n_kv_heads: 2, head_dim: 8,
        vocab_size: 100, max_position_embeddings: 32,
        rope_theta: 10000.0, rms_norm_eps: 1e-5,
        tie_word_embeddings: true, family: ModelFamily::Llama3,
    };
    let model = Model::<B>::with_random_weights(&cfg, &dev);
    let token_ids = Tensor::<B, 2, burn::tensor::Int>::from_data(
        burn::tensor::TensorData::new(vec![1_i32, 2, 3, 4, 5], [1, 5]),
        &dev,
    );
    let logits = model.forward(token_ids, /*start_pos=*/0);
    assert_eq!(logits.dims(), [1, 5, 100]);
}
```

- [ ] **Step 2: Implement embedding + output projection**

`crates/ai-engine-runtime/src/arch/embedding.rs`:

```rust
use burn::tensor::{backend::Backend, Int, Tensor};

pub struct TokenEmbedding<B: Backend> {
    pub weight: Tensor<B, 2>,        // [vocab, hidden]
}

impl<B: Backend> TokenEmbedding<B> {
    pub fn new(weight: Tensor<B, 2>) -> Self { Self { weight } }

    /// `ids: [batch, seq]` -> `[batch, seq, hidden]`.
    /// Algorithm: select rows from self.weight at the flat ids, then reshape.
    pub fn forward(&self, ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        todo!("implementer: embedding lookup via Tensor::select on dim 0")
    }
}

pub struct OutputProjection<B: Backend> {
    pub weight: Tensor<B, 2>,        // [hidden, vocab]
}

impl<B: Backend> OutputProjection<B> {
    pub fn new(weight: Tensor<B, 2>) -> Self { Self { weight } }

    /// `x: [batch, seq, hidden]` -> `[batch, seq, vocab]`.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        x.matmul(self.weight.clone().unsqueeze())
    }
}
```

- [ ] **Step 3: Implement decoder block**

`crates/ai-engine-runtime/src/arch/block.rs`:

```rust
use crate::arch::{attention::Attention, ffn::SwiGluFfn, rmsnorm::RmsNorm};
use crate::kv_cache::KvCacheSlot;
use burn::tensor::{backend::Backend, Tensor};

pub struct DecoderBlock<B: Backend> {
    pub attn_norm: RmsNorm<B>,
    pub attn: Attention<B>,
    pub ffn_norm: RmsNorm<B>,
    pub ffn: SwiGluFfn<B>,
}

impl<B: Backend> DecoderBlock<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        positions: &[i32],
        cache: &mut KvCacheSlot<B>,
    ) -> Tensor<B, 3> {
        // Residual 1: x = x + attn(norm(x))
        let h = self.attn_norm.forward(x.clone());
        let h = self.attn.forward(h, positions, cache);
        let x = x.add(h);
        // Residual 2: x = x + ffn(norm(x))
        let h = self.ffn_norm.forward(x.clone());
        let h = self.ffn.forward(h);
        x.add(h)
    }
}
```

- [ ] **Step 4: Implement Model**

`crates/ai-engine-runtime/src/arch/model.rs`:

```rust
use crate::arch::{block::DecoderBlock, embedding::{OutputProjection, TokenEmbedding}, rmsnorm::RmsNorm};
use crate::config::ModelConfig;
use crate::kv_cache::KvCacheSlot;
use burn::tensor::{backend::Backend, Int, Tensor};

pub struct Model<B: Backend> {
    pub embedding: TokenEmbedding<B>,
    pub blocks: Vec<DecoderBlock<B>>,
    pub final_norm: RmsNorm<B>,
    pub output: OutputProjection<B>,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq: usize,
}

impl<B: Backend> Model<B> {
    /// Random-weight constructor for shape / smoke tests.
    /// Production path uses `from_loaded` (Task 11).
    pub fn with_random_weights(cfg: &ModelConfig, device: &B::Device) -> Self {
        // Build TokenEmbedding, DecoderBlocks, RmsNorm, OutputProjection
        // all from random weights of the right shapes per cfg.
        // (Implementer composes the existing `with_random_weights` constructors
        //  for each primitive — no new ML logic.)
        todo!("implementer: wire random-weight constructor from cfg dimensions")
    }

    /// Used only by the shape test in Task 8 — each block gets a fresh KV cache.
    /// Production calls go through `forward_with_caches` (Task 13).
    pub fn forward(&self, token_ids: Tensor<B, 2, Int>, start_pos: usize) -> Tensor<B, 3> {
        let [batch, seq] = token_ids.dims();
        let positions: Vec<i32> = (start_pos..start_pos + seq).map(|p| p as i32).collect();
        let mut x = self.embedding.forward(token_ids);
        let device = x.device();
        for block in &self.blocks {
            let mut cache = KvCacheSlot::<B>::new(batch, self.n_kv_heads, self.max_seq, self.head_dim, &device);
            x = block.forward(x, &positions, &mut cache);
        }
        let x = self.final_norm.forward(x);
        self.output.forward(x)
    }
}
```

- [ ] **Step 5: arch/mod.rs adds**

```rust
pub mod block;
pub mod embedding;
pub mod model;
```

- [ ] **Step 6: Verify + commit**

```bash
cargo test -p ai-engine-runtime --tests model_shape
cargo clippy --workspace --all-targets -- -D warnings
git add -A
git commit -m "feat(runtime): DecoderBlock + Model with random-weight constructor"
```

---

### Task 9: Name map for the four model families

**Files:**
- Modify: `crates/ai-engine-runtime/src/name_map.rs`
- Create: `crates/ai-engine-runtime/tests/name_map.rs`

Maps logical tensor identifiers (e.g., `LayerQProj(12)`) to HF safetensors keys (`model.layers.12.self_attn.q_proj.weight`).

- [ ] **Step 1: Failing test**

```rust
use ai_engine_runtime::config::ModelFamily;
use ai_engine_runtime::name_map::{TensorId, WeightNameMap};

#[test]
fn llama3_q_proj_layer_12() {
    let nm = WeightNameMap::for_family(ModelFamily::Llama3);
    assert_eq!(
        nm.lookup(TensorId::LayerQProj(12)),
        "model.layers.12.self_attn.q_proj.weight"
    );
    assert_eq!(
        nm.lookup(TensorId::LayerKProj(12)),
        "model.layers.12.self_attn.k_proj.weight"
    );
}

#[test]
fn llama3_boundary_tensors() {
    let nm = WeightNameMap::for_family(ModelFamily::Llama3);
    assert_eq!(nm.lookup(TensorId::Embedding), "model.embed_tokens.weight");
    assert_eq!(nm.lookup(TensorId::FinalNorm), "model.norm.weight");
    assert_eq!(nm.lookup(TensorId::OutputProjection), "lm_head.weight");
}
```

- [ ] **Step 2: Implement**

```rust
use crate::config::ModelFamily;

#[derive(Debug, Clone, Copy)]
pub enum TensorId {
    Embedding,
    FinalNorm,
    OutputProjection,
    LayerAttnNorm(usize),
    LayerQProj(usize),
    LayerKProj(usize),
    LayerVProj(usize),
    LayerOProj(usize),
    LayerFfnNorm(usize),
    LayerFfnGate(usize),
    LayerFfnUp(usize),
    LayerFfnDown(usize),
}

pub struct WeightNameMap { family: ModelFamily }

impl WeightNameMap {
    pub fn for_family(family: ModelFamily) -> Self { Self { family } }

    pub fn lookup(&self, id: TensorId) -> String {
        match self.family {
            ModelFamily::Llama3 | ModelFamily::Mistral | ModelFamily::DeepSeekV2 => Self::llama_style(id),
            ModelFamily::Qwen25 => Self::qwen_style(id),
        }
    }

    fn llama_style(id: TensorId) -> String {
        use TensorId::*;
        match id {
            Embedding         => "model.embed_tokens.weight".into(),
            FinalNorm         => "model.norm.weight".into(),
            OutputProjection  => "lm_head.weight".into(),
            LayerAttnNorm(i)  => format!("model.layers.{i}.input_layernorm.weight"),
            LayerQProj(i)     => format!("model.layers.{i}.self_attn.q_proj.weight"),
            LayerKProj(i)     => format!("model.layers.{i}.self_attn.k_proj.weight"),
            LayerVProj(i)     => format!("model.layers.{i}.self_attn.v_proj.weight"),
            LayerOProj(i)     => format!("model.layers.{i}.self_attn.o_proj.weight"),
            LayerFfnNorm(i)   => format!("model.layers.{i}.post_attention_layernorm.weight"),
            LayerFfnGate(i)   => format!("model.layers.{i}.mlp.gate_proj.weight"),
            LayerFfnUp(i)     => format!("model.layers.{i}.mlp.up_proj.weight"),
            LayerFfnDown(i)   => format!("model.layers.{i}.mlp.down_proj.weight"),
        }
    }

    fn qwen_style(id: TensorId) -> String {
        // Qwen 2.5 uses the same `model.layers.N.self_attn.q_proj.weight` pattern as Llama.
        // Implementer verifies via an actual Qwen 2.5 safetensors dump and branches here
        // if any names differ.
        Self::llama_style(id)
    }
}
```

- [ ] **Step 3: Verify + commit**

```bash
cargo test -p ai-engine-runtime --tests name_map
git add -A
git commit -m "feat(runtime): WeightNameMap for Llama-3 / Mistral / Qwen-2.5 / DeepSeek-V2"
```

---

### Task 10: safetensors loader + `load_range`

**Files:**
- Create: `crates/ai-engine-runtime/src/loader.rs`
- Create: `crates/ai-engine-runtime/tests/loader.rs`

- [ ] **Step 1: Failing test against the toy fixture from Task 4**

```rust
use ai_engine_runtime::config::ModelConfig;
use ai_engine_runtime::loader::{load_range, LoadedWeights};
use std::path::PathBuf;

type B = burn_ndarray::NdArray;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/toy-llama-3")
}

#[test]
fn load_full_model_for_single_node() {
    let cfg = ModelConfig::from_file(&fixture_path().join("config.json")).unwrap();
    let dev = Default::default();
    let weights: LoadedWeights<B> = load_range(
        &fixture_path().join("model.safetensors"),
        &cfg,
        /*layer_range=*/0..cfg.n_layers,
        /*hosts_embedding=*/true,
        /*hosts_output=*/true,
        &dev,
    ).unwrap();
    assert!(weights.embedding.is_some());
    assert!(weights.final_norm.is_some());
    assert_eq!(weights.layers.len(), cfg.n_layers);
}

#[test]
fn load_layer_range_for_worker_node() {
    let cfg = ModelConfig::from_file(&fixture_path().join("config.json")).unwrap();
    let dev = Default::default();
    let weights: LoadedWeights<B> = load_range(
        &fixture_path().join("model.safetensors"),
        &cfg,
        1..3,
        false,
        false,
        &dev,
    ).unwrap();
    assert!(weights.embedding.is_none());
    assert!(weights.final_norm.is_none());
    assert!(weights.output_proj.is_none());
    assert_eq!(weights.layers.len(), 2);
}
```

- [ ] **Step 2: Implement loader**

`crates/ai-engine-runtime/src/loader.rs`:

```rust
use crate::config::ModelConfig;
use crate::name_map::{TensorId, WeightNameMap};
use anyhow::Context;
use burn::tensor::{backend::Backend, Tensor};
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::ops::Range;
use std::path::Path;

pub struct LayerWeights<B: Backend> {
    pub attn_norm: Tensor<B, 1>,
    pub q_proj: Tensor<B, 2>,
    pub k_proj: Tensor<B, 2>,
    pub v_proj: Tensor<B, 2>,
    pub o_proj: Tensor<B, 2>,
    pub ffn_norm: Tensor<B, 1>,
    pub ffn_gate: Tensor<B, 2>,
    pub ffn_up: Tensor<B, 2>,
    pub ffn_down: Tensor<B, 2>,
}

pub struct LoadedWeights<B: Backend> {
    pub embedding: Option<Tensor<B, 2>>,
    pub layers: Vec<LayerWeights<B>>,
    pub final_norm: Option<Tensor<B, 1>>,
    pub output_proj: Option<Tensor<B, 2>>,
}

pub fn load_range<B: Backend>(
    path: &Path,
    cfg: &ModelConfig,
    layer_range: Range<usize>,
    hosts_embedding: bool,
    hosts_output: bool,
    device: &B::Device,
) -> anyhow::Result<LoadedWeights<B>> {
    let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mmap = unsafe { Mmap::map(&file) }.with_context(|| format!("mmap {}", path.display()))?;
    let st = SafeTensors::deserialize(&mmap)
        .with_context(|| format!("parse safetensors header from {}", path.display()))?;

    let nm = WeightNameMap::for_family(cfg.family);

    let load_2d = |id: TensorId| -> anyhow::Result<Tensor<B, 2>> {
        let name = nm.lookup(id);
        let view = st.tensor(&name).with_context(|| format!("missing tensor `{name}`"))?;
        let data = view.data();
        let shape = view.shape();
        let f32_data = bytes_to_f32_vec(data, view.dtype())?;
        Ok(Tensor::<B, 2>::from_floats(&f32_data[..], device).reshape([shape[0], shape[1]]))
    };

    let load_1d = |id: TensorId| -> anyhow::Result<Tensor<B, 1>> {
        let name = nm.lookup(id);
        let view = st.tensor(&name).with_context(|| format!("missing tensor `{name}`"))?;
        let data = view.data();
        let f32_data = bytes_to_f32_vec(data, view.dtype())?;
        Ok(Tensor::<B, 1>::from_floats(&f32_data[..], device))
    };

    let embedding = if hosts_embedding {
        Some(load_2d(TensorId::Embedding)?)
    } else { None };

    let mut layers = Vec::with_capacity(layer_range.len());
    for i in layer_range.clone() {
        layers.push(LayerWeights {
            attn_norm: load_1d(TensorId::LayerAttnNorm(i))?,
            q_proj:    load_2d(TensorId::LayerQProj(i))?,
            k_proj:    load_2d(TensorId::LayerKProj(i))?,
            v_proj:    load_2d(TensorId::LayerVProj(i))?,
            o_proj:    load_2d(TensorId::LayerOProj(i))?,
            ffn_norm:  load_1d(TensorId::LayerFfnNorm(i))?,
            ffn_gate:  load_2d(TensorId::LayerFfnGate(i))?,
            ffn_up:    load_2d(TensorId::LayerFfnUp(i))?,
            ffn_down:  load_2d(TensorId::LayerFfnDown(i))?,
        });
    }

    let final_norm = if hosts_output {
        Some(load_1d(TensorId::FinalNorm)?)
    } else { None };

    let output_proj = if hosts_output && !cfg.tie_word_embeddings {
        Some(load_2d(TensorId::OutputProjection)?)
    } else { None };

    Ok(LoadedWeights { embedding, layers, final_norm, output_proj })
}

fn bytes_to_f32_vec(raw: &[u8], dtype: safetensors::Dtype) -> anyhow::Result<Vec<f32>> {
    use safetensors::Dtype::*;
    match dtype {
        F32 => Ok(bytemuck::cast_slice::<u8, f32>(raw).to_vec()),
        F16 => Ok(bytemuck::cast_slice::<u8, half::f16>(raw).iter().map(|x| x.to_f32()).collect()),
        BF16 => Ok(bytemuck::cast_slice::<u8, half::bf16>(raw).iter().map(|x| x.to_f32()).collect()),
        other => anyhow::bail!("unsupported safetensors dtype: {other:?}"),
    }
}
```

(Implementer: verify `safetensors::Dtype` variant names and `half::bf16` API against the actual crate versions.)

- [ ] **Step 3: Verify + commit**

```bash
cargo test -p ai-engine-runtime --tests loader
git add -A
git commit -m "feat(runtime): safetensors loader with per-layer range support"
```

---

### Task 11: Wire `Model::from_loaded` constructor

**Files:**
- Modify: `crates/ai-engine-runtime/src/arch/model.rs`

Now add a production constructor that takes `LoadedWeights` and builds the `Model`.

- [ ] **Step 1: Implement**

Add to `crates/ai-engine-runtime/src/arch/model.rs`:

```rust
use crate::arch::{attention::Attention, ffn::SwiGluFfn, rope::RotaryEmbedding};
use crate::loader::LoadedWeights;

impl<B: Backend> Model<B> {
    pub fn from_loaded(
        cfg: &ModelConfig,
        weights: LoadedWeights<B>,
        device: &B::Device,
    ) -> anyhow::Result<Self> {
        let embedding = TokenEmbedding::new(weights.embedding
            .ok_or_else(|| anyhow::anyhow!("embedding required but not loaded"))?);
        let final_norm = RmsNorm::new(weights.final_norm
            .ok_or_else(|| anyhow::anyhow!("final norm required but not loaded"))?, cfg.rms_norm_eps);

        let output_weight = match (cfg.tie_word_embeddings, weights.output_proj) {
            // tied: reuse the embedding matrix transposed
            (true, _) => embedding.weight.clone().transpose(),
            (false, Some(w)) => w,
            (false, None) => anyhow::bail!("untied output projection missing"),
        };
        let output = OutputProjection::new(output_weight);

        let mut blocks = Vec::with_capacity(weights.layers.len());
        for layer in weights.layers {
            let attn_norm = RmsNorm::new(layer.attn_norm, cfg.rms_norm_eps);
            let ffn_norm = RmsNorm::new(layer.ffn_norm, cfg.rms_norm_eps);
            let rope = RotaryEmbedding::new(cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta, device);
            let attn = Attention::new(
                layer.q_proj, layer.k_proj, layer.v_proj, layer.o_proj,
                rope, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim,
            );
            let ffn = SwiGluFfn::new(layer.ffn_gate, layer.ffn_up, layer.ffn_down);
            blocks.push(DecoderBlock { attn_norm, attn, ffn_norm, ffn });
        }

        Ok(Self {
            embedding, blocks, final_norm, output,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            max_seq: cfg.max_position_embeddings,
        })
    }
}
```

The `embedding.weight.clone().transpose()` may need adjustment depending on burn's transpose for 2D tensors.

- [ ] **Step 2: Verify + commit**

```bash
cargo test -p ai-engine-runtime
cargo clippy --workspace --all-targets -- -D warnings
git add -A
git commit -m "feat(runtime): Model::from_loaded constructor wiring LoadedWeights"
```

---

### Task 12: The bytes-tolerant correctness gate

**Files:**
- Create: `crates/ai-engine-runtime/tests/reference_logits.rs`

The single most important test in this entire plan. If it passes, the math is right. If it doesn't, dig in — most likely culprits: RoPE convention (interleaved vs split-halves), KV head broadcast for GQA, RMSNorm broadcasting, or matmul dimension ordering.

- [ ] **Step 1: Write the gate test**

```rust
use ai_engine_runtime::arch::model::Model;
use ai_engine_runtime::config::ModelConfig;
use ai_engine_runtime::loader::load_range;
use ai_engine_tokenizer::{HfTokenizer, Tokenizer};
use burn::tensor::Tensor;
use std::path::PathBuf;

type B = burn_ndarray::NdArray;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/toy-llama-3")
}

#[test]
fn forward_matches_reference_logits_within_tolerance() {
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
    let token_ids = Tensor::<B, 2, burn::tensor::Int>::from_data(
        burn::tensor::TensorData::new(ids_i32.clone(), [1, ids.len()]),
        &dev,
    );
    let logits = model.forward(token_ids, 0);
    let last_pos_logits: Tensor<B, 1> = logits
        .slice([0..1, (ids.len() - 1)..ids.len(), 0..cfg.vocab_size])
        .reshape([cfg.vocab_size]);

    let ref_bytes = std::fs::read(fix.join("reference_logits.bin")).unwrap();
    let ref_f32: &[f32] = bytemuck::cast_slice(&ref_bytes);
    assert_eq!(ref_f32.len(), cfg.vocab_size, "reference logits length matches vocab");

    let got: Vec<f32> = last_pos_logits.into_data().to_vec().unwrap();
    assert_eq!(got.len(), cfg.vocab_size);

    let mut max_abs_diff = 0.0_f32;
    for (i, (a, b)) in got.iter().zip(ref_f32.iter()).enumerate() {
        let d = (a - b).abs();
        if d > max_abs_diff { max_abs_diff = d; }
        if d >= 1e-3 {
            eprintln!("logit[{i}] diff = {d}  ours={a}  ref={b}");
        }
    }
    assert!(
        max_abs_diff < 1e-3,
        "bytes-tolerant gate failed: max |a-b| = {max_abs_diff}"
    );
}
```

- [ ] **Step 2: Run + iterate**

```bash
cargo test -p ai-engine-runtime --test reference_logits -- --nocapture
```

**This test will almost certainly fail on the first run.** Common failure modes (debug in this order):

1. **`max_abs_diff` ~ huge (>>1)** — likely RoPE convention is wrong (interleaved vs split-halves). Fix: swap the rotation formula in `arch/rope.rs::apply`.
2. **`max_abs_diff` ~ 1–10** — likely KV head broadcast in GQA is wrong. Fix: check `repeat_heads` in `arch/attention.rs` — it should repeat consecutively (`[k0, k0, k1, k1]`), not interleave (`[k0, k1, k0, k1]`).
3. **`max_abs_diff` ~ 0.1–1** — likely RMSNorm broadcasting weight wrong, or matmul dim ordering wrong somewhere.
4. **`max_abs_diff` ~ 1e-3 to 0.1** — close but not within tolerance. Try forcing f32 throughout the pipeline temporarily; if it then passes, the issue is precision-related (bf16 numerical drift).
5. **NaN / Inf** — a division by zero or saturation; check rsqrt in RMSNorm and softmax in attention.

This is iteration territory. Budget ≥1 day of debugging here, walking through each primitive's output against a Python reference if the gate fails.

- [ ] **Step 3: Commit when passing**

```bash
git add -A
git commit -m "test(runtime): bytes-tolerant correctness gate against transformers reference

forward_matches_reference_logits_within_tolerance: load toy-llama-3
weights, run a forward pass, compare logits at the final position
against transformers' reference (precomputed in fixtures/). Tolerance:
max |a - b| < 1e-3 in bf16."
```

---

### Task 13: Multi-step generation + KV cache cross-check + `forward_with_caches`

**Files:**
- Modify: `crates/ai-engine-runtime/src/arch/model.rs`
- Create: `crates/ai-engine-runtime/tests/generation.rs`

- [ ] **Step 1: Add `forward_with_caches` to Model**

Append to `arch/model.rs`:

```rust
impl<B: Backend> Model<B> {
    /// Production single-stream forward. Caller owns the cache slots
    /// (one per block) and they persist across calls within one request.
    pub fn forward_with_caches(
        &self,
        token_ids: Tensor<B, 2, Int>,
        start_pos: usize,
        caches: &mut [KvCacheSlot<B>],
    ) -> Tensor<B, 3> {
        assert_eq!(caches.len(), self.blocks.len(), "one cache per block");
        let [_batch, seq] = token_ids.dims();
        let positions: Vec<i32> = (start_pos..start_pos + seq).map(|p| p as i32).collect();
        let mut x = self.embedding.forward(token_ids);
        for (block, cache) in self.blocks.iter().zip(caches.iter_mut()) {
            x = block.forward(x, &positions, cache);
        }
        let x = self.final_norm.forward(x);
        self.output.forward(x)
    }
}
```

- [ ] **Step 2: Cross-check test**

`crates/ai-engine-runtime/tests/generation.rs`:

```rust
use ai_engine_runtime::arch::model::Model;
use ai_engine_runtime::config::ModelConfig;
use ai_engine_runtime::loader::load_range;
use ai_engine_runtime::kv_cache::KvCacheSlot;
use ai_engine_tokenizer::{HfTokenizer, Tokenizer};
use burn::tensor::Tensor;
use std::path::PathBuf;

type B = burn_ndarray::NdArray;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/toy-llama-3")
}

#[test]
fn cached_generation_matches_fresh_full_forward() {
    let fix = fixture();
    let cfg = ModelConfig::from_file(&fix.join("config.json")).unwrap();
    let tok = HfTokenizer::from_path(fix.join("tokenizer.json")).unwrap();
    let dev = Default::default();
    let weights = load_range::<B>(&fix.join("model.safetensors"), &cfg, 0..cfg.n_layers, true, true, &dev).unwrap();
    let model = Model::<B>::from_loaded(&cfg, weights, &dev).unwrap();

    let prompt = "The quick brown fox";
    let prompt_ids: Vec<u32> = tok.encode(prompt).unwrap();

    // Path A: feed all tokens at once (prefill) and read final-position logits.
    let prefill = Tensor::<B, 2, burn::tensor::Int>::from_data(
        burn::tensor::TensorData::new(prompt_ids.iter().map(|x| *x as i32).collect::<Vec<_>>(), [1, prompt_ids.len()]),
        &dev,
    );
    let logits_a = model.forward(prefill, 0);
    let last_a: Vec<f32> = logits_a.slice([0..1, (prompt_ids.len()-1)..prompt_ids.len(), 0..cfg.vocab_size])
        .reshape([cfg.vocab_size]).into_data().to_vec().unwrap();

    // Path B: feed tokens one-at-a-time, reusing the SAME caches across steps.
    let device = dev.clone();
    let mut caches: Vec<KvCacheSlot<B>> = (0..cfg.n_layers)
        .map(|_| KvCacheSlot::<B>::new(1, cfg.n_kv_heads, cfg.max_position_embeddings, cfg.head_dim, &device))
        .collect();
    let mut last_b: Vec<f32> = vec![];
    for (i, tid) in prompt_ids.iter().enumerate() {
        let t = Tensor::<B, 2, burn::tensor::Int>::from_data(
            burn::tensor::TensorData::new(vec![*tid as i32], [1, 1]),
            &device,
        );
        let logits = model.forward_with_caches(t, i, &mut caches);
        last_b = logits.reshape([cfg.vocab_size]).into_data().to_vec().unwrap();
    }

    let diff: f32 = last_a.iter().zip(last_b.iter()).map(|(a,b)| (a-b).abs()).fold(0., f32::max);
    assert!(diff < 1e-3, "cached and fresh logits should match within bf16 tolerance, diff = {diff}");
}
```

- [ ] **Step 3: Verify + commit**

```bash
cargo test -p ai-engine-runtime --test generation
cargo clippy --workspace --all-targets -- -D warnings
git add -A
git commit -m "test(runtime): cached generation matches fresh full forward (RoPE/KV cross-check)"
```

---

### Task 14: Sampling

**Files:**
- Create: `crates/ai-engine-runtime/src/sample.rs`
- Create: `crates/ai-engine-runtime/tests/sample.rs`
- Modify: `crates/ai-engine-runtime/src/lib.rs`
- Modify: root `Cargo.toml`

- [ ] **Step 1: Add `rand` workspace dep**

Append to root `Cargo.toml` `[workspace.dependencies]`:

```toml
rand = "0.8"
```

Add `rand.workspace = true` to `crates/ai-engine-runtime/Cargo.toml` `[dependencies]`.

- [ ] **Step 2: Failing tests**

```rust
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
```

- [ ] **Step 3: Implement**

```rust
use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;

#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub seed: u64,
}

/// `logits: [vocab]`. Returns the chosen token id.
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
    let mut rng = SmallRng::seed_from_u64(cfg.seed);
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
```

- [ ] **Step 4: Verify + commit**

```bash
cargo test -p ai-engine-runtime --test sample
git add -A
git commit -m "feat(runtime): sampling (greedy / temperature / top-p / top-k)"
```

---

### Task 15: Public surface in lib.rs + backend factory

**Files:**
- Modify: `crates/ai-engine-runtime/src/lib.rs`
- Modify: `crates/ai-engine-runtime/src/backend.rs`

- [ ] **Step 1: Expose public surface**

```rust
//! ai-engine-runtime
//!
//! Single-node inference primitives. Distributed orchestration lives in
//! ai-engine-cluster (Plan 2).

pub mod arch;
pub mod backend;
pub mod config;
pub mod kv_cache;
pub mod loader;
pub mod name_map;
pub mod request;
pub mod sample;

pub use arch::model::Model;
pub use backend::BackendKind;
pub use config::{ModelConfig, ModelFamily};
pub use kv_cache::KvCacheSlot;
pub use loader::{load_range, LoadedWeights};
pub use request::RequestState;   // added in Task 17 — declare here for forward compat
pub use sample::{sample, SamplingConfig};
```

(If Task 17's `request.rs` doesn't exist yet at the time you do Task 15, gate the `pub mod request;` line behind a `#[cfg(not_yet_used)]` placeholder OR move the lib.rs update to inside Task 17.)

- [ ] **Step 2: Backend factory**

```rust
//! Backend selection. v0.2 supports CPU (ndarray), CUDA, WGPU (covers Metal).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind { Cpu, Cuda, Metal, Wgpu }

impl BackendKind {
    pub fn from_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda),
            "metal" => Ok(Self::Metal),
            "wgpu" => Ok(Self::Wgpu),
            other => anyhow::bail!("unknown backend kind: {other}"),
        }
    }
}

#[cfg(feature = "backend-cpu")]
pub type CpuBackend = burn_ndarray::NdArray;

#[cfg(feature = "backend-cuda")]
pub type CudaBackend = burn_cuda::Cuda;

#[cfg(feature = "backend-wgpu")]
pub type WgpuBackend = burn_wgpu::Wgpu;
```

- [ ] **Step 3: Verify + commit**

```bash
cargo test -p ai-engine-runtime
cargo clippy --workspace --all-targets -- -D warnings
git add -A
git commit -m "feat(runtime): public surface + backend kind factory"
```

---

### Task 16: Backend parity test (CPU vs WGPU)

**Files:**
- Create: `crates/ai-engine-runtime/tests/backend_parity.rs`

- [ ] **Step 1: Test**

```rust
#[cfg(all(feature = "backend-cpu", feature = "backend-wgpu"))]
mod parity {
    use ai_engine_runtime::arch::model::Model;
    use ai_engine_runtime::config::ModelConfig;
    use ai_engine_runtime::loader::load_range;
    use ai_engine_tokenizer::{HfTokenizer, Tokenizer};
    use burn::tensor::Tensor;
    use std::path::PathBuf;

    fn fixture() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/toy-llama-3")
    }

    #[test]
    fn cpu_and_wgpu_produce_matching_logits() {
        let fix = fixture();
        let cfg = ModelConfig::from_file(&fix.join("config.json")).unwrap();
        let tok = HfTokenizer::from_path(fix.join("tokenizer.json")).unwrap();
        let prompt = "The quick brown fox";
        let ids: Vec<i32> = tok.encode(prompt).unwrap().iter().map(|x| *x as i32).collect();

        type Cpu = burn_ndarray::NdArray;
        let cpu_dev = Default::default();
        let cpu_w = load_range::<Cpu>(&fix.join("model.safetensors"), &cfg, 0..cfg.n_layers, true, true, &cpu_dev).unwrap();
        let cpu_model = Model::<Cpu>::from_loaded(&cfg, cpu_w, &cpu_dev).unwrap();
        let cpu_input = Tensor::<Cpu, 2, burn::tensor::Int>::from_data(
            burn::tensor::TensorData::new(ids.clone(), [1, ids.len()]), &cpu_dev,
        );
        let cpu_logits: Vec<f32> = cpu_model.forward(cpu_input, 0)
            .slice([0..1, (ids.len()-1)..ids.len(), 0..cfg.vocab_size])
            .reshape([cfg.vocab_size]).into_data().to_vec().unwrap();

        type Wgpu = burn_wgpu::Wgpu;
        let wgpu_dev = burn_wgpu::WgpuDevice::default();
        let wgpu_w = load_range::<Wgpu>(&fix.join("model.safetensors"), &cfg, 0..cfg.n_layers, true, true, &wgpu_dev).unwrap();
        let wgpu_model = Model::<Wgpu>::from_loaded(&cfg, wgpu_w, &wgpu_dev).unwrap();
        let wgpu_input = Tensor::<Wgpu, 2, burn::tensor::Int>::from_data(
            burn::tensor::TensorData::new(ids.clone(), [1, ids.len()]), &wgpu_dev,
        );
        let wgpu_logits: Vec<f32> = wgpu_model.forward(wgpu_input, 0)
            .slice([0..1, (ids.len()-1)..ids.len(), 0..cfg.vocab_size])
            .reshape([cfg.vocab_size]).into_data().to_vec().unwrap();

        let max_diff: f32 = cpu_logits.iter().zip(wgpu_logits.iter())
            .map(|(a, b)| (a - b).abs()).fold(0., f32::max);
        // WGPU tolerance is looser than CPU-vs-CPU because of slightly
        // different reduction ordering in shaders.
        assert!(max_diff < 5e-3, "CPU vs WGPU max diff = {max_diff}");
    }
}
```

This test only runs when both features are enabled. In CI environments without a GPU (no Vulkan/Metal device), `burn-wgpu` may fail to initialize — annotate `#[ignore]` if the implementer's environment lacks a device.

- [ ] **Step 2: Verify + commit**

```bash
cargo test -p ai-engine-runtime --test backend_parity
git add -A
git commit -m "test(runtime): CPU vs WGPU backend parity within shader-precision tolerance"
```

---

### Task 17: Per-block KV cache lifecycle (`RequestState`)

**Files:**
- Create: `crates/ai-engine-runtime/src/request.rs`
- Modify: `crates/ai-engine-runtime/src/lib.rs`
- Create: `crates/ai-engine-runtime/tests/request_lifecycle.rs`

Forward-looking infrastructure for Plan 2's cluster path. A `RequestState` bundles one cache slot per layer + the current `current_pos` and is passed across forward calls.

- [ ] **Step 1: Implement**

```rust
use crate::config::ModelConfig;
use crate::kv_cache::KvCacheSlot;
use burn::tensor::backend::Backend;

/// All the per-request state that persists across token generations.
pub struct RequestState<B: Backend> {
    pub caches: Vec<KvCacheSlot<B>>,
    pub current_pos: usize,
}

impl<B: Backend> RequestState<B> {
    pub fn new(cfg: &ModelConfig, batch: usize, max_tokens: usize, device: &B::Device) -> Self {
        let caches = (0..cfg.n_layers).map(|_| {
            KvCacheSlot::<B>::new(batch, cfg.n_kv_heads, max_tokens, cfg.head_dim, device)
        }).collect();
        Self { caches, current_pos: 0 }
    }

    pub fn advance(&mut self, n: usize) { self.current_pos += n; }
}
```

`crates/ai-engine-runtime/src/lib.rs` (add):

```rust
pub mod request;
pub use request::RequestState;
```

- [ ] **Step 2: Test**

```rust
use ai_engine_runtime::request::RequestState;
use ai_engine_runtime::config::{ModelConfig, ModelFamily};

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
```

- [ ] **Step 3: Verify + commit**

```bash
cargo test -p ai-engine-runtime --test request_lifecycle
git add -A
git commit -m "feat(runtime): RequestState bundling per-block KV caches for request lifecycles"
```

---

### Task 18: End-of-plan verification + tag + README

**Files:** README.md modification + git tag

- [ ] **Step 1: Full workspace verification**

```bash
cd /home/alessio/aip/airproxy
cargo test --workspace 2>&1 | grep -E "^test result:" | awk '{sum += $4} END {print "TOTAL_PASSED=" sum}'
# Expected: 78 (baseline) + ~30 (new in this plan) = ~108 tests.
cargo clippy --workspace --all-targets -- -D warnings
cargo build --workspace --release
```

- [ ] **Step 2: Update README**

Append to README.md:

```markdown
## Single-node inference (v0.2-alpha preview)

ai-engine v0.2-alpha can load a Llama-3-family safetensors checkpoint
and run inference directly — no cluster yet. See the test fixture at
`crates/ai-engine-runtime/fixtures/toy-llama-3/` for the canonical example.

This is preview functionality; the v0.2.0 release will require the cluster
configuration described in
`docs/superpowers/specs/2026-05-23-ai-engine-distributed-inference-design.md`.
Plan 2 (in `docs/superpowers/plans/`) describes the cluster work.
```

- [ ] **Step 3: Commit + tag**

```bash
git add README.md
git commit -m "docs: announce single-node runtime preview (v0.2.0-alpha.1)"
git tag v0.2.0-alpha.1
git log --oneline -5
git tag | grep v0.2
```

---

## Self-review

**Spec coverage:**

| Spec requirement (from §4 of the design spec) | Plan 1 task |
|---|---|
| `ai-engine-tokenizer` wrapping HF tokenizers | Task 1, Task 2 |
| `ai-engine-runtime` parameterized transformer (RoPE/GQA/SwiGLU/RMSNorm) | Tasks 5, 6, 7, 8 |
| `ModelConfig` + HF config.json loader | Task 3 |
| `WeightNameMap` for Llama/Mistral/Qwen/DeepSeek | Task 9 |
| safetensors loader with `load_range` | Task 10 |
| `Model::from_loaded` constructor | Task 11 |
| Bytes-tolerant correctness gate | Task 12 |
| Multi-step generation / KV cache cross-check | Task 13 |
| Sampling (greedy/temp/top-p/top-k) | Task 14 |
| Backend abstraction (4 burn backends) | Tasks 3, 15, 16 |
| `RequestState` for per-request cache lifecycle (forward-looking) | Task 17 |
| README + tag for v0.2.0-alpha.1 | Task 18 |

Everything in §4 of the design spec (model layer) is covered. The cluster pieces (§§5–8) are deferred to Plan 2.

**Placeholder scan:**

The plan contains five `todo!()` markers in code blocks:

1. `RotaryEmbedding::apply` (Task 5) — split-halves rotation; algorithm documented in prose
2. `KvCacheSlot::append` (Task 5) — slice_assign on dim 2; algorithm documented
3. `repeat_heads` for GQA (Task 7) — consecutive repetition; algorithm + Llama convention documented
4. `apply_causal_mask` (Task 7) — mask construction; algorithm documented
5. `TokenEmbedding::forward` (Task 8) — `Tensor::select`; algorithm documented
6. `Model::with_random_weights` (Task 8) — compose existing per-primitive constructors

These are deliberate handoffs to the burn API. Each `todo!()` has the algorithm in prose and code comments at the site so the implementer's job is "write the burn-specific call" rather than "design the algorithm." Not plan-failure placeholders in the writing-plans-skill sense.

**Type consistency:**

- `LoadedWeights<B>` (Task 10) → consumed by `Model::from_loaded` (Task 11). Field names match: `embedding`, `layers`, `final_norm`, `output_proj`. ✓
- `LayerWeights<B>` (Task 10) → fields match `DecoderBlock<B>` constructor (Task 11). ✓
- `RotaryEmbedding<B>::apply(x, positions: &[i32])` (Task 5) → called from `Attention::forward(x, positions, cache)` (Task 7). ✓
- `KvCacheSlot<B>::{append, read, current_len}` (Task 5) → called from `Attention::forward` (Task 7) and bundled in `RequestState` (Task 17). ✓
- `Model::forward_with_caches(token_ids, start_pos, caches)` (Task 13) → references `RequestState::caches` (Task 17). ✓
- `TensorId` enum variants (Task 9) → match the loader's loop in Task 10. ✓
- `ModelConfig` fields (Task 3) → consumed by `Model::with_random_weights` (Task 8), `Model::from_loaded` (Task 11), `RequestState::new` (Task 17). ✓

**Acknowledged risks:**

1. **Task 12 (the bytes-tolerant gate) is the riskiest single step in the plan.** Most likely 1–3 days of debugging on first attempt. Failure modes are documented; expect to iterate on Tasks 5 (RoPE) and 7 (GQA broadcast) after Task 12 reveals problems.
2. **Five `todo!()` markers** are deliberate handoffs to the burn API. Each is one well-defined operation against a publicly-documented library.
3. **Python is required for Task 4** (one-time fixture generation). If a Python env can't be set up, alternative: generate the fixture on a different machine and copy the artifacts in. The fixture itself is binary data — once present, no further Python needed.
4. **WGPU may not be available in CI** — Task 16's parity test should be `#[ignore]`d gracefully if the runner lacks a Vulkan / Metal device.
5. **burn version drift.** The plan pins to burn 0.18. If a newer version with breaking changes ships before this plan executes, expect 1–2 days of API porting. The model algorithm is stable; only the calls change.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-23-plan-1-tokenizer-and-runtime.md`. Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, two-stage review between tasks. With 18 tasks and significant ML complexity in Tasks 5–13, the per-task review checkpoints catch issues early.

**2. Inline Execution** — possible but ill-advised for a plan of this length and depth.

Plan 2 (cluster) will be written after Plan 1's `v0.2.0-alpha.1` tag lands.
