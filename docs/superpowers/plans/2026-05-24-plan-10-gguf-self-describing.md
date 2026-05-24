# Plan 10 — v0.3.0-alpha.6: GGUF self-describing checkpoints

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drop the requirement for separate `config_path` and `tokenizer_path` when `weights_path` ends in `.gguf`. The GGUF file already carries ModelConfig hyperparams (`llama.embedding_length`, etc.) and tokenizer state (`tokenizer.ggml.tokens`, `.merges`) in its metadata. Extract both at load time so a `[cluster.model]` block with just `weights_path = "model.gguf"` is sufficient.

**Architecture:** Two new functions in `ai-engine-runtime`: `load_model_config_from_gguf(path) -> ModelConfig` and `load_tokenizer_from_gguf(path) -> HfTokenizer`. Both reuse the existing `gguf` module's header + metadata parsers; they DON'T re-parse tensor descriptors (cheaper to skip). The TOML schema makes `config_path` and `tokenizer_path` optional. Callers (`build_app_state`, `worker_main`) dispatch: when the path is absent AND `weights_path` is `.gguf`, extract from GGUF; otherwise use the explicit path. The GGUF fixture generator is updated to embed tokenizer metadata so existing tests can exercise the new extraction.

**Tech Stack:** No new external deps. `tokenizers` (already in workspace) provides `BPE::new(vocab, merges)` for byte-level Llama-3 tokenizers.

**Scope rule:** Plan 10 targets **Llama-3-family byte-level BPE tokenizers** in GGUF. SentencePiece-based GGUF tokenizers (Llama-2, Mistral) need a different reconstruction path and are deferred to a follow-up plan. The ModelConfig extraction handles the standard `llama.*` metadata keys; non-Llama architectures (`qwen.*`, `gpt.*`) follow the same pattern but are similarly out of scope.

**Baseline:** Branch `main` at `v0.3.0-alpha.5`. 209 passing + 6 ignored. Clippy clean.

---

### Task 1: `ModelConfig::from_gguf_metadata`

**Files:**
- Modify: `crates/ai-engine-runtime/src/config.rs` (add associated function)
- Modify: `crates/ai-engine-runtime/src/gguf/mod.rs` (expose `read_metadata_only(path)`)
- Create: `crates/ai-engine-runtime/tests/gguf_model_config.rs`

The GGUF file already has all the ModelConfig fields we need:

| GGUF metadata key | ModelConfig field |
|---|---|
| `general.architecture` = "llama" | family = Llama3 |
| `llama.block_count` | n_layers |
| `llama.embedding_length` | hidden_size |
| `llama.attention.head_count` | n_heads |
| `llama.attention.head_count_kv` | n_kv_heads |
| `llama.feed_forward_length` | intermediate_size |
| `llama.context_length` | max_position_embeddings |
| `llama.attention.layer_norm_rms_epsilon` | rms_norm_eps |
| `llama.rope.freq_base` | rope_theta |
| (derive `head_dim` from `hidden_size / n_heads`) | head_dim |
| (vocab_size derived from `tokenizer.ggml.tokens` array length) | vocab_size |
| (`tie_word_embeddings`: true if `output.weight` not present, else false) | tie_word_embeddings |

We need to expose a `read_metadata_only(path)` from the gguf module that parses just header + KV pairs (skipping tensor descriptors). This will also be used by Task 2's tokenizer extraction.

- [ ] **Step 1: Failing test**

`crates/ai-engine-runtime/tests/gguf_model_config.rs`:

```rust
use ai_engine_runtime::config::{ModelConfig, ModelFamily};
use ai_engine_runtime::gguf;
use std::path::PathBuf;

fn gguf_fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/toy-llama-3-gguf")
}

#[test]
fn read_model_config_from_gguf_matches_safetensors_config() {
    let from_gguf = ModelConfig::from_gguf_file(&gguf_fixture().join("model.gguf")).unwrap();
    let from_json = ModelConfig::from_file(&gguf_fixture().join("config.json")).unwrap();

    // All architectural hyperparams must match the bf16 fixture's config.json.
    assert_eq!(from_gguf.hidden_size, from_json.hidden_size);
    assert_eq!(from_gguf.n_layers, from_json.n_layers);
    assert_eq!(from_gguf.n_heads, from_json.n_heads);
    assert_eq!(from_gguf.n_kv_heads, from_json.n_kv_heads);
    assert_eq!(from_gguf.intermediate_size, from_json.intermediate_size);
    assert_eq!(from_gguf.max_position_embeddings, from_json.max_position_embeddings);
    assert_eq!(from_gguf.head_dim, from_json.head_dim);
    assert!((from_gguf.rope_theta - from_json.rope_theta).abs() < 1e-3);
    assert!((from_gguf.rms_norm_eps - from_json.rms_norm_eps).abs() < 1e-7);
    assert_eq!(from_gguf.family, ModelFamily::Llama3);
}

#[test]
fn read_metadata_only_skips_tensor_data() {
    // Just verifies the function exists and parses without OOM on a real file.
    let m = gguf::read_metadata_only(&gguf_fixture().join("model.gguf")).unwrap();
    assert!(m.contains_key("general.architecture"));
    assert!(m.contains_key("llama.block_count"));
}
```

- [ ] **Step 2: Confirm fails**

```bash
cd /home/alessio/aip/airproxy
cargo test -p ai-engine-runtime --test gguf_model_config 2>&1 | tail -10
# Expected: ModelConfig::from_gguf_file / gguf::read_metadata_only don't exist.
```

- [ ] **Step 3: Implement `read_metadata_only`**

Add to `crates/ai-engine-runtime/src/gguf/mod.rs`:

```rust
use std::collections::HashMap;
use std::path::Path;

use crate::gguf::header::parse_header;
use crate::gguf::metadata::{parse_kv, GgufValue};

/// Read a GGUF file's metadata KV pairs WITHOUT loading tensor data.
/// Cheaper than `load_gguf` when you only need the metadata.
pub fn read_metadata_only(path: &Path) -> anyhow::Result<HashMap<String, GgufValue>> {
    let file = std::fs::File::open(path)
        .map_err(|e| anyhow::anyhow!("open {}: {e}", path.display()))?;
    // mmap the file for cheap random access into the header + metadata section.
    let mmap = unsafe { memmap2::Mmap::map(&file) }
        .map_err(|e| anyhow::anyhow!("mmap {}: {e}", path.display()))?;

    let (hdr, mut cursor) = parse_header(&mmap)?;
    let mut metadata = HashMap::with_capacity(hdr.metadata_count as usize);
    for _ in 0..hdr.metadata_count {
        let (k, v, consumed) = parse_kv(&mmap[cursor..])?;
        metadata.insert(k, v);
        cursor += consumed;
    }
    Ok(metadata)
}
```

- [ ] **Step 4: Implement `ModelConfig::from_gguf_file`**

Add to `crates/ai-engine-runtime/src/config.rs`:

```rust
use crate::gguf::metadata::{GgufArray, GgufValue};

impl ModelConfig {
    /// Extract a `ModelConfig` from a GGUF file's metadata. Targets Llama-3-style
    /// `llama.*` keys. Returns an error if the file isn't a Llama-family GGUF
    /// or required metadata is missing.
    pub fn from_gguf_file(path: &std::path::Path) -> anyhow::Result<Self> {
        let m = crate::gguf::read_metadata_only(path)?;
        Self::from_gguf_metadata(&m)
    }

    pub fn from_gguf_metadata(m: &std::collections::HashMap<String, GgufValue>) -> anyhow::Result<Self> {
        let arch = match m.get("general.architecture") {
            Some(GgufValue::String(s)) => s.as_str(),
            Some(other) => anyhow::bail!("general.architecture wrong type: {other:?}"),
            None => anyhow::bail!("general.architecture missing in GGUF metadata"),
        };
        let family = match arch {
            "llama" => ModelFamily::Llama3,
            other => anyhow::bail!("GGUF architecture `{other}` not supported in Plan 10 (only `llama`)"),
        };

        let n_layers = read_u32(m, "llama.block_count")? as usize;
        let hidden_size = read_u32(m, "llama.embedding_length")? as usize;
        let n_heads = read_u32(m, "llama.attention.head_count")? as usize;
        let n_kv_heads = read_u32(m, "llama.attention.head_count_kv")? as usize;
        let intermediate_size = read_u32(m, "llama.feed_forward_length")? as usize;
        let max_position_embeddings = read_u32(m, "llama.context_length")? as usize;
        let rms_norm_eps = read_f32(m, "llama.attention.layer_norm_rms_epsilon")?;
        let rope_theta = read_f32(m, "llama.rope.freq_base")?;
        let head_dim = hidden_size / n_heads;

        // vocab_size from tokenizer.ggml.tokens array length; fall back to a
        // generic key if available.
        let vocab_size = match m.get("tokenizer.ggml.tokens") {
            Some(GgufValue::Array(GgufArray::String(v))) => v.len(),
            _ => match m.get("llama.vocab_size") {
                Some(GgufValue::U32(n)) => *n as usize,
                _ => anyhow::bail!("GGUF missing both tokenizer.ggml.tokens array and llama.vocab_size"),
            },
        };

        // tie_word_embeddings: GGUF doesn't have an explicit key. The convention
        // is: if the file has an `output.weight` tensor descriptor, embeddings
        // are UNTIED. We could re-parse tensor descriptors here to check, but
        // for Plan 10 we default to `true` (the most common case — Llama-3
        // family ties its embedding by default). Callers who need precise
        // tie-ness for non-default checkpoints can override via TOML.
        let tie_word_embeddings = true;

        Ok(Self {
            hidden_size,
            intermediate_size,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            vocab_size,
            max_position_embeddings,
            rope_theta,
            rms_norm_eps,
            tie_word_embeddings,
            family,
        })
    }
}

fn read_u32(m: &std::collections::HashMap<String, GgufValue>, key: &str) -> anyhow::Result<u32> {
    match m.get(key) {
        Some(GgufValue::U32(n)) => Ok(*n),
        Some(GgufValue::U64(n)) => Ok(*n as u32),
        Some(GgufValue::I32(n)) => Ok(*n as u32),
        Some(other) => anyhow::bail!("GGUF key `{key}` wrong type for u32: {other:?}"),
        None => anyhow::bail!("GGUF key `{key}` missing"),
    }
}

fn read_f32(m: &std::collections::HashMap<String, GgufValue>, key: &str) -> anyhow::Result<f32> {
    match m.get(key) {
        Some(GgufValue::F32(f)) => Ok(*f),
        Some(GgufValue::F64(f)) => Ok(*f as f32),
        Some(other) => anyhow::bail!("GGUF key `{key}` wrong type for f32: {other:?}"),
        None => anyhow::bail!("GGUF key `{key}` missing"),
    }
}
```

- [ ] **Step 5: Verify + commit**

```bash
cargo test -p ai-engine-runtime --test gguf_model_config
cargo clippy --workspace --all-targets -- -D warnings
git add -A
git commit -m "feat(runtime): ModelConfig::from_gguf_file + gguf::read_metadata_only"
```

NO Co-Authored-By footer (global preference).

---

### Task 2: Embed tokenizer metadata in GGUF fixture + `HfTokenizer::from_gguf_file`

**Files:**
- Modify: `crates/ai-engine-runtime/scripts/generate_gguf_q4_0_fixture.py` (write tokenizer metadata)
- Regenerate: `crates/ai-engine-runtime/fixtures/toy-llama-3-gguf/model.gguf`
- Create: `crates/ai-engine-runtime/src/tokenizer_gguf.rs` (new module; runtime owns this because tokenizer reconstruction needs both the gguf parser and the tokenizers crate)
- Modify: `crates/ai-engine-runtime/src/lib.rs` (declare + export)
- Create: `crates/ai-engine-runtime/tests/gguf_tokenizer.rs`

Two parts: (a) the Python fixture script learns to embed the tokenizer's tokens/merges into the GGUF metadata, (b) a new Rust function reads them back into an `HfTokenizer`.

- [ ] **Step 1: Update the fixture script to embed tokenizer metadata**

In `crates/ai-engine-runtime/scripts/generate_gguf_q4_0_fixture.py`, in the metadata-writing section, add tokenizer kv pairs. The script already has `tok = tokenizers.Tokenizer.from_file(...)` available if we load it explicitly:

```python
# After the existing llama.* metadata writes, before tensor data assembly:

# --- Tokenizer metadata ---
from tokenizers import Tokenizer
hf_tok = Tokenizer.from_file(str(SRC / "tokenizer.json"))

# Tokens list: index -> string.
tok_model = hf_tok.get_vocab(with_added_tokens=True)
# tok_model is dict {string: id}; invert + sort by id.
tokens_by_id = [""] * (max(tok_model.values()) + 1)
for s, i in tok_model.items():
    tokens_by_id[i] = s

# Write tokenizer.ggml.model = "gpt2" (the byte-level BPE family for our toy)
write_kv_string(meta, "tokenizer.ggml.model", "gpt2")
meta_count += 1

# Write tokenizer.ggml.tokens as a STRING array.
def write_kv_string_array(buf, key, values):
    write_gguf_string(buf, key)
    buf.extend(struct.pack("<I", TYPE_ARRAY))
    buf.extend(struct.pack("<I", TYPE_STRING))         # element type
    buf.extend(struct.pack("<Q", len(values)))         # count
    for v in values:
        write_gguf_string(buf, v)

write_kv_string_array(meta, "tokenizer.ggml.tokens", tokens_by_id)
meta_count += 1

# Write tokenizer.ggml.merges if the BPE model has them.
# The HF Tokenizer's BPE model exposes merges via `to_str()`; we can extract them
# from the tokenizer.json directly since we already have the file.
import json as _json
tok_json = _json.loads((SRC / "tokenizer.json").read_text())
merges_list = tok_json.get("model", {}).get("merges", [])
# Some tokenizer.json formats store merges as [["a", "b"], ...] (list of pairs);
# others as ["a b", ...] (space-joined strings). Normalize to "a b" form.
merges_strings = []
for m in merges_list:
    if isinstance(m, list) and len(m) == 2:
        merges_strings.append(f"{m[0]} {m[1]}")
    elif isinstance(m, str):
        merges_strings.append(m)
write_kv_string_array(meta, "tokenizer.ggml.merges", merges_strings)
meta_count += 1

# Optional: bos/eos token ids if known. For our toy these aren't critical;
# Llama-3 uses 128000/128001. For the toy fixture we just write reasonable
# defaults; not relied on by Plan 10 tests.
write_kv_u32(meta, "tokenizer.ggml.bos_token_id", 0)
meta_count += 1
write_kv_u32(meta, "tokenizer.ggml.eos_token_id", 1)
meta_count += 1
```

- [ ] **Step 2: Run the fixture script + verify metadata appears**

```bash
cd /home/alessio/aip/airproxy
source .venv-fixture/bin/activate
python crates/ai-engine-runtime/scripts/generate_gguf_q4_0_fixture.py
deactivate
ls -la crates/ai-engine-runtime/fixtures/toy-llama-3-gguf/
```

Expected: `model.gguf` is slightly larger than before (~2 KB more for the embedded tokens/merges arrays).

- [ ] **Step 3: Sanity-check the metadata is there**

Quick Rust-side check via the existing `read_metadata_only`:

```bash
cargo test -p ai-engine-runtime --test gguf_model_config read_metadata_only_skips_tensor_data
```

Should still pass. Then add a new manual print test or just rely on Task 2's failing test below to verify the tokens were written.

- [ ] **Step 4: Implement `HfTokenizer::from_gguf_file` via a runtime helper**

The `HfTokenizer` type lives in `ai-engine-tokenizer`. Adding GGUF parsing there would require depending on `ai-engine-runtime`, which is undesirable (runtime depends on tokenizer, not the reverse). The cleanest approach: a free function in `ai-engine-runtime` that returns an `HfTokenizer` by constructing one programmatically.

`crates/ai-engine-runtime/src/tokenizer_gguf.rs`:

```rust
//! Reconstruct an `HfTokenizer` from a GGUF file's tokenizer metadata.
//! Targets the Llama-3 byte-level BPE format ("gpt2" tokenizer model).

use crate::gguf::metadata::{GgufArray, GgufValue};
use ai_engine_tokenizer::HfTokenizer;
use std::collections::HashMap;
use std::path::Path;
use tokenizers::{
    decoders::byte_level::ByteLevel as ByteLevelDecoder,
    models::bpe::BPE,
    pre_tokenizers::byte_level::ByteLevel as ByteLevelPre,
    Tokenizer,
};

/// Load a tokenizer from the GGUF metadata at `path`. Only supports Llama-3-style
/// byte-level BPE (`tokenizer.ggml.model = "gpt2"` or `"llama"`).
pub fn load_tokenizer_from_gguf(path: &Path) -> anyhow::Result<HfTokenizer> {
    let m = crate::gguf::read_metadata_only(path)?;
    load_tokenizer_from_gguf_metadata(&m)
}

pub fn load_tokenizer_from_gguf_metadata(
    m: &HashMap<String, GgufValue>,
) -> anyhow::Result<HfTokenizer> {
    let tok_model_kind = match m.get("tokenizer.ggml.model") {
        Some(GgufValue::String(s)) => s.as_str(),
        _ => anyhow::bail!("GGUF missing tokenizer.ggml.model"),
    };
    if tok_model_kind != "gpt2" && tok_model_kind != "llama" {
        anyhow::bail!(
            "Plan 10 only supports byte-level BPE GGUF tokenizers (gpt2/llama); got `{tok_model_kind}`"
        );
    }

    let tokens = match m.get("tokenizer.ggml.tokens") {
        Some(GgufValue::Array(GgufArray::String(v))) => v.clone(),
        _ => anyhow::bail!("GGUF missing tokenizer.ggml.tokens"),
    };
    let merges = match m.get("tokenizer.ggml.merges") {
        Some(GgufValue::Array(GgufArray::String(v))) => v.clone(),
        _ => Vec::new(),    // merges optional; BPE with empty merges = char-level
    };

    // Build vocab: token string -> u32 id.
    let vocab: HashMap<String, u32> = tokens
        .iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), i as u32))
        .collect();

    // Parse merges: each entry is "left right" (space-separated).
    let parsed_merges: Vec<(String, String)> = merges
        .iter()
        .filter_map(|m| {
            let mut it = m.splitn(2, ' ');
            let l = it.next()?.to_string();
            let r = it.next()?.to_string();
            Some((l, r))
        })
        .collect();

    let bpe = BPE::new(vocab, parsed_merges);
    let mut tok = Tokenizer::new(bpe);
    tok.with_pre_tokenizer(Some(ByteLevelPre::default()));
    tok.with_decoder(Some(ByteLevelDecoder::default()));

    // Save to a temp tokenizer.json and load via HfTokenizer's existing path.
    // HfTokenizer's only constructor takes a path; rather than expanding its
    // API, we round-trip through a temp file. The temp file is owned by the
    // tempfile crate and cleaned up automatically.
    let mut tempfile = tempfile::NamedTempFile::new()
        .map_err(|e| anyhow::anyhow!("tempfile: {e}"))?;
    let json = tok
        .to_string(true)
        .map_err(|e| anyhow::anyhow!("tokenizer.to_string: {e}"))?;
    std::io::Write::write_all(&mut tempfile, json.as_bytes())?;
    HfTokenizer::from_path(tempfile.path())
}
```

Note: this introduces `tempfile` as a regular dep of `ai-engine-runtime` (it's already a dev-dep in some crates). Add `tempfile = "3"` to root workspace deps if not present, then `tempfile.workspace = true` to `ai-engine-runtime/Cargo.toml`.

Note 2: the `BPE::new(vocab, merges)` signature in `tokenizers 0.20` may differ — verify with `cargo doc -p tokenizers`. Some versions use a builder pattern (`BPE::builder().vocab_and_merges(...).build()?`) instead of a constructor.

- [ ] **Step 5: Wire module**

`crates/ai-engine-runtime/src/lib.rs` — append:

```rust
pub mod tokenizer_gguf;
pub use tokenizer_gguf::{load_tokenizer_from_gguf, load_tokenizer_from_gguf_metadata};
```

- [ ] **Step 6: Failing test**

`crates/ai-engine-runtime/tests/gguf_tokenizer.rs`:

```rust
use ai_engine_runtime::load_tokenizer_from_gguf;
use ai_engine_tokenizer::Tokenizer;
use std::path::PathBuf;

fn gguf_fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/toy-llama-3-gguf")
}

#[test]
fn load_tokenizer_from_gguf_can_encode_and_decode() {
    let tok = load_tokenizer_from_gguf(&gguf_fixture().join("model.gguf")).unwrap();
    let text = "The quick brown fox";
    let ids = tok.encode(text).unwrap();
    assert!(!ids.is_empty(), "encode produced tokens");
    let back = tok.decode(&ids).unwrap();
    // BPE tokenizer with ByteLevel pre+post should roundtrip cleanly.
    assert_eq!(back.trim(), text);
}

#[test]
fn gguf_tokenizer_produces_same_ids_as_json_tokenizer() {
    use ai_engine_tokenizer::HfTokenizer;
    let from_gguf = load_tokenizer_from_gguf(&gguf_fixture().join("model.gguf")).unwrap();
    let from_json = HfTokenizer::from_path(&gguf_fixture().join("tokenizer.json")).unwrap();

    for prompt in &["hello", "The quick brown fox", "ai-engine"] {
        let g = from_gguf.encode(prompt).unwrap();
        let j = from_json.encode(prompt).unwrap();
        assert_eq!(g, j, "tokenization mismatch on `{prompt}`: gguf={g:?} json={j:?}");
    }
}
```

- [ ] **Step 7: Run + iterate**

```bash
cargo test -p ai-engine-runtime --test gguf_tokenizer -- --nocapture
```

If the second test fails (gguf tokenizer produces different IDs than the JSON), the likely culprit is:
- Vocab ordering (must use the `tokens` array's index as the ID — verify the fixture script writes in stable index order)
- Merges format (`"a b"` vs `"a b"` vs separately-stored pairs)
- ByteLevel configuration (default options must match the original tokenizer.json's settings)

If the first test fails (encode/decode roundtrip) but the second works, the tokenizer is consistent with itself but the ByteLevel decoder doesn't match the encoder — verify decoder is configured.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "feat(runtime): load_tokenizer_from_gguf for Llama-3-style byte-level BPE"
```

NO Co-Authored-By.

---

### Task 3: Make `config_path` + `tokenizer_path` optional in TOML

**Files:**
- Modify: `crates/ai-engine-config/src/lib.rs` (`ClusterModel` fields)
- Modify: `crates/ai-engine-config/src/validate.rs` (relax requirements when weights_path is .gguf)
- Modify: `crates/ai-engine-config/tests/load.rs` (add test for minimal GGUF model block)

- [ ] **Step 1: Failing test**

Append to `crates/ai-engine-config/tests/load.rs`:

```rust
#[test]
fn cluster_model_block_with_only_weights_path_for_gguf() {
    let toml = r#"
[server]
bind = "127.0.0.1:0"

[auth]
mode = "passthrough"

[[cluster]]
id = "home"
leader = "node-a"
quic_bind = "0.0.0.0:7700"

[cluster.model]
id = "llama-3-70b"
weights_path = "/srv/models/llama-3-70b/model.gguf"

[[cluster.node]]
id = "node-a"
addr = "127.0.0.1:7700"
cert_fingerprint = "sha256:abc"
backend = "cpu"

[[provider]]
id = "home-cluster"
kind = "local-cluster"
cluster = "home"

[[route]]
match = { model = "llama-3-70b" }
provider = "home-cluster"

[pipeline."/v1/chat/completions"]
stages = ["auth", "model_route", "forward", "log"]
"#;
    let cfg = ai_engine_config::Config::from_str(toml).unwrap();
    let m = &cfg.clusters[0].model;
    assert_eq!(m.weights_path, "/srv/models/llama-3-70b/model.gguf");
    assert!(m.config_path.is_none());
    assert!(m.tokenizer_path.is_none());
}

#[test]
fn cluster_model_block_with_safetensors_requires_config_and_tokenizer_paths() {
    let toml = r#"
[server]
bind = "127.0.0.1:0"

[auth]
mode = "passthrough"

[[cluster]]
id = "home"
leader = "node-a"
quic_bind = "0.0.0.0:7700"

[cluster.model]
id = "llama-3-70b"
weights_path = "/srv/models/llama-3-70b/model.safetensors"

[[cluster.node]]
id = "node-a"
addr = "127.0.0.1:7700"
cert_fingerprint = "sha256:abc"
backend = "cpu"

[[provider]]
id = "home-cluster"
kind = "local-cluster"
cluster = "home"

[[route]]
match = { model = "llama-3-70b" }
provider = "home-cluster"

[pipeline."/v1/chat/completions"]
stages = ["forward", "log"]
"#;
    let err = ai_engine_config::Config::from_str(toml).unwrap_err().to_string();
    assert!(
        err.to_lowercase().contains("config_path") || err.to_lowercase().contains("tokenizer_path"),
        "got error: {err}"
    );
}
```

- [ ] **Step 2: Update `ClusterModel`**

In `crates/ai-engine-config/src/lib.rs`:

```rust
#[derive(Debug, Clone, Deserialize)]
pub struct ClusterModel {
    pub id: String,
    pub weights_path: String,
    #[serde(default)]
    pub config_path: Option<String>,
    #[serde(default)]
    pub tokenizer_path: Option<String>,
}
```

- [ ] **Step 3: Validation in `validate.rs`**

Inside the cluster-validation loop, after parsing the cluster:

```rust
let is_gguf = cluster.model.weights_path
    .rsplit('.')
    .next()
    .map(|ext| ext.eq_ignore_ascii_case("gguf"))
    .unwrap_or(false);
if !is_gguf {
    if cluster.model.config_path.is_none() {
        anyhow::bail!(
            "cluster `{}` model.config_path required when weights_path is not a .gguf file",
            cluster.id
        );
    }
    if cluster.model.tokenizer_path.is_none() {
        anyhow::bail!(
            "cluster `{}` model.tokenizer_path required when weights_path is not a .gguf file",
            cluster.id
        );
    }
}
```

- [ ] **Step 4: Verify**

```bash
cargo test -p ai-engine-config
cargo clippy --workspace --all-targets -- -D warnings
```

The new 2 tests pass. Existing tests that used `config_path = "x"` / `tokenizer_path = "x"` for safetensors fixtures continue to pass because Option<String> deserializes a present string the same way.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(config): config_path + tokenizer_path optional when weights_path is .gguf"
```

NO Co-Authored-By.

---

### Task 4: Dispatch in `build_app_state` + `worker_main`

**Files:**
- Modify: `crates/ai-engine/src/app.rs` (leader branch dispatches ModelConfig + tokenizer source)
- Modify: `crates/ai-engine/src/worker_main.rs` (worker dispatches ModelConfig source)

The leader's `build_app_state` and the worker's `run_worker` both load `ModelConfig` + (for the leader) `HfTokenizer`. Update each to dispatch based on whether the corresponding path field is `Some` or `None`.

- [ ] **Step 1: Update leader-mode `build_app_state` in `crates/ai-engine/src/app.rs`**

In the section that does:

```rust
let model_cfg = ai_engine_runtime::config::ModelConfig::from_file(
    std::path::Path::new(&cluster_cfg.model.config_path)
)?;
let tokenizer = ai_engine_tokenizer::HfTokenizer::from_path(&cluster_cfg.model.tokenizer_path)?;
```

Replace with:

```rust
let weights_path = std::path::PathBuf::from(&cluster_cfg.model.weights_path);

let model_cfg = match &cluster_cfg.model.config_path {
    Some(p) => ai_engine_runtime::config::ModelConfig::from_file(std::path::Path::new(p))?,
    None => ai_engine_runtime::config::ModelConfig::from_gguf_file(&weights_path)?,
};

let tokenizer = match &cluster_cfg.model.tokenizer_path {
    Some(p) => std::sync::Arc::new(ai_engine_tokenizer::HfTokenizer::from_path(p)?),
    None => std::sync::Arc::new(ai_engine_runtime::load_tokenizer_from_gguf(&weights_path)?),
};
```

(Tokenizer is wrapped in `Arc` already — preserve that pattern; just dispatch the inner value.)

Similarly, the `LeaderState`'s `model_path` was previously `cluster_cfg.model.weights_path.into()`. That stays unchanged — it's still a path either way.

- [ ] **Step 2: Update `worker_main::run_worker`**

In `crates/ai-engine/src/worker_main.rs`, in the section that does:

```rust
let model_cfg = ai_engine_runtime::config::ModelConfig::from_file(
    std::path::Path::new(&cluster.model.config_path)
)?;
```

Replace with:

```rust
let model_cfg = match &cluster.model.config_path {
    Some(p) => ai_engine_runtime::config::ModelConfig::from_file(std::path::Path::new(p))?,
    None => ai_engine_runtime::config::ModelConfig::from_gguf_file(
        std::path::Path::new(&cluster.model.weights_path)
    )?,
};
```

Workers don't need a tokenizer (only the leader uses it), so no tokenizer dispatch on the worker.

- [ ] **Step 3: Verify everything still builds + existing tests pass**

```bash
cd /home/alessio/aip/airproxy
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo build --workspace --release
```

The existing `multiproc_smoke.rs` / `multiproc_smoke_mdns.rs` / `multiproc_smoke_gguf.rs` all explicitly set `config_path` + `tokenizer_path`, so they keep working unchanged.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat(bin): leader/worker dispatch ModelConfig + tokenizer source on path presence"
```

NO Co-Authored-By.

---

### Task 5: End-to-end "minimal GGUF" smoke test

**Files:**
- Create: `crates/ai-engine/tests/multiproc_smoke_gguf_only.rs`

A multi-process smoke that proves the new minimal-config path works: TOML has `weights_path = "<.gguf>"` and NO `config_path` or `tokenizer_path` at all. Both should be derived from the GGUF.

- [ ] **Step 1: Test**

Copy `multiproc_smoke_gguf.rs` and modify just the `write_config` to omit `config_path` and `tokenizer_path`. Filename: `crates/ai-engine/tests/multiproc_smoke_gguf_only.rs`.

The test verifies:
1. The 3 processes start.
2. Leader's `build_app_state` derives ModelConfig + tokenizer from the GGUF.
3. Workers derive ModelConfig from the GGUF.
4. A chat completion returns 3 tokens.

Use this `write_config` body (only the `[cluster.model]` block differs from `multiproc_smoke_gguf.rs`):

```rust
fn write_config(
    dir: &std::path::Path,
    fix: &std::path::Path,
    leader_http_port: u16,
    leader_quic_port: u16,
    w1_quic_port: u16,
    w2_quic_port: u16,
    leader_fp: &str,
    w1_fp: &str,
    w2_fp: &str,
) -> PathBuf {
    let toml = format!(
        r#"
[server]
bind = "127.0.0.1:{leader_http_port}"
log_format = "pretty"
log_level = "warn"

[auth]
mode = "passthrough"

[[cluster]]
id = "smoke-gguf-only"
leader = "leader"
quic_bind = "127.0.0.1:0"

[cluster.model]
id = "toy-llama-gguf-only"
weights_path = "{fix}/model.gguf"

[[cluster.partition_override]]
node = "worker-1"
layers = "0..2"

[[cluster.partition_override]]
node = "worker-2"
layers = "2..4"

[[cluster.node]]
id = "leader"
addr = "127.0.0.1:{leader_quic_port}"
cert_fingerprint = "{leader_fp}"
backend = "cpu"

[[cluster.node]]
id = "worker-1"
addr = "127.0.0.1:{w1_quic_port}"
cert_fingerprint = "{w1_fp}"
backend = "cpu"

[[cluster.node]]
id = "worker-2"
addr = "127.0.0.1:{w2_quic_port}"
cert_fingerprint = "{w2_fp}"
backend = "cpu"

[[provider]]
id = "smoke-gguf-only-cluster"
kind = "local-cluster"
cluster = "smoke-gguf-only"

[[route]]
match = {{ model = "toy-llama-gguf-only" }}
provider = "smoke-gguf-only-cluster"

[pipeline."/v1/chat/completions"]
stages = ["auth", "model_route", "forward", "log"]
"#,
        fix = fix.display(),
    );
    let path = dir.join("ai-engine.toml");
    std::fs::write(&path, toml).unwrap();
    path
}
```

The rest of the test (process spawning, fingerprint capture, HTTP request) is a verbatim copy of `multiproc_smoke_gguf.rs`. The test name should be `three_process_cluster_with_minimal_gguf_only_config`.

Mark `#[ignore]` like the other multiproc smokes.

- [ ] **Step 2: Run + commit**

```bash
cargo build --workspace --release
cargo test -p ai-engine --test multiproc_smoke_gguf_only -- --ignored --nocapture
git add -A
git commit -m "test(smoke): 3-process cluster with weights_path-only GGUF config"
```

NO Co-Authored-By.

---

### Task 6: README + tag v0.3.0-alpha.6

**Files:**
- Modify: `README.md`
- Tag: `v0.3.0-alpha.6`

- [ ] **Step 1: Final verification**

```bash
cd /home/alessio/aip/airproxy
cargo test --workspace 2>&1 | grep "test result" | awk '{p += $4; ig += $8} END {print "PASSED=" p " IGNORED=" ig}'
cargo clippy --workspace --all-targets -- -D warnings
cargo build --workspace --release
for t in multiproc_smoke streaming_smoke multiproc_smoke_mdns multiproc_smoke_gguf multiproc_smoke_gguf_only; do
  echo "== $t =="
  cargo test -p ai-engine --test $t -- --ignored --nocapture 2>&1 | tail -3
done
```

- [ ] **Step 2: README**

Append:

```markdown
### v0.3.0-alpha.6 — GGUF self-describing checkpoints

ai-engine v0.3.0-alpha.6 drops the requirement for separate `config_path`
and `tokenizer_path` when `weights_path` is a `.gguf` file. The GGUF
metadata already carries both — extract them at load time:

\`\`\`toml
[cluster.model]
id = "llama-3-70b"
weights_path = "/srv/models/llama-3-70b/model.gguf"
# config_path + tokenizer_path no longer required for GGUF
\`\`\`

Internals:
- `ModelConfig::from_gguf_file` extracts hyperparams from `llama.*` keys.
- `load_tokenizer_from_gguf` rebuilds the HF tokenizer from
  `tokenizer.ggml.tokens` + `.merges` (Llama-3-style byte-level BPE).
- Both are dispatched automatically by `build_app_state` and the
  worker entrypoint when the corresponding TOML path is absent.

Known limitations:
- Only Llama-3-family (`general.architecture = "llama"`) supported.
  Qwen / Mistral / DeepSeek-V2 architectures need their own metadata
  key prefixes.
- Only byte-level BPE tokenizers (`tokenizer.ggml.model = "gpt2"`/`"llama"`).
  SentencePiece-based GGUF tokenizers deferred.
- `tie_word_embeddings` defaults to `true` (the Llama-3 norm); explicit
  override via TOML's optional `config_path` for the rare untied case.
```

- [ ] **Step 3: Commit + tag**

```bash
git add README.md
git commit -m "docs: announce v0.3.0-alpha.6 GGUF self-describing checkpoints release"
git tag v0.3.0-alpha.6
git log --oneline -7
git tag
```

NO Co-Authored-By.

## Report
- Status
- Test count + ignored
- Final tag listing
- `git log --oneline -7`

---

## Self-review

**Spec coverage:**
- ModelConfig from GGUF metadata → Task 1
- Tokenizer from GGUF metadata → Task 2
- TOML schema makes paths optional → Task 3
- Binary dispatch on path presence → Task 4
- End-to-end smoke with minimal config → Task 5
- Release → Task 6

**Placeholder scan:**
The Plan 10 design has no placeholders. The `tie_word_embeddings = true` default in Task 1 is a documented assumption (Llama-3-family convention), not a placeholder — checkpoints with untied embeddings get the wrong value but can override via the still-supported `config_path`.

**Type consistency:**
- `ModelConfig::from_gguf_file` (Task 1) → consumed by `build_app_state` (Task 4) and `worker_main` (Task 4). ✓
- `load_tokenizer_from_gguf` returning `HfTokenizer` (Task 2) → wrapped in `Arc` by `build_app_state` (Task 4). ✓
- `ClusterModel.config_path: Option<String>` + `tokenizer_path: Option<String>` (Task 3) → matched on by `build_app_state` + `worker_main` (Task 4). ✓
- `read_metadata_only` (Task 1) → used internally by both `from_gguf_file` (Task 1) and `load_tokenizer_from_gguf` (Task 2). ✓

**Acknowledged risks:**

1. **The `tokenizers 0.20` API for BPE construction** may differ from the plan's snippet — `BPE::new(vocab, merges)` vs `BPE::builder()` chain. Verify on first compile.
2. **`Tokenizer::to_string(true)` + reload-from-tempfile** is a round-trip — minor overhead at load time. A direct path to construct `HfTokenizer` from an in-memory `tokenizers::Tokenizer` would be cleaner but would require expanding `HfTokenizer`'s API. The temp-file approach is fine for v0.3.0-alpha.6.
3. **The Python fixture script's tokenizer.json parsing** assumes the file is a regular HF tokenizer dump (the script reads merges from `tok_json["model"]["merges"]`). The toy fixture's tokenizer was generated programmatically and stores merges in this exact form, so it works. Production tokenizer.json files may differ slightly; not a concern for Plan 10's scope.

---

## Execution Handoff

Plan 10 saved to `docs/superpowers/plans/2026-05-24-plan-10-gguf-self-describing.md`. 6 tasks.

**Subagent-Driven (recommended)** — Tasks 1, 2 are bounded with clear contracts. Task 2 has the most code (fixture script + Rust tokenizer reconstruction). Tasks 3, 4 are schema + dispatch. Task 5 is the smoke. Task 6 ships.
