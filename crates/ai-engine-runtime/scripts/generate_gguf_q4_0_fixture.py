#!/usr/bin/env python3
"""
Generate toy-llama-3-gguf from toy-llama-3 (bf16). Single-file GGUF v3 with
Q4_0 quantized Linear weights and F32 boundary tensors (embedding, layernorms).

Hand-written GGUF v3 writer (avoids dependency on llama.cpp tooling).

Run once; commit outputs.
"""

import json
import shutil
import struct
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

SRC = Path(__file__).resolve().parent.parent / "fixtures" / "toy-llama-3"
OUT = Path(__file__).resolve().parent.parent / "fixtures" / "toy-llama-3-gguf"
OUT.mkdir(parents=True, exist_ok=True)

# --- GGUF constants ---
GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
TYPE_U8, TYPE_I8, TYPE_U16, TYPE_I16, TYPE_U32, TYPE_I32 = range(6)
TYPE_F32, TYPE_BOOL, TYPE_STRING, TYPE_ARRAY = 6, 7, 8, 9
TYPE_U64, TYPE_I64, TYPE_F64 = 10, 11, 12
GGML_F32, GGML_F16, GGML_Q4_0, GGML_BF16 = 0, 1, 2, 30

Q4_0_BLOCK = 32
Q4_0_BYTES_PER_BLOCK = 18

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

HF_TO_GGUF = {
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
}
def hf_to_gguf_name(name: str) -> str:
    if name in HF_TO_GGUF:
        return HF_TO_GGUF[name]
    if name.startswith("model.layers."):
        parts = name.split(".")
        # model.layers.N.<rest>
        i = parts[2]
        rest = ".".join(parts[3:])
        return {
            "input_layernorm.weight": f"blk.{i}.attn_norm.weight",
            "self_attn.q_proj.weight": f"blk.{i}.attn_q.weight",
            "self_attn.k_proj.weight": f"blk.{i}.attn_k.weight",
            "self_attn.v_proj.weight": f"blk.{i}.attn_v.weight",
            "self_attn.o_proj.weight": f"blk.{i}.attn_output.weight",
            "post_attention_layernorm.weight": f"blk.{i}.ffn_norm.weight",
            "mlp.gate_proj.weight": f"blk.{i}.ffn_gate.weight",
            "mlp.up_proj.weight": f"blk.{i}.ffn_up.weight",
            "mlp.down_proj.weight": f"blk.{i}.ffn_down.weight",
        }[rest]
    raise ValueError(f"unhandled tensor name: {name}")

def quantize_q4_0(w: np.ndarray) -> bytes:
    """Quantize a 1D f32 array (length multiple of 32) into Q4_0 blocks.
    Block layout: f16 scale + 16 bytes of nibbles, with low nibbles = indices 0..16, high = 16..32.
    """
    assert w.dtype == np.float32 and w.size % Q4_0_BLOCK == 0
    n_blocks = w.size // Q4_0_BLOCK
    out = bytearray()
    for b in range(n_blocks):
        block = w[b * Q4_0_BLOCK:(b + 1) * Q4_0_BLOCK]
        max_abs = float(np.max(np.abs(block)))
        if max_abs == 0.0:
            scale = 1.0
            quantized = np.zeros(Q4_0_BLOCK, dtype=np.int32)
        else:
            scale = max_abs / 8.0  # GGUF uses signed range -8..7, so divide by 8
            quantized = np.clip(np.round(block / scale).astype(np.int32), -8, 7)
        # f16 scale
        out.extend(np.float16(scale).tobytes())
        # 16 bytes of nibbles
        # qs[j] (j in 0..16): low nibble = block[j], high nibble = block[j + 16]
        for j in range(16):
            lo = (quantized[j] + 8) & 0x0F
            hi = (quantized[j + 16] + 8) & 0x0F
            out.append((hi << 4) | lo)
    return bytes(out)

def write_gguf_string(buf: bytearray, s: str):
    data = s.encode("utf-8")
    buf.extend(struct.pack("<Q", len(data)))
    buf.extend(data)

def write_kv_u32(buf, key, value):
    write_gguf_string(buf, key)
    buf.extend(struct.pack("<I", TYPE_U32))
    buf.extend(struct.pack("<I", value))

def write_kv_u64(buf, key, value):
    write_gguf_string(buf, key)
    buf.extend(struct.pack("<I", TYPE_U64))
    buf.extend(struct.pack("<Q", value))

def write_kv_f32(buf, key, value):
    write_gguf_string(buf, key)
    buf.extend(struct.pack("<I", TYPE_F32))
    buf.extend(struct.pack("<f", value))

def write_kv_string(buf, key, value):
    write_gguf_string(buf, key)
    buf.extend(struct.pack("<I", TYPE_STRING))
    write_gguf_string(buf, value)

# --- Build GGUF file ---

# Load bf16 source.
src = load_file(SRC / "model.safetensors")
cfg = json.loads((SRC / "config.json").read_text())

# Metadata
meta = bytearray()
meta_count = 0

write_kv_string(meta, "general.architecture", "llama"); meta_count += 1
write_kv_string(meta, "general.name", "toy-llama-3-q4-0"); meta_count += 1
write_kv_u32(meta, "general.alignment", 32); meta_count += 1
write_kv_u32(meta, "llama.block_count", cfg["num_hidden_layers"]); meta_count += 1
write_kv_u32(meta, "llama.embedding_length", cfg["hidden_size"]); meta_count += 1
write_kv_u32(meta, "llama.attention.head_count", cfg["num_attention_heads"]); meta_count += 1
write_kv_u32(meta, "llama.attention.head_count_kv", cfg["num_key_value_heads"]); meta_count += 1
write_kv_u32(meta, "llama.feed_forward_length", cfg["intermediate_size"]); meta_count += 1
write_kv_u32(meta, "llama.context_length", cfg["max_position_embeddings"]); meta_count += 1
write_kv_f32(meta, "llama.attention.layer_norm_rms_epsilon", cfg["rms_norm_eps"]); meta_count += 1
rope_theta = cfg.get("rope_theta", cfg.get("rope_parameters", {}).get("rope_theta", 10000.0))
write_kv_f32(meta, "llama.rope.freq_base", float(rope_theta)); meta_count += 1

# Build tensors
tensor_descs = bytearray()
tensor_data = bytearray()
tensor_count = 0

def write_tensor_desc(name, n_dims, shape, ggml_type, offset):
    write_gguf_string(tensor_descs, name)
    tensor_descs.extend(struct.pack("<I", n_dims))
    for d in shape:
        tensor_descs.extend(struct.pack("<Q", d))
    tensor_descs.extend(struct.pack("<I", ggml_type))
    tensor_descs.extend(struct.pack("<Q", offset))

def should_quantize(name: str) -> bool:
    return any(p in name for p in LINEAR_PATTERNS)

ALIGN = 32

def append_data_with_alignment(data: bytes) -> int:
    # Align to 32 bytes within tensor_data buffer.
    while len(tensor_data) % ALIGN != 0:
        tensor_data.append(0)
    off = len(tensor_data)
    tensor_data.extend(data)
    return off

for hf_name, t in src.items():
    gguf_name = hf_to_gguf_name(hf_name)
    arr = t.to(torch.float32).numpy()
    if should_quantize(hf_name):
        # GGUF stores tensors flat-column-major over [in, out].
        # arr has HF shape [out, in]; arr.flatten(C-order) yields
        # (out0_in0, out0_in1, ..., out0_in_last, out1_in0, ...)
        # which is column-major over [in, out] under relabeling.
        # So we flatten without transposing.
        flat = arr.flatten().astype(np.float32)
        q = quantize_q4_0(flat)
        offset = append_data_with_alignment(q)
        # GGUF tensor shape is [in, out] (math order).
        write_tensor_desc(gguf_name, 2, [arr.shape[1], arr.shape[0]], GGML_Q4_0, offset)
    else:
        # F32 for non-linear tensors.
        if arr.ndim == 1:
            data = arr.astype(np.float32).tobytes()
            offset = append_data_with_alignment(data)
            write_tensor_desc(gguf_name, 1, [arr.shape[0]], GGML_F32, offset)
        elif arr.ndim == 2:
            # Same flat-layout reasoning as above.
            data = arr.astype(np.float32).flatten().tobytes()
            offset = append_data_with_alignment(data)
            write_tensor_desc(gguf_name, 2, [arr.shape[1], arr.shape[0]], GGML_F32, offset)
        else:
            raise ValueError(f"unhandled tensor rank: {arr.shape}")
    tensor_count += 1

# Assemble final file.
header = bytearray()
header.extend(GGUF_MAGIC)
header.extend(struct.pack("<I", GGUF_VERSION))
header.extend(struct.pack("<Q", tensor_count))
header.extend(struct.pack("<Q", meta_count))

# Pad after tensor descriptors to ALIGN before tensor data.
total_before_data = len(header) + len(meta) + len(tensor_descs)
pad = (ALIGN - (total_before_data % ALIGN)) % ALIGN

with open(OUT / "model.gguf", "wb") as f:
    f.write(header)
    f.write(meta)
    f.write(tensor_descs)
    f.write(b"\x00" * pad)
    f.write(tensor_data)

# Copy reference logits + tokenizer + reference prompt + config.
shutil.copy(SRC / "reference_logits.bin", OUT / "reference_logits.bin")
shutil.copy(SRC / "tokenizer.json", OUT / "tokenizer.json")
shutil.copy(SRC / "reference_prompt.txt", OUT / "reference_prompt.txt")
shutil.copy(SRC / "config.json", OUT / "config.json")

(OUT / "README.md").write_text(
"""# toy-llama-3-gguf fixture

Generated by `scripts/generate_gguf_q4_0_fixture.py`. Do not edit by hand.

Single-file GGUF v3 with Q4_0 quantized Linear weights and F32 boundary tensors.

| File | Purpose |
|---|---|
| model.gguf | GGUF v3 file with Q4_0 Linear weights |
| config.json | Same as toy-llama-3 (for ModelConfig parsing) |
| tokenizer.json | Same as toy-llama-3 |
| reference_prompt.txt | Same as toy-llama-3 |
| reference_logits.bin | Same as toy-llama-3 - GGUF Q4_0 forward must match within ~5e-1 (Q4 noise on random toy is large) |

GGUF Q4_0 layout: per-32 block, f16 scale + 16 bytes of biased nibbles (low half = indices 0..16, high half = 16..32).
""")

# Stats
bf16_size = sum(t.numel() * t.element_size() for t in src.values())
gguf_size = (OUT / "model.gguf").stat().st_size
print(f"bf16 fixture:  {bf16_size} bytes")
print(f"GGUF fixture:  {gguf_size} bytes")
print(f"compression:   {bf16_size / gguf_size:.2f}x")
