"""Dump intermediate tensors from the reference model for Go tests.

Captures: tokenization, embedding, RoPE cos/sin, per-layer attention outputs,
per-layer MLP outputs, final hidden states and logits, and routing info.

We run the short Sherlock Holmes text only to keep fixtures small.
"""
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
OUT_DIR = Path(__file__).resolve().parent.parent / "fixtures"
OUT_DIR.mkdir(exist_ok=True)

TEXT = "My name is Sherlock Holmes and my email is sherlock.holmes@scotlandyard.uk."


def _save_np(name: str, arr: np.ndarray) -> dict:
    arr = np.ascontiguousarray(arr)
    path = OUT_DIR / f"{name}.f32.bin"
    arr.astype(np.float32).tofile(path)
    return {"name": name, "shape": list(arr.shape), "path": path.name}


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR, dtype=torch.float32)
    model.eval()

    enc = tokenizer(TEXT, return_tensors="pt", return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].tolist()
    input_ids = enc["input_ids"][0].tolist()
    print(f"num tokens: {len(input_ids)}")

    # Attach hooks to capture intermediate tensors from the encoder.
    inner = model.model  # OpenAIPrivacyFilterModel
    captures: dict[str, torch.Tensor] = {}

    def save_hook(name):
        def _hook(module, inp, out):
            tensor = out[0] if isinstance(out, tuple) else out
            captures[name] = tensor.detach().float().cpu().numpy()
        return _hook

    # Embedding
    handles = []
    handles.append(inner.embed_tokens.register_forward_hook(save_hook("embedding")))

    # Per-layer attention and MLP outputs
    for i, layer in enumerate(inner.layers):
        handles.append(layer.input_layernorm.register_forward_hook(save_hook(f"layer{i}_pre_attn_norm")))
        handles.append(layer.self_attn.register_forward_hook(save_hook(f"layer{i}_attn_out")))
        handles.append(layer.post_attention_layernorm.register_forward_hook(save_hook(f"layer{i}_pre_mlp_norm")))
        handles.append(layer.mlp.register_forward_hook(save_hook(f"layer{i}_mlp_out")))
        handles.append(layer.register_forward_hook(save_hook(f"layer{i}_out")))

    handles.append(inner.norm.register_forward_hook(save_hook("final_norm")))

    # Capture RoPE cos/sin
    def rope_hook(module, inp, out):
        cos, sin = out
        captures["rope_cos"] = cos.detach().float().cpu().numpy()
        captures["rope_sin"] = sin.detach().float().cpu().numpy()

    handles.append(inner.rotary_emb.register_forward_hook(rope_hook))

    # Also capture router decisions for layer 0 only.
    layer0_router = inner.layers[0].mlp.router

    def router_hook(module, inp, out):
        logits, scores, indices = out
        captures["layer0_router_logits"] = logits.detach().float().cpu().numpy()
        captures["layer0_router_scores"] = scores.detach().float().cpu().numpy()
        captures["layer0_router_indices"] = indices.detach().cpu().numpy()

    handles.append(layer0_router.register_forward_hook(router_hook))

    # Forward pass
    with torch.no_grad():
        out = model(**enc)

    logits = out.logits[0].float().cpu().numpy()  # [seq, 33]
    probs = torch.softmax(out.logits[0], dim=-1).float().cpu().numpy()

    for h in handles:
        h.remove()

    # Dump everything
    manifest = {
        "text": TEXT,
        "input_ids": input_ids,
        "offsets": offsets,
        "seq_len": len(input_ids),
        "hidden_size": 640,
        "head_dim": 64,
        "num_q_heads": 14,
        "num_kv_heads": 2,
        "num_experts": 128,
        "num_experts_per_tok": 4,
        "intermediate_size": 640,
        "rms_norm_eps": 1e-5,
        "sliding_window": 129,
        "tensors": [],
    }

    # Compressed format for float arrays.
    for name, arr in captures.items():
        info = _save_np(name, arr)
        manifest["tensors"].append(info)
        print(f"  dumped {name}: shape={arr.shape}")

    # Final logits
    manifest["tensors"].append(_save_np("logits", logits))
    manifest["tensors"].append(_save_np("probs", probs))

    # Predicted labels
    pred_ids = logits.argmax(axis=-1).tolist()
    manifest["pred_label_ids"] = pred_ids
    manifest["pred_labels"] = [model.config.id2label[i] for i in pred_ids]
    manifest["pred_scores"] = [float(probs[i, pid]) for i, pid in enumerate(pred_ids)]

    # Id -> label map
    manifest["id2label"] = {str(k): v for k, v in model.config.id2label.items()}

    with open(OUT_DIR / "fixtures.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    # Also dump the attention mask we expect (sliding window of size 129 = 128+1, bidirectional).
    T = len(input_ids)
    mask = np.full((T, T), -np.inf, dtype=np.float32)
    W = 128  # half window
    for i in range(T):
        for j in range(T):
            if abs(i - j) <= W:
                mask[i, j] = 0.0
    _save_np("attention_mask_bias", mask)
    manifest["tensors"].append({"name": "attention_mask_bias", "shape": [T, T], "path": "attention_mask_bias.f32.bin"})

    # Dump raw score.bias weight (bf16 -> f32) for cross-checking the
    # SafeTensors loader's BF16 decoding. We read the tensor directly off the
    # model so the comparison is against the canonical bf16->f32 expansion.
    score_bias = model.score.bias.detach().to(torch.bfloat16).to(torch.float32).cpu().numpy()
    manifest["tensors"].append(_save_np("score_bias", score_bias))

    with open(OUT_DIR / "fixtures.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"wrote {OUT_DIR / 'fixtures.json'}")


if __name__ == "__main__":
    main()
