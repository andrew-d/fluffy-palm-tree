# OpenAI Privacy Filter — architecture notes

These are the details future sub-agents need to implement the model in pure Go.

## High level

- Encoder-only token classifier, bidirectional sliding-window attention.
- 8 transformer blocks, `d_model=640`.
- MoE FFN: 128 experts, top-4 routing, intermediate_size=640 per expert (so the GLU hidden dim per expert is 2*640=1280 because gate/up are concatenated).
- Grouped-query attention: 14 query heads, 2 KV heads (group size 7), `head_dim=64`.
- Attention sinks: per-query-head learnable fp32 scalar appended to logits before softmax.
- RoPE: yarn variant, `theta=150000`, `original_max_position=4096`, scaling `factor=32`, `beta_fast=32`, `beta_slow=1`. Head-dim=64 → 32 pairs.
- Sliding window size: `sliding_window + 1 = 129` (cfg value 128, code adds +1 for FA symmetry). Bidirectional (left=128, right=128).
- Tokenizer: `o200k_base` (GPT-4 BPE), vocab 200064.
- Labels: 33 (1 O + 8 entity types × BIOES).
- Weights: bfloat16 except attention `sinks` which are float32.

## Tensor shapes (from `model.safetensors`)

```
model.embed_tokens.weight                         (200064, 640)  bf16
model.layers.{0..7}.input_layernorm.weight              (640,)  bf16
model.layers.{0..7}.self_attn.q_proj.weight         (896, 640)  bf16   # out = num_q_heads*head_dim = 14*64
model.layers.{0..7}.self_attn.q_proj.bias             (896,)    bf16
model.layers.{0..7}.self_attn.k_proj.weight         (128, 640)  bf16   # out = num_kv_heads*head_dim = 2*64
model.layers.{0..7}.self_attn.k_proj.bias             (128,)    bf16
model.layers.{0..7}.self_attn.v_proj.weight         (128, 640)  bf16
model.layers.{0..7}.self_attn.v_proj.bias             (128,)    bf16
model.layers.{0..7}.self_attn.o_proj.weight         (640, 896)  bf16
model.layers.{0..7}.self_attn.o_proj.bias             (640,)    bf16
model.layers.{0..7}.self_attn.sinks                    (14,)    fp32
model.layers.{0..7}.post_attention_layernorm.weight     (640,)  bf16
model.layers.{0..7}.mlp.router.weight               (128, 640)  bf16   # [num_experts, hidden]
model.layers.{0..7}.mlp.router.bias                    (128,)   bf16
model.layers.{0..7}.mlp.experts.gate_up_proj   (128, 640, 1280) bf16   # [E, hidden, 2*intermediate]
model.layers.{0..7}.mlp.experts.gate_up_proj_bias   (128, 1280) bf16
model.layers.{0..7}.mlp.experts.down_proj       (128, 640, 640) bf16   # [E, intermediate, hidden]
model.layers.{0..7}.mlp.experts.down_proj_bias       (128, 640) bf16
model.norm.weight                                       (640,)  bf16
score.weight                                          (33, 640) bf16
score.bias                                              (33,)   bf16
```

Total 140 tensors. File: `model/model.safetensors` (~2.8GB bf16).

## Forward pass

```
x = embed_tokens(input_ids)       # [B, T, 640]
cos, sin = rotary_emb(positions)  # [B, T, head_dim/2] each (yarn-scaled)
mask = bidirectional_sliding_window_mask(T, window=129)

for layer in 0..7:
    residual = x
    h = rms_norm(x, weight=input_layernorm.weight, eps=1e-5)
    q = linear(h, q_proj.weight, q_proj.bias)              # [B, T, 896]
    k = linear(h, k_proj.weight, k_proj.bias)              # [B, T, 128]
    v = linear(h, v_proj.weight, v_proj.bias)              # [B, T, 128]
    # reshape to heads
    q = reshape(q, [B, T, 14, 64]).transpose(1,2)
    k = reshape(k, [B, T, 2, 64]).transpose(1,2)
    v = reshape(v, [B, T, 2, 64]).transpose(1,2)
    q, k = apply_rotary(q, k, cos, sin)      # gpt_oss concatenated layout
    q = q * head_dim**-0.25
    k = k * head_dim**-0.25                  # so q*k^T is already scaled by head_dim^-0.5
    k = repeat_kv(k, group=7)                # [B, T, 14, 64]
    v = repeat_kv(v, group=7)
    scores = q @ k^T                         # [B, 14, T, T]
    scores += mask                           # bidir sliding window: 0 in-window, -inf out
    sinks = broadcast(self.sinks, [B, 14, T, 1])   # fp32
    combined = concat([scores, sinks], dim=-1)      # [B, 14, T, T+1]
    combined = combined - combined.max(dim=-1, keep=True)
    probs = softmax(combined, dim=-1)   # fp32
    attn_weights = probs[..., :-1]      # drop sink column
    out = attn_weights @ v              # [B, 14, T, 64]
    out = transpose(1,2) -> [B, T, 14, 64] -> [B, T, 896]
    out = linear(out, o_proj.weight, o_proj.bias)   # [B, T, 640]
    x = residual + out

    residual = x
    h = rms_norm(x, weight=post_attention_layernorm.weight, eps=1e-5)
    # MoE
    h2 = h.reshape(-1, 640)
    router_logits = linear(h2.fp32, router.weight.fp32, router.bias.fp32)   # [T, 128]
    top_vals, top_idx = topk(router_logits, 4)
    router_scores = softmax(top_vals, dim=-1) / 4             # [T, 4], fp32
    out2 = zeros_like(h2, fp32)
    for each unique expert e used by any token:
        tokens = {(token_idx, top_pos): router_idx[token_idx, top_pos] == e}
        for (t, pos) in tokens:
            state = h2[t].fp32
            gate_up = state @ gate_up_proj[e].fp32 + gate_up_proj_bias[e].fp32        # [1280]
            gate, up = gate_up.chunk(2, -1)                                           # concatenated layout; each [640]
            gate = min(gate, 7.0)
            up  = clamp(up, -7.0, 7.0)
            glu = gate * sigmoid(gate * 1.702)
            gated = (up + 1.0) * glu
            y = gated @ down_proj[e].fp32 + down_proj_bias[e].fp32                    # [640]
            out2[t] += y * router_scores[t, pos]
    out2 = out2 * num_experts_per_tok  (4)  # additional scaling
    out2 = out2.reshape(B, T, 640)
    x = residual + out2.to(bf16)

x = rms_norm(x, weight=model.norm.weight, eps=1e-5)
logits = linear(x, score.weight, score.bias)    # [B, T, 33]
```

## RMSNorm formula

```
rsqrt(mean(x^2) + eps) * x * weight      # in fp32, then cast back
```

## RoPE (yarn-scaled)

Inverse frequencies are computed with yarn scaling. See `transformers.modeling_rope_utils.compute_yarn_parameters`. For the given config:
- theta=150000, head_dim=64, dim/2 = 32 pairs
- original_max_positions=4096, factor=32, beta_fast=32, beta_slow=1
- yarn blends low-frequency "scaled" and high-frequency "unscaled" components.

The cos/sin tables are `[T, head_dim/2]`. Applied via "concatenated" layout:
```
first_half, second_half = chunk(x, 2, dim=-1)   # each [..., head_dim/2]
cos/sin expanded to [..., head_dim/2]
first_out  = first_half * cos - second_half * sin
second_out = second_half * cos + first_half * sin
x_out      = concat([first_out, second_out], dim=-1)
```
(NOT interleaved — the modular file's `_apply_rotary_emb` is unused; the attention block calls `apply_rotary_pos_emb` from `gpt_oss`.)

## Bidirectional sliding window mask

For token i and token j, attention is allowed iff `|i-j| <= sliding_window` where `sliding_window = cfg.sliding_window + 1 = 129`. The mask is `0` where allowed, `-inf` where not (added as bias to scores).

## Tokenizer

`tokenizer.json` is HuggingFace tokenizers library BPE with `o200k_base` encoding (same as GPT-4o). The tokenizer applies the pre-tokenizer, BPE merges, and returns `input_ids`. No special tokens are added by default for this model. For our test inputs, the tokenizer runs `pre_tokenizer` with a splits-on-whitespace-etc regex then BPE-merges.

## Aggregation ("simple" BIOES-aware)

The transformers.js model card example shows:
- Sherlock Holmes example → entities `private_person: " Sherlock Holmes"` and `private_email: " sherlock.holmes@scotlandyard.uk"`.
- Per-token labels we observed:
  - ` Sherlock` → B-private_person, ` Holmes` → E-private_person
  - ` sher` → B-private_email, `lock`…`yard` → I-private_email, `.uk` → E-private_email

The transformers' `simple` strategy only merges B and I of the same entity type. For this model with BIOES, we need to merge B+I*+E into one entity (and S alone as a separate entity). Aggregation steps:

1. For each token pick the argmax label and its softmax score.
2. Walk tokens left-to-right. When we see `B-X` or `I-X` without an open span, open a span of type X. When we see `I-X`/`E-X` with an open X span, extend it. When we see `S-X`, emit a one-token span. Close on `O`, `E-X`, or on a mismatched label.
3. Entity score = mean of constituent token softmax scores.
4. Word = concat of constituent token text (using offsets on original string).

## Public Go API surface

```go
package privacyfilter

type Model struct { /* weights + config */ }

func LoadModel(dir string) (*Model, error)

type Entity struct {
    EntityGroup string  // "private_person", "private_email", ...
    Score       float32 // mean of token softmax scores
    Word        string  // substring of input text
    Start, End  int     // byte offsets into input text
}

func (m *Model) Classify(text string) ([]Entity, error)
```

Internal packages:
- `safetensors` — pure-Go reader (mmap-friendly)
- `bfloat16` — bf16/fp32 conversion
- `tokenizer` — o200k_base BPE, using `tokenizer.json`
- `tensor` — minimal math: matmul, add, rope, softmax, rms_norm
- (model internals live in the main package)
