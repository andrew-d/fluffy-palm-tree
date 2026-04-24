# Fused RMSNorm + Linear

**Expected gain: 1.03–1.08× (smallest of the untried levers).**

## What it is

RMSNorm does a per-row reduction (`sum-of-squares` over D=640) and then
a per-element scale `x_norm[i] = x[i] * rms * weight[i]`. Linear
computes per-(token, out) dot products `y[t,o] = Σ_k W[o,k] * x_norm[t,k]`.
The structural trick is that the row reduction — computing `rms(t)` —
only depends on row `t`, not on which output column `o` is being
produced. Once `rms(t)` and the gain vector `weight[i] * rms(t)`
(length D) are known, Linear can consume `x_norm[t,k]` implicitly as
`x[t,k] * rms(t) * weight[k]` inside its FMA without ever materializing
the T×D normalized buffer.

Two lowering strategies exist:

1. **Pre-scale.** Compute `rms(t)` first, then modify the `xBase` slice
   (or a small per-tile scratch) by multiplying by `rms * weight` before
   entering the K-loop. Matmul kernel unchanged.
2. **Baked-in.** Push one extra mul per FMA into the hot loop, so it
   becomes `acc += (x[k] * rms * weight[k]) * W[o, k]`.

See llama.cpp PR #16220 for the Metal `NORM + MUL + ADD` fusion pattern
and the CUDA graph-scanner described in discussion #17621.

## Why it might help on our shapes

RMSNorm currently writes a fresh 780 KB buffer
(`T*D*4 = 306*640*4 ≈ 783 KB`) and the next Linear call re-reads it.
At 1 MB L2 per core, that fresh buffer evicts both weights and its own
earlier portion across the 8 workers sharing L3. With the current
parallel-by-output split (`attention.go:34`), each worker restreams the
full 780 KB of h across its weight slab.

Fusing removes the materialization entirely: the 780 KB DRAM write
vanishes, and the re-read becomes a read of the raw `x` (same size,
but it was already hot from the previous layer's `Add`). For the
attention block this is one saved round-trip; for the MLP-side RMSNorm
it's another.

At ~25 GB/s effective DRAM bandwidth, saving ~1.5 MB per layer across
12 layers is ~720 µs per sequence — a plausible 1.03–1.08× envelope
matching the Transformers/CTranslate2 agent hints.

## What makes it hard

**Q/K/V share the same normalized h.** A naive "fuse into Linear"
recomputes `rms(t)` three times unless we hoist the `rms * weight`
vector (length 640, 2.5 KB — trivially L1-resident) and pass it to all
three projections. CTranslate2 doesn't actually fuse the weights at
conversion time (see Qwen3 issue #1902) precisely because RMSNorm is
nonlinear — it fuses the *execution*, not the weights, exactly matching
approach (1).

**`linearTile4x4` integration.** For
`internal/nn/axpy_simd_amd64.go:526`, the cleanest integration is
pre-scaling: compute `rms(t)` once per row outside the output-column
parallel region, store `scaled_h[t,k] = x[t,k] * rms(t) * weight[k]`
into a reusable 780 KB scratch that all three of Q/K/V share — which
recovers materialization and loses the win. The real win requires
either (a) a per-tile 4-row scratch (`4 * 640 * 4 = 10 KB`,
L1-resident) refilled inside each tile, or (b) fusing the mul into the
inner FMA, adding one mul per 16 FMAs (negligible on AVX-512).

**MoE side is awkward.** The MLP-side RMSNorm feeds `MoELayer`, whose
first op is the router `Linear(hidden, wRouter, ...)` at `moe.go:47`
plus the per-expert GateUp. The router absolutely benefits. The GateUp
projections gather tokens by expert assignment **after** routing, so
the fusion there requires either pre-scaling into the gather buffer
(which recovers the materialization and loses the win) or pushing the
scale into the gather — the former is the lazy option and still avoids
the 780 KB intermediate if the scratch is per-expert-tile.

**Pure-Go concern.** `RMSNorm` in `tensor.go` is a reference
implementation and not SIMD-accelerated. The fused version needs to
vectorize the sum-of-squares (`ReduceAddFloat32x16`) to avoid trading
DRAM savings for scalar reduction latency.

## Relevant files

- `internal/nn/tensor.go` — `RMSNorm` (lines 23-48)
- `internal/nn/attention.go` — `Linear` (17-87), `GQAAttentionWithSinks` QKV calls (203-205)
- `internal/nn/axpy_simd_amd64.go` — `linearTile4x4` (514-587+)
- `internal/nn/moe.go` — router Linear (line 47)
- `internal/privatemodel/model.go` — forward-pass sequence (240-270)

## Sources

- [Optimizing Token Generation in llama.cpp's CUDA Backend — Discussion #17621](https://github.com/ggml-org/llama.cpp/discussions/17621)
- [Aman's blog: Optimizing Token Generation in llama.cpp's CUDA Backend](https://am17an.bearblog.dev/new-post/)
- [llama.cpp PR #16220 — metal: fuse NORM + MUL + ADD](https://github.com/ggml-org/llama.cpp/pull/16220)
- [llama.cpp PR #16991 — CUDA: add stream-based concurrency](https://github.com/ggml-org/llama.cpp/pull/16991)
- [CTranslate2 Qwen3 issue #1902 — why RMSNorm γ can't be fused into linear weights](https://github.com/OpenNMT/CTranslate2/issues/1902)
- [NVIDIA Transformer Engine `LayerNormLinear` docs](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html)
