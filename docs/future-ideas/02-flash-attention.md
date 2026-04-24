# Flash-Attention on CPU

**Expected gain: 1.15–1.35× end-to-end (mostly from pass fusion, not cache).**

## What it is

Flash-Attention (Dao et al.) replaces the classic three-pass softmax —
score, max/normalize, weight V — with a single fused streaming pass
over K/V tiles. For each query (or Q-tile), the kernel maintains three
running scalars per query row: a running max `M`, a running denominator
`S`, and a running output accumulator `O` (shape `headDim`). For each
KV tile it:

1. Computes the tile's QK block and scales.
2. Finds that tile's max.
3. If the new max `Mnew` exceeds the old `Mold`, rescales both `S` and
   `O` by `exp(Mold − Mnew)`.
4. Adds the tile's softmax partials to `S` and its softmax-weighted V
   contribution to `O`.

At the end, `O` is divided by `S`. Because the full [T, T] score matrix
is never materialized, memory traffic collapses from `O(T²·bytes)` to
`O(T·headDim·bytes)` per head, and the softmax and V-gather fuse into
one sweep over K/V.

llama.cpp implements this at `ggml/src/ggml-cpu/ops.cpp:8406–8680` with
`Q_TILE=KV_TILE=64` (`common.h:9–10`), packing Q/K/V into per-thread
scratch (`Q_q`, `K_f32`, `V32`, `VKQ32`) and performing tile matmuls
via `simd_gemm`.

## Why it might help us

The cache-pressure argument is near-zero on our shapes: a `T=306` score
row is ~1.2 KB per head, and the full `[T, T]` block is ~374 KB — well
within L2 on any modern core. The sliding window shrinks the live band
further to `≤2·128·4 = 1 KB` per row.

The real win is **pass fusion**. Today's kernel makes two sweeps over K
(dot into `scores[j]`) and V (axpy-per-j) with an `exp/sum/scale` pass
in between — three loads of the KV row per head per query, in three
separate vector loops. A flash kernel fuses QK, softmax, and
V-accumulation into one sweep, and — critically — **reshapes the
workload from "306 independent single-row attentions per head" into a
Q-tile GEMM**: one `Q_TILE × headDim × KV_TILE` matmul per tile
instead of dot-per-i-per-j.

On our shapes, `Q_TILE=64` against `KV_TILE=64` with `headDim=64` maps
to dense 64×64 GEMMs, which our existing `linearTile4x4`-style kernel
already services efficiently. Secondary gain: 14 Q heads share 2 KV
heads (group=7), so packed K/V tiles amortize across 7 Q rows.

## What makes it hard

The refactor is invasive. Our current five-step pipeline becomes:
outer loop over Q-tiles, inner loop over KV-tiles, with
`QK-GEMM → (optional mask-skip) → per-row online max/rescale →
softmax-exp → VKQ-GEMM` (`ops.cpp:8598–8620`).

Packed K/V transposed scratches (`K_f32[dk*KV+tk]`,
`V32[tk*DV+dv]`, `ops.cpp:8578–8596, 8637–8644`) are required for the
matmuls to be SIMD-contiguous — this is new allocation and
transposition logic we don't have today.

The sliding window maps cleanly onto llama.cpp's `can_skip` pattern
(`ops.cpp:8555–8577`): precompute a mask tile per `(Q-tile, KV-tile)`
and skip whole tiles when `|i−j| > 127` for every pair. But
diagonal-straddling tiles require per-cell `-inf` masking, adding a
small fixup pass.

Attention sinks compose cleanly and are applied **once at the end**,
not per-tile: llama.cpp treats the sink as a single extra
"pseudo-tile" that may bump `M` (`ops.cpp:8646–8662`), scaling both `S`
and `O` accordingly before the final `O /= S` — exactly our "included
in denominator, never multiplied into output" semantics.

The online rescale costs one `expf` and one vector scale of `O` per
tile per row where `Mnew > Mold`, which is strictly more work than our
single-pass post-hoc softmax, but is amortized by the fused GEMMs. The
rescale invariant is the main correctness risk — easy to get subtly
wrong.

Net: the algorithm is right, composes with our sinks/window, but the
refactor touches packed scratch management, two new GEMM call sites,
tile-boundary masking, and a nontrivial correctness surface. The
claimed 1.15–1.35× is plausible as a **pass-fusion / GEMM-density**
win, not a cache win.

## Relevant files

- `internal/nn/attention.go` — current kernel
- llama.cpp `ggml/src/ggml-cpu/ops.cpp:8406–8680` — tiled flash-attn
- llama.cpp `ggml/src/ggml-cpu/common.h:9–10` — `GGML_FA_TILE_Q=64`, `GGML_FA_TILE_KV=64`
- llama.cpp `ops.cpp:8555–8577` — mask-skip
- llama.cpp `ops.cpp:8598–8620` — online max/rescale
- llama.cpp `ops.cpp:8646–8662` — sinks
