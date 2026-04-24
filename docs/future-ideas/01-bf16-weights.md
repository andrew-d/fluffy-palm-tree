# bf16 weight storage with on-the-fly fp32 upconvert

**Expected gain: 1.3–1.5× on MoE-heavy layers (highest ROI of the untried levers).**

## What it is

bf16 is a 16-bit float that keeps fp32's 8-bit exponent and truncates the
mantissa to 7 bits. fp16 splits the budget as 5 exponent / 10 mantissa,
giving it more precision but a ~65k dynamic range that silently overflows
on LLM activations. bf16 keeps fp32's full ~1e-38 to ~3e38 range, so an
fp32→bf16 cast is a pure 16-bit right-truncation of the mantissa — no
rescaling, no NaN retraps. That is why every modern training stack (our
weights included) ships bf16 rather than fp16.

On Cascade Lake (our CPU: Xeon Platinum 8259CL), which lacks
`AVX512_BF16` (Cooper Lake+), the upconvert bf16→fp32 is literally
"zero-extend the 16 bits into the high half of a 32-bit lane." The
sequence

```
LoadInt16x16 → ExtendToInt32 (zero-extend) → ShiftAllLeft(16) → AsFloat32x16
```

is correct and canonical — no fix-up for denormals or NaN is needed
because the bit-patterns of bf16 map 1:1 to the upper 16 bits of fp32.
Versus a straight `LoadFloat32x16Slice`, we trade one 512-bit load for
one 256-bit load plus three ALU ops (zext, vpslld, reinterpret). On
SKX/CLX these are all 1-cycle throughput on port 0/5, so the added
latency is ~3–4 cycles per 16 weights (~0.2 cycles/element), easily
hidden behind the FMA chain in `axpyBatch16`.

## Why it might help us

MoE weights are the dominant term: 5.6 GB/layer × 8 layers streamed
once per forward. The profile has `LoadFloat32x16Slice` at 13%
self-time inside an `axpyBatch16` that is 27% self-time — i.e. ~half of
axpy's time is the load itself. That is the DRAM-bound fingerprint.
Halving weight bytes drops that bandwidth requirement by 2×; the extra
~0.2 cycles/element of decode is fully overlapped with an FMA chain
that is already ≥4 cycles. Expected net: **1.3–1.5× on MoE-heavy
layers**, tapering on the non-MoE projections where weight reuse across
tokens already amortizes loads.

This composes with the k-unroll and d-fusion wins — those reduce
**activation** traffic and increase arithmetic intensity, which is
orthogonal to shrinking the weight stream. If anything, tighter k/d
tiling makes the weight load a larger fraction of the remaining time,
so bf16 helps more after those wins, not less.

**Crucially, this is bit-exact lossless for our codebase.** Our weights
are already bf16 on disk (`model.safetensors`); the loader currently
inflates them to fp32 at `safetensors.go:86-98`. The 1e-3 tolerance
bands in `moe_test.go` and `attention_test.go` are already budgeted
against the bf16 source. An existing test at `nn_test.go:193` asserts
the bf16→fp32 round-trip is bit-exact.

## What makes it hard

Three surgical fronts:

**archsimd surface.** We currently use only `Float32x16` intrinsics. A
bf16 path needs `Int16x16` load, `ExtendToInt32` (zero-extend),
`ShiftLeftInt32`, and bitcast. Every `axpyBatch{1,2,4,8,16}` and
`linearTile4x4` variant (see `internal/nn/axpy_simd_amd64.go`) needs a
parallel "bf16-W" version, plus matching scalar fallbacks in
`axpy_fallback.go`.

**Loader surgery.** `safetensors.go:86-98` currently widens bf16→fp32
at `Tensor.Float32s()` and `privatemodel/model.go` holds all weights as
`[]float32` (the doc explicitly notes 2× inflation to 5.6 GB). We would
keep weights as `[]uint16`/`[]int16`, change every `Linear`/`MoE` call
signature (or introduce a typed `BF16Weights` wrapper), and update
router/norm/score paths that currently assume fp32 weight slices.
Activations, biases, sinks, and router scores should stay fp32.

**Accuracy — actually not a concern.** The fp32-held weights are
already the result of a bf16→fp32 expansion, so keeping them as bf16 is
**lossless** (per `nn_test.go:193`). The only failure mode is an
accidental double-rounding if we ever re-quantize fp32 activations —
this proposal does not do that.

## Relevant files

Internal references the research agent highlighted:

- `internal/nn/axpy_simd_amd64.go` — all axpy/linearTile kernels needing bf16-W variants
- `internal/nn/axpy_fallback.go` — scalar fallbacks
- `internal/safetensors/safetensors.go` (lines 30, 86-98) — bf16 decode site
- `internal/privatemodel/model.go` (line 67) — 5.6 GB fp32 footprint note
- `internal/nn/moe.go` — `MoEExperts`/`TopKRouter` consumers
- `internal/nn/moe_test.go` (lines 164, 175) and `internal/nn/attention_test.go` (line 188) — 1e-3 tolerance bands
- `internal/nn/nn_test.go` (line 193) — confirms bf16→fp32 is bit-exact
