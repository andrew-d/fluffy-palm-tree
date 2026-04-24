# BLIS packed-B GEMM

**Expected gain: 1.15–1.25× on the MoE matmul (MoE-specific).**

The pragmatic variant — "transpose B once at load time, keep using
`axpyBatch16`" — probably captures 80% of the win at 20% of the
engineering cost.

## What it is

BLIS decomposes GEMM into **five nested loops around a microkernel**
(Goto/Van Zee 2015). From outer to inner:

- **Loop 5** slices N into NC-wide panels.
- **Loop 4** slices K into KC-tall panels and **packs B into a
  contiguous `[KC × NC]` buffer** that lives in L3.
- **Loop 3** slices M into MC-tall blocks and **packs A into
  `[MC × KC]`** living in L2.
- **Loops 2 and 1** iterate NR-wide and MR-tall strips.
- The **microkernel** updates an `MR × NR` register tile of C by
  outer-product accumulation along KC.

The key insight — and the thing our earlier "raw BLIS-style tile"
attempts (iter 16, 42, 43) missed — is that `packB` **does not just
copy B**. It **transposes-interleaves** B so that the microkernel's
inner K-loop reads `NR` B-values as a single contiguous vector, then
advances K by exactly `NR * sizeof(f32)` bytes. That kills the strided
`ldb * (jj + j) + l` gather pattern that blew up our register file
last time.

MC/KC/NC are sized so `MC*KC*4 ≤ L2`, `KC*NR*4 ≤ L1`, and
`KC*NC*4 ≤ L3` (typically 72/256/4080 on Zen, 144/336/6144 on SKX).

Interestingly, llama.cpp's tinyBLAS
(`ggml/src/ggml-cpu/llamafile/sgemm.cpp:549-580`) **does not pack** —
it assumes `C = A^T · B` so both operands are already row-major over
K, and the kernel at line 563
(`Cv[j][i] = madd(Av[i], Bv, Cv[j][i])`) terminates in `hsum` at
line 579. It's a dot-tile, not a BLIS outer-product kernel. The
template dispatcher at lines 501–527 picks `RM × RN` in
`{4×6, 4×3, 2×2}` with a BM (unroll) factor — no packA, no packB.

## Why it might help us

For N=10, K=640, output=1280: a single expert's B is
`640 · 1280 · 4 = 3.125 MB` — already too big for typical per-core L2
(1–1.25 MB on Zen4/SPR), so every call cold-reads B from L3 or DRAM.
**Pre-packing B once at load time** into NR-wide strips (e.g. NR=16,
layout `[1280/16][640][16]`) would:

1. Eliminate the 1280-stride gather.
2. Make prefetching linear.
3. Let the microkernel use a single contiguous vector load per
   K-step.

With 128 experts × 32 layers × 3.125 MB = ~12.8 GB extra RAM if we
pack all gate/up/down — **not feasible**. Packing only gate+up (~4 GB)
or only the hot experts (top-k ~8 of 128 active per token) is
realistic.

However, `axpyBatch16` is hard to beat at N=10. The M dimension cannot
fill an MR=4 or MR=8 register tile (only 2.5 or 1.25 tiles' worth), so
MR×NR tiling amortizes wRow loads only ~10× vs `axpyBatch16`'s 1600×.

**Realistic speedup: 1.15–1.25× on the MoE matmul** — the win comes
almost entirely from the contiguous B stream (memory bandwidth), not
from register reuse. On output=1280 with K=640, if we're currently
DRAM-bound at ~30 GB/s effective, packed B could push to ~60 GB/s (L3
streaming), which translates to roughly +10–15% end-to-end given MoE
is 45% of forward.

## What makes it hard

Three concrete blockers for pure-Go + `simd/archsimd`:

**No register allocator contract.** BLIS microkernels work because the
author knows all 32 ZMM registers are live for the C-tile and nothing
else spills. Go's SSA backend has no way to pin
`[MR][NR/16]Float32x16` accumulators to registers across a
640-iteration K-loop — iter 47's regression
(`for s := range [16]Float32x16{}` spilling to stack) is exactly this.
Even `[4][2]Float32x16` (the minimum viable tile) is 8 ZMMs, and Go's
allocator will happily shuffle them through stack slots on any
function call boundary, killing the inner loop.

**No aligned struct layout for packed buffers.** We would want
`[KC][NR]float32` laid out so `_mm512_load_ps` can use aligned moves;
Go's allocator gives 8-byte alignment on `make([]float32, ...)` with
no way to request 64-byte alignment short of over-allocating and
slicing.

**Packing cost amortization.** Load-time packing is free, but if we
ever need re-pack (e.g. LoRA merge, dynamic expert eviction), the cost
is `K*N*2` memory traffic which at N=10 would take longer than the
matmul itself.

`archsimd`'s `Float32x16` has no FMA3 231-form guarantee — the
compiler emits `VFMADD231PS` today but there's no intrinsic like
`_mm512_fmadd_ps` that locks in the accumulator register.

Given the 1.15–1.25× estimate and these blockers, **`axpyBatch16` with
a B-layout-swap at load time** (transpose once so the K-stride is
contiguous) probably captures 80% of the win at 20% of the engineering
cost. That's the pragmatic intermediate.

## Relevant files

- llama.cpp `ggml/src/ggml-cpu/llamafile/sgemm.cpp:549-580` — non-packing dot-tile kernel
- llama.cpp `ggml/src/ggml-cpu/llamafile/sgemm.cpp:501-527` — mnpack RM×RN dispatch
- llama.cpp `ggml/src/ggml-cpu/llamafile/sgemm.cpp:274-278` — AVX-512 hsum

## Sources

- [BLIS: A Framework for Rapidly Instantiating BLAS Functionality (Van Zee & van de Geijn, 2015)](https://www.cs.utexas.edu/~flame/pubs/BLISTOMSrev2.pdf)
- [The BLIS Framework: Experiments in Portability (TOMS 2016)](https://www.cs.utexas.edu/~flame/pubs/BLIS_TOMS2.pdf)
- [BLISlab: A Sandbox for Optimizing GEMM (FLAME WN #80)](https://arxiv.org/pdf/1609.00076)
- [GEMMFIP: Unifying GEMM in BLIS (Xu, 2023)](https://arxiv.org/pdf/2302.08417)
- [flame/how-to-optimize-gemm wiki](https://github.com/flame/how-to-optimize-gemm/wiki)
