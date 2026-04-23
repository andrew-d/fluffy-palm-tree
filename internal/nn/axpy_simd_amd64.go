//go:build goexperiment.simd && amd64

package nn

import "simd/archsimd"

// axpy computes y[i] += alpha * x[i] for i in [0, len(x)). Requires
// len(x) == len(y). The body issues VFMADD231PS via simd/archsimd so a
// single instruction processes 16 fp32s (AVX-512) or 8 fp32s (AVX2); the
// tail covers any residual.
//
// This is the hottest kernel in MoEExperts (gate-up and down matmuls) and
// in attention's V accumulation. The GOAMD64=v3 scalar VFMADD231SS the
// compiler was emitting tops out near 6 GFLOPS/core; VFMADD231PS at 512-
// bit width targets ~80 GFLOPS/core.
func axpy(alpha float32, x, y []float32) {
	if len(x) != len(y) {
		panic("axpy: length mismatch")
	}
	n := len(x)
	a16 := archsimd.BroadcastFloat32x16(alpha)
	i := 0
	for ; i+16 <= n; i += 16 {
		xv := archsimd.LoadFloat32x16Slice(x[i:])
		yv := archsimd.LoadFloat32x16Slice(y[i:])
		a16.MulAdd(xv, yv).StoreSlice(y[i:])
	}
	if i+8 <= n {
		a8 := archsimd.BroadcastFloat32x8(alpha)
		xv := archsimd.LoadFloat32x8Slice(x[i:])
		yv := archsimd.LoadFloat32x8Slice(y[i:])
		a8.MulAdd(xv, yv).StoreSlice(y[i:])
		i += 8
	}
	for ; i < n; i++ {
		y[i] += alpha * x[i]
	}
}

// axpy2 fuses two axpys into a single store per block: for each 16-wide
// chunk of y, loads y once, issues two FMAs (a0*x0 + a1*x1 + y), stores
// y once. Used by MoE to roll two adjacent rows of gate-up / down-proj
// into one pass so store traffic halves and the gateUp buffer stays
// loaded in a register across a pair of input-dim iterations.
//
// Requires len(x0) == len(x1) == len(y).
func axpy2(a0, a1 float32, x0, x1, y []float32) {
	n := len(y)
	if len(x0) != n || len(x1) != n {
		panic("axpy2: length mismatch")
	}
	a0v16 := archsimd.BroadcastFloat32x16(a0)
	a1v16 := archsimd.BroadcastFloat32x16(a1)
	i := 0
	for ; i+16 <= n; i += 16 {
		yv := archsimd.LoadFloat32x16Slice(y[i:])
		x0v := archsimd.LoadFloat32x16Slice(x0[i:])
		x1v := archsimd.LoadFloat32x16Slice(x1[i:])
		yv = a0v16.MulAdd(x0v, yv)
		yv = a1v16.MulAdd(x1v, yv)
		yv.StoreSlice(y[i:])
	}
	if i+8 <= n {
		a0v8 := archsimd.BroadcastFloat32x8(a0)
		a1v8 := archsimd.BroadcastFloat32x8(a1)
		yv := archsimd.LoadFloat32x8Slice(y[i:])
		x0v := archsimd.LoadFloat32x8Slice(x0[i:])
		x1v := archsimd.LoadFloat32x8Slice(x1[i:])
		yv = a0v8.MulAdd(x0v, yv)
		yv = a1v8.MulAdd(x1v, yv)
		yv.StoreSlice(y[i:])
		i += 8
	}
	for ; i < n; i++ {
		y[i] += a0*x0[i] + a1*x1[i]
	}
}

// dot computes sum_i x[i] * w[i] using a packed 16-lane accumulator (with
// an 8-lane step-down for sub-16 tails), then horizontally sums at the
// end. len(x) must equal len(w). Used by Linear (Q/K/V + output + router +
// classifier) and attention's QK scoring loop.
func dot(x, w []float32) float32 {
	if len(x) != len(w) {
		panic("dot: length mismatch")
	}
	n := len(x)
	acc16 := archsimd.BroadcastFloat32x16(0)
	i := 0
	for ; i+16 <= n; i += 16 {
		xv := archsimd.LoadFloat32x16Slice(x[i:])
		wv := archsimd.LoadFloat32x16Slice(w[i:])
		acc16 = xv.MulAdd(wv, acc16)
	}
	var lanes [16]float32
	acc16.Store(&lanes)
	s := (((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) +
		((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]))) +
		(((lanes[8] + lanes[9]) + (lanes[10] + lanes[11])) +
			((lanes[12] + lanes[13]) + (lanes[14] + lanes[15])))
	if i+8 <= n {
		xv := archsimd.LoadFloat32x8Slice(x[i:])
		wv := archsimd.LoadFloat32x8Slice(w[i:])
		acc8 := archsimd.BroadcastFloat32x8(0).Add(xv.Mul(wv))
		var l8 [8]float32
		acc8.Store(&l8)
		s += ((l8[0] + l8[1]) + (l8[2] + l8[3])) +
			((l8[4] + l8[5]) + (l8[6] + l8[7]))
		i += 8
	}
	for ; i < n; i++ {
		s += x[i] * w[i]
	}
	return s
}
