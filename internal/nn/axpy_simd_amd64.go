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
	// Two independent 16-wide FMA chains per iteration: 32 fp32s of work
	// with two loads, two FMAs, two stores — breaks the per-iteration
	// store→load dep the single-chain body leaves on the critical path.
	for ; i+32 <= n; i += 32 {
		xv0 := archsimd.LoadFloat32x16Slice(x[i:])
		yv0 := archsimd.LoadFloat32x16Slice(y[i:])
		xv1 := archsimd.LoadFloat32x16Slice(x[i+16:])
		yv1 := archsimd.LoadFloat32x16Slice(y[i+16:])
		a16.MulAdd(xv0, yv0).StoreSlice(y[i:])
		a16.MulAdd(xv1, yv1).StoreSlice(y[i+16:])
	}
	if i+16 <= n {
		xv := archsimd.LoadFloat32x16Slice(x[i:])
		yv := archsimd.LoadFloat32x16Slice(y[i:])
		a16.MulAdd(xv, yv).StoreSlice(y[i:])
		i += 16
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
