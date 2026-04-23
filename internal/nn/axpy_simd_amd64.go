//go:build goexperiment.simd && amd64

package nn

import "simd/archsimd"

// axpy computes y[i] += alpha * x[i] for i in [0, len(x)). Requires
// len(x) == len(y). The 8-wide body uses VFMADD231PS via the simd/archsimd
// package so each instruction processes 8 fp32s at once; the tail covers
// any residual.
//
// This is the hottest kernel in MoEExperts (gate-up and down matmuls).
// The GOAMD64=v3 scalar VFMADD231SS the compiler was emitting tops out
// near 6 GFLOPS/core; VFMADD231PS lifts the ceiling to ~40 GFLOPS/core.
func axpy(alpha float32, x, y []float32) {
	if len(x) != len(y) {
		panic("axpy: length mismatch")
	}
	a := archsimd.BroadcastFloat32x8(alpha)
	n := len(x)
	i := 0
	for ; i+8 <= n; i += 8 {
		xv := archsimd.LoadFloat32x8Slice(x[i:])
		yv := archsimd.LoadFloat32x8Slice(y[i:])
		// MulAdd is x*y + z, computed in one VFMADD231PS per lane-group.
		r := a.MulAdd(xv, yv)
		r.StoreSlice(y[i:])
	}
	for ; i < n; i++ {
		y[i] += alpha * x[i]
	}
}

// dot computes sum_i x[i] * w[i] using a packed 8-lane accumulator, then
// horizontally sums at the end. len(x) must equal len(w). Used by Linear
// for Q/K/V + output projections, router, and the classifier head.
func dot(x, w []float32) float32 {
	if len(x) != len(w) {
		panic("dot: length mismatch")
	}
	n := len(x)
	acc := archsimd.BroadcastFloat32x8(0)
	i := 0
	for ; i+8 <= n; i += 8 {
		xv := archsimd.LoadFloat32x8Slice(x[i:])
		wv := archsimd.LoadFloat32x8Slice(w[i:])
		acc = xv.MulAdd(wv, acc)
	}
	var lanes [8]float32
	acc.Store(&lanes)
	s := ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) +
		((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]))
	for ; i < n; i++ {
		s += x[i] * w[i]
	}
	return s
}
