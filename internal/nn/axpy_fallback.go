//go:build !goexperiment.simd || !amd64

package nn

import "math"

// softmaxExpSum: scalar fallback of the SIMD softmax exp+sum pass.
func softmaxExpSum(scores []float32, lo, hi int, maxLogit float32) float32 {
	var sum float32
	for j := lo; j < hi; j++ {
		e := float32(math.Exp(float64(scores[j] - maxLogit)))
		scores[j] = e
		sum += e
	}
	return sum
}

// softmaxScale: scalar fallback of the SIMD softmax normalize pass.
func softmaxScale(scores []float32, lo, hi int, factor float32) {
	for j := lo; j < hi; j++ {
		scores[j] *= factor
	}
}

// moeActivation: scalar fallback of the SIMD Quick-GELU-GLU activation.
func moeActivation(gateUp, gated []float32, I int, limit, alpha float32) {
	if len(gateUp) < 2*I || len(gated) < I {
		panic("moeActivation: length mismatch")
	}
	for i := 0; i < I; i++ {
		g := gateUp[i]
		u := gateUp[I+i]
		if g > limit {
			g = limit
		}
		if u > limit {
			u = limit
		} else if u < -limit {
			u = -limit
		}
		sig := float32(1.0 / (1.0 + math.Exp(-float64(g)*float64(alpha))))
		glu := g * sig
		gated[i] = (u + 1) * glu
	}
}

// axpyBatch: scalar fallback of the SIMD batched axpy.
func axpyBatch(alphas []float32, wRow []float32, y []float32, stride int) {
	n := len(alphas)
	w := len(wRow)
	for k := 0; k < n; k++ {
		alpha := alphas[k]
		off := k * stride
		for i := 0; i < w; i++ {
			y[off+i] += alpha * wRow[i]
		}
	}
}

// axpy: scalar fallback used when goexperiment.simd is off or we're not
// on amd64. Unrolled by 8 to match the hot-path layout; the SIMD variant
// in axpy_simd_amd64.go is drop-in equivalent.
func axpy(alpha float32, x, y []float32) {
	if len(x) != len(y) {
		panic("axpy: length mismatch")
	}
	n := len(x)
	i := 0
	for ; i+8 <= n; i += 8 {
		y[i] += alpha * x[i]
		y[i+1] += alpha * x[i+1]
		y[i+2] += alpha * x[i+2]
		y[i+3] += alpha * x[i+3]
		y[i+4] += alpha * x[i+4]
		y[i+5] += alpha * x[i+5]
		y[i+6] += alpha * x[i+6]
		y[i+7] += alpha * x[i+7]
	}
	for ; i < n; i++ {
		y[i] += alpha * x[i]
	}
}

// dot: scalar fallback of the SIMD dot product. Uses 8 parallel
// accumulators to break the reduction dependency chain.
func dot(x, w []float32) float32 {
	if len(x) != len(w) {
		panic("dot: length mismatch")
	}
	n := len(x)
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	i := 0
	for ; i+8 <= n; i += 8 {
		s0 += x[i] * w[i]
		s1 += x[i+1] * w[i+1]
		s2 += x[i+2] * w[i+2]
		s3 += x[i+3] * w[i+3]
		s4 += x[i+4] * w[i+4]
		s5 += x[i+5] * w[i+5]
		s6 += x[i+6] * w[i+6]
		s7 += x[i+7] * w[i+7]
	}
	s := ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7))
	for ; i < n; i++ {
		s += x[i] * w[i]
	}
	return s
}
