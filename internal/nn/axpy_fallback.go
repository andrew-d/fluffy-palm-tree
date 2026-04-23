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

// axpyBatch2: scalar fallback of the pair-d-fused batched axpy.
func axpyBatch2(alphas0, alphas1 []float32, wRow0, wRow1 []float32, y []float32, stride int) {
	n := len(alphas0)
	w := len(wRow0)
	for k := 0; k < n; k++ {
		a0 := alphas0[k]
		a1 := alphas1[k]
		off := k * stride
		for i := 0; i < w; i++ {
			y[off+i] += a0*wRow0[i] + a1*wRow1[i]
		}
	}
}

// axpyBatch4: scalar fallback of the 4-d-fused batched axpy.
func axpyBatch4(alphas0, alphas1, alphas2, alphas3 []float32, wRow0, wRow1, wRow2, wRow3 []float32, y []float32, stride int) {
	n := len(alphas0)
	w := len(wRow0)
	for k := 0; k < n; k++ {
		a0 := alphas0[k]
		a1 := alphas1[k]
		a2 := alphas2[k]
		a3 := alphas3[k]
		off := k * stride
		for i := 0; i < w; i++ {
			y[off+i] += a0*wRow0[i] + a1*wRow1[i] + a2*wRow2[i] + a3*wRow3[i]
		}
	}
}

// axpyBatch8: scalar fallback of the 8-d-fused batched axpy.
func axpyBatch8(
	a0, a1, a2, a3, a4, a5, a6, a7 []float32,
	w0, w1, w2, w3, w4, w5, w6, w7 []float32,
	y []float32, stride int,
) {
	n := len(a0)
	w := len(w0)
	for k := 0; k < n; k++ {
		ak0 := a0[k]
		ak1 := a1[k]
		ak2 := a2[k]
		ak3 := a3[k]
		ak4 := a4[k]
		ak5 := a5[k]
		ak6 := a6[k]
		ak7 := a7[k]
		off := k * stride
		for i := 0; i < w; i++ {
			y[off+i] += ak0*w0[i] + ak1*w1[i] + ak2*w2[i] + ak3*w3[i] +
				ak4*w4[i] + ak5*w5[i] + ak6*w6[i] + ak7*w7[i]
		}
	}
}

// axpyBatch16: scalar fallback of the 16-d-fused batched axpy.
func axpyBatch16(a [16][]float32, w [16][]float32, y []float32, stride int) {
	n := len(a[0])
	width := len(w[0])
	for k := 0; k < n; k++ {
		off := k * stride
		for i := 0; i < width; i++ {
			var acc float32
			for s := 0; s < 16; s++ {
				acc += a[s][k] * w[s][i]
			}
			y[off+i] += acc
		}
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

// dotBatch16: scalar fallback of the 16-token batched dot product.
func dotBatch16(w []float32, xs [16][]float32) [16]float32 {
	var result [16]float32
	n := len(w)
	for i := 0; i < n; i++ {
		wi := w[i]
		for s := 0; s < 16; s++ {
			result[s] += xs[s][i] * wi
		}
	}
	return result
}

// dotBatch8: scalar fallback of the 8-token batched dot product.
func dotBatch8(w []float32, xs [8][]float32) [8]float32 {
	var result [8]float32
	n := len(w)
	for i := 0; i < n; i++ {
		wi := w[i]
		for s := 0; s < 8; s++ {
			result[s] += xs[s][i] * wi
		}
	}
	return result
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
