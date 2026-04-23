//go:build goexperiment.simd && amd64

package nn

import (
	"math"
	"simd/archsimd"
)

// axpyBatch fans out a single wRow across N output buffers. Each y[k]
// gets y[k][i] += alphas[k] * wRow[i] for i in [0, len(wRow)). Equivalent
// to calling axpy N times with the same wRow; the combined form hoists
// one wRow load into the outer loop so the scheduler sees a cleaner
// sequence of independent FMAs and skips per-call frame overhead.
//
// y must hold N*stride floats; each y[k] occupies offsets [k*stride,
// k*stride+len(wRow)). strides larger than len(wRow) are fine (caller
// pads between buffers).
func axpyBatch(alphas []float32, wRow []float32, y []float32, stride int) {
	n := len(alphas)
	w := len(wRow)
	i := 0
	for ; i+16 <= w; i += 16 {
		wv := archsimd.LoadFloat32x16Slice(wRow[i:])
		for k := 0; k < n; k++ {
			off := k*stride + i
			yv := archsimd.LoadFloat32x16Slice(y[off:])
			a := archsimd.BroadcastFloat32x16(alphas[k])
			a.MulAdd(wv, yv).StoreSlice(y[off:])
		}
	}
	if i+8 <= w {
		wv := archsimd.LoadFloat32x8Slice(wRow[i:])
		for k := 0; k < n; k++ {
			off := k*stride + i
			yv := archsimd.LoadFloat32x8Slice(y[off:])
			a := archsimd.BroadcastFloat32x8(alphas[k])
			a.MulAdd(wv, yv).StoreSlice(y[off:])
		}
		i += 8
	}
	for ; i < w; i++ {
		wi := wRow[i]
		for k := 0; k < n; k++ {
			y[k*stride+i] += alphas[k] * wi
		}
	}
}

// axpyBatch2 fuses two adjacent "d" iterations of the MoE gate-up / down
// matmul so each N-wide gateUp/out block incurs ONE load + ONE store per
// pair of d steps instead of two. Saves ~50% of the store traffic on the
// hot kernel.
//
// y[k][i:+16] += alphas0[k]*wRow0[i:+16] + alphas1[k]*wRow1[i:+16]
// for k in [0, len(alphas0)), i striding 16.
func axpyBatch2(alphas0, alphas1 []float32, wRow0, wRow1 []float32, y []float32, stride int) {
	n := len(alphas0)
	w := len(wRow0)
	i := 0
	for ; i+16 <= w; i += 16 {
		wv0 := archsimd.LoadFloat32x16Slice(wRow0[i:])
		wv1 := archsimd.LoadFloat32x16Slice(wRow1[i:])
		for k := 0; k < n; k++ {
			off := k*stride + i
			yv := archsimd.LoadFloat32x16Slice(y[off:])
			a0 := archsimd.BroadcastFloat32x16(alphas0[k])
			a1 := archsimd.BroadcastFloat32x16(alphas1[k])
			yv = a0.MulAdd(wv0, yv)
			yv = a1.MulAdd(wv1, yv)
			yv.StoreSlice(y[off:])
		}
	}
	if i+8 <= w {
		wv0 := archsimd.LoadFloat32x8Slice(wRow0[i:])
		wv1 := archsimd.LoadFloat32x8Slice(wRow1[i:])
		for k := 0; k < n; k++ {
			off := k*stride + i
			yv := archsimd.LoadFloat32x8Slice(y[off:])
			a0 := archsimd.BroadcastFloat32x8(alphas0[k])
			a1 := archsimd.BroadcastFloat32x8(alphas1[k])
			yv = a0.MulAdd(wv0, yv)
			yv = a1.MulAdd(wv1, yv)
			yv.StoreSlice(y[off:])
		}
		i += 8
	}
	for ; i < w; i++ {
		w0i := wRow0[i]
		w1i := wRow1[i]
		for k := 0; k < n; k++ {
			y[k*stride+i] += alphas0[k]*w0i + alphas1[k]*w1i
		}
	}
}

// axpyBatch4 fuses four adjacent "d" iterations. Each 16-wide gateUp/out
// block incurs ONE load + ONE store per four d-steps instead of four.
// Further cuts MoE store traffic beyond axpyBatch2's 2× reduction.
//
// y[k][i:+16] += sum_s alphas[s][k]*wRows[s][i:+16] for s in 0..3.
func axpyBatch4(alphas0, alphas1, alphas2, alphas3 []float32, wRow0, wRow1, wRow2, wRow3 []float32, y []float32, stride int) {
	n := len(alphas0)
	w := len(wRow0)
	i := 0
	for ; i+16 <= w; i += 16 {
		wv0 := archsimd.LoadFloat32x16Slice(wRow0[i:])
		wv1 := archsimd.LoadFloat32x16Slice(wRow1[i:])
		wv2 := archsimd.LoadFloat32x16Slice(wRow2[i:])
		wv3 := archsimd.LoadFloat32x16Slice(wRow3[i:])
		for k := 0; k < n; k++ {
			off := k*stride + i
			yv := archsimd.LoadFloat32x16Slice(y[off:])
			yv = archsimd.BroadcastFloat32x16(alphas0[k]).MulAdd(wv0, yv)
			yv = archsimd.BroadcastFloat32x16(alphas1[k]).MulAdd(wv1, yv)
			yv = archsimd.BroadcastFloat32x16(alphas2[k]).MulAdd(wv2, yv)
			yv = archsimd.BroadcastFloat32x16(alphas3[k]).MulAdd(wv3, yv)
			yv.StoreSlice(y[off:])
		}
	}
	// Tail: fall back to axpyBatch2 + axpyBatch on the remainder rather
	// than repeating the 8-wide / scalar spaghetti.
	if i < w {
		rem0 := wRow0[i:]
		rem1 := wRow1[i:]
		rem2 := wRow2[i:]
		rem3 := wRow3[i:]
		// scalar tail (uncommon on our shapes)
		for j := 0; j < len(rem0); j++ {
			w0 := rem0[j]
			w1 := rem1[j]
			w2 := rem2[j]
			w3 := rem3[j]
			for k := 0; k < n; k++ {
				y[k*stride+i+j] += alphas0[k]*w0 + alphas1[k]*w1 +
					alphas2[k]*w2 + alphas3[k]*w3
			}
		}
	}
}

// axpyBatch8 fuses eight adjacent "d" iterations per gateUp/out block.
// One load + one store per eight d-steps. Fits comfortably in AVX-512's
// 32 ZMM registers: 8 wRow registers + 1 y/accumulator + scratch for the
// per-k alpha broadcast.
func axpyBatch8(
	a0, a1, a2, a3, a4, a5, a6, a7 []float32,
	w0, w1, w2, w3, w4, w5, w6, w7 []float32,
	y []float32, stride int,
) {
	n := len(a0)
	w := len(w0)
	i := 0
	for ; i+16 <= w; i += 16 {
		wv0 := archsimd.LoadFloat32x16Slice(w0[i:])
		wv1 := archsimd.LoadFloat32x16Slice(w1[i:])
		wv2 := archsimd.LoadFloat32x16Slice(w2[i:])
		wv3 := archsimd.LoadFloat32x16Slice(w3[i:])
		wv4 := archsimd.LoadFloat32x16Slice(w4[i:])
		wv5 := archsimd.LoadFloat32x16Slice(w5[i:])
		wv6 := archsimd.LoadFloat32x16Slice(w6[i:])
		wv7 := archsimd.LoadFloat32x16Slice(w7[i:])
		for k := 0; k < n; k++ {
			off := k*stride + i
			yv := archsimd.LoadFloat32x16Slice(y[off:])
			yv = archsimd.BroadcastFloat32x16(a0[k]).MulAdd(wv0, yv)
			yv = archsimd.BroadcastFloat32x16(a1[k]).MulAdd(wv1, yv)
			yv = archsimd.BroadcastFloat32x16(a2[k]).MulAdd(wv2, yv)
			yv = archsimd.BroadcastFloat32x16(a3[k]).MulAdd(wv3, yv)
			yv = archsimd.BroadcastFloat32x16(a4[k]).MulAdd(wv4, yv)
			yv = archsimd.BroadcastFloat32x16(a5[k]).MulAdd(wv5, yv)
			yv = archsimd.BroadcastFloat32x16(a6[k]).MulAdd(wv6, yv)
			yv = archsimd.BroadcastFloat32x16(a7[k]).MulAdd(wv7, yv)
			yv.StoreSlice(y[off:])
		}
	}
	if i < w {
		for j := i; j < w; j++ {
			v0 := w0[j]
			v1 := w1[j]
			v2 := w2[j]
			v3 := w3[j]
			v4 := w4[j]
			v5 := w5[j]
			v6 := w6[j]
			v7 := w7[j]
			for k := 0; k < n; k++ {
				y[k*stride+j] += a0[k]*v0 + a1[k]*v1 + a2[k]*v2 + a3[k]*v3 +
					a4[k]*v4 + a5[k]*v5 + a6[k]*v6 + a7[k]*v7
			}
		}
	}
}

// axpyBatch16 fuses sixteen adjacent "d" iterations. One load + one store
// per sixteen d-steps. 16 wRow registers + accumulator + broadcasts are
// at the edge of AVX-512's 32-ZMM budget but still fit; if the compiler
// spills, L1 latency is cheap compared to the store savings.
func axpyBatch16(
	a [16][]float32,
	w [16][]float32,
	y []float32, stride int,
) {
	n := len(a[0])
	width := len(w[0])
	i := 0
	for ; i+16 <= width; i += 16 {
		wv0 := archsimd.LoadFloat32x16Slice(w[0][i:])
		wv1 := archsimd.LoadFloat32x16Slice(w[1][i:])
		wv2 := archsimd.LoadFloat32x16Slice(w[2][i:])
		wv3 := archsimd.LoadFloat32x16Slice(w[3][i:])
		wv4 := archsimd.LoadFloat32x16Slice(w[4][i:])
		wv5 := archsimd.LoadFloat32x16Slice(w[5][i:])
		wv6 := archsimd.LoadFloat32x16Slice(w[6][i:])
		wv7 := archsimd.LoadFloat32x16Slice(w[7][i:])
		wv8 := archsimd.LoadFloat32x16Slice(w[8][i:])
		wv9 := archsimd.LoadFloat32x16Slice(w[9][i:])
		wv10 := archsimd.LoadFloat32x16Slice(w[10][i:])
		wv11 := archsimd.LoadFloat32x16Slice(w[11][i:])
		wv12 := archsimd.LoadFloat32x16Slice(w[12][i:])
		wv13 := archsimd.LoadFloat32x16Slice(w[13][i:])
		wv14 := archsimd.LoadFloat32x16Slice(w[14][i:])
		wv15 := archsimd.LoadFloat32x16Slice(w[15][i:])
		k := 0
		for ; k+2 <= n; k += 2 {
			off0 := k*stride + i
			off1 := (k+1)*stride + i
			yv0 := archsimd.LoadFloat32x16Slice(y[off0:])
			yv1 := archsimd.LoadFloat32x16Slice(y[off1:])
			yv0 = archsimd.BroadcastFloat32x16(a[0][k]).MulAdd(wv0, yv0)
			yv1 = archsimd.BroadcastFloat32x16(a[0][k+1]).MulAdd(wv0, yv1)
			yv0 = archsimd.BroadcastFloat32x16(a[1][k]).MulAdd(wv1, yv0)
			yv1 = archsimd.BroadcastFloat32x16(a[1][k+1]).MulAdd(wv1, yv1)
			yv0 = archsimd.BroadcastFloat32x16(a[2][k]).MulAdd(wv2, yv0)
			yv1 = archsimd.BroadcastFloat32x16(a[2][k+1]).MulAdd(wv2, yv1)
			yv0 = archsimd.BroadcastFloat32x16(a[3][k]).MulAdd(wv3, yv0)
			yv1 = archsimd.BroadcastFloat32x16(a[3][k+1]).MulAdd(wv3, yv1)
			yv0 = archsimd.BroadcastFloat32x16(a[4][k]).MulAdd(wv4, yv0)
			yv1 = archsimd.BroadcastFloat32x16(a[4][k+1]).MulAdd(wv4, yv1)
			yv0 = archsimd.BroadcastFloat32x16(a[5][k]).MulAdd(wv5, yv0)
			yv1 = archsimd.BroadcastFloat32x16(a[5][k+1]).MulAdd(wv5, yv1)
			yv0 = archsimd.BroadcastFloat32x16(a[6][k]).MulAdd(wv6, yv0)
			yv1 = archsimd.BroadcastFloat32x16(a[6][k+1]).MulAdd(wv6, yv1)
			yv0 = archsimd.BroadcastFloat32x16(a[7][k]).MulAdd(wv7, yv0)
			yv1 = archsimd.BroadcastFloat32x16(a[7][k+1]).MulAdd(wv7, yv1)
			yv0 = archsimd.BroadcastFloat32x16(a[8][k]).MulAdd(wv8, yv0)
			yv1 = archsimd.BroadcastFloat32x16(a[8][k+1]).MulAdd(wv8, yv1)
			yv0 = archsimd.BroadcastFloat32x16(a[9][k]).MulAdd(wv9, yv0)
			yv1 = archsimd.BroadcastFloat32x16(a[9][k+1]).MulAdd(wv9, yv1)
			yv0 = archsimd.BroadcastFloat32x16(a[10][k]).MulAdd(wv10, yv0)
			yv1 = archsimd.BroadcastFloat32x16(a[10][k+1]).MulAdd(wv10, yv1)
			yv0 = archsimd.BroadcastFloat32x16(a[11][k]).MulAdd(wv11, yv0)
			yv1 = archsimd.BroadcastFloat32x16(a[11][k+1]).MulAdd(wv11, yv1)
			yv0 = archsimd.BroadcastFloat32x16(a[12][k]).MulAdd(wv12, yv0)
			yv1 = archsimd.BroadcastFloat32x16(a[12][k+1]).MulAdd(wv12, yv1)
			yv0 = archsimd.BroadcastFloat32x16(a[13][k]).MulAdd(wv13, yv0)
			yv1 = archsimd.BroadcastFloat32x16(a[13][k+1]).MulAdd(wv13, yv1)
			yv0 = archsimd.BroadcastFloat32x16(a[14][k]).MulAdd(wv14, yv0)
			yv1 = archsimd.BroadcastFloat32x16(a[14][k+1]).MulAdd(wv14, yv1)
			yv0 = archsimd.BroadcastFloat32x16(a[15][k]).MulAdd(wv15, yv0)
			yv1 = archsimd.BroadcastFloat32x16(a[15][k+1]).MulAdd(wv15, yv1)
			yv0.StoreSlice(y[off0:])
			yv1.StoreSlice(y[off1:])
		}
		for ; k < n; k++ {
			off := k*stride + i
			yv := archsimd.LoadFloat32x16Slice(y[off:])
			yv = archsimd.BroadcastFloat32x16(a[0][k]).MulAdd(wv0, yv)
			yv = archsimd.BroadcastFloat32x16(a[1][k]).MulAdd(wv1, yv)
			yv = archsimd.BroadcastFloat32x16(a[2][k]).MulAdd(wv2, yv)
			yv = archsimd.BroadcastFloat32x16(a[3][k]).MulAdd(wv3, yv)
			yv = archsimd.BroadcastFloat32x16(a[4][k]).MulAdd(wv4, yv)
			yv = archsimd.BroadcastFloat32x16(a[5][k]).MulAdd(wv5, yv)
			yv = archsimd.BroadcastFloat32x16(a[6][k]).MulAdd(wv6, yv)
			yv = archsimd.BroadcastFloat32x16(a[7][k]).MulAdd(wv7, yv)
			yv = archsimd.BroadcastFloat32x16(a[8][k]).MulAdd(wv8, yv)
			yv = archsimd.BroadcastFloat32x16(a[9][k]).MulAdd(wv9, yv)
			yv = archsimd.BroadcastFloat32x16(a[10][k]).MulAdd(wv10, yv)
			yv = archsimd.BroadcastFloat32x16(a[11][k]).MulAdd(wv11, yv)
			yv = archsimd.BroadcastFloat32x16(a[12][k]).MulAdd(wv12, yv)
			yv = archsimd.BroadcastFloat32x16(a[13][k]).MulAdd(wv13, yv)
			yv = archsimd.BroadcastFloat32x16(a[14][k]).MulAdd(wv14, yv)
			yv = archsimd.BroadcastFloat32x16(a[15][k]).MulAdd(wv15, yv)
			yv.StoreSlice(y[off:])
		}
	}
	if i < width {
		for j := i; j < width; j++ {
			var v [16]float32
			for s := 0; s < 16; s++ {
				v[s] = w[s][j]
			}
			for k := 0; k < n; k++ {
				var acc float32
				for s := 0; s < 16; s++ {
					acc += a[s][k] * v[s]
				}
				y[k*stride+j] += acc
			}
		}
	}
}

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

// fastExp16 approximates exp(x) lanewise to ~2-3 ULP for x in roughly
// [-87, 87]. Outside that range lanes over/underflow normally; for our
// use the argument is bounded by limit*alpha ≈ ±12.
//
// Algorithm: x = n*ln2 + r with n ∈ Z and r ∈ (-ln2/2, ln2/2], so exp(x)
// = 2^n * exp(r). 2^n comes from packing (n+127) into the fp32 exponent
// field; exp(r) is a degree-6 Horner polynomial tuned for minimax over
// the reduced range (the Cephes/Cody-Waite choice).
func fastExp16(x archsimd.Float32x16) archsimd.Float32x16 {
	log2e := archsimd.BroadcastFloat32x16(1.4426950408889634)  // 1/ln2
	ln2 := archsimd.BroadcastFloat32x16(0.6931471805599453)    // ln2
	nRound := x.Mul(log2e).RoundToEvenScaled(0)
	nInt := nRound.ConvertToInt32()
	r := x.Sub(nRound.Mul(ln2))

	c6 := archsimd.BroadcastFloat32x16(1.0 / 720)
	c5 := archsimd.BroadcastFloat32x16(1.0 / 120)
	c4 := archsimd.BroadcastFloat32x16(1.0 / 24)
	c3 := archsimd.BroadcastFloat32x16(1.0 / 6)
	c2 := archsimd.BroadcastFloat32x16(1.0 / 2)
	one := archsimd.BroadcastFloat32x16(1.0)

	p := c6.MulAdd(r, c5)
	p = p.MulAdd(r, c4)
	p = p.MulAdd(r, c3)
	p = p.MulAdd(r, c2)
	p = p.MulAdd(r, one)
	p = p.MulAdd(r, one)

	// Pack (n+127) into the fp32 exponent field. Clamp to [0, 254] so
	// lanes that would underflow (n < -127) produce +0 instead of wrapping
	// into the sign bit, and lanes that would overflow (n > 127) produce
	// the largest finite 2^n instead of NaN. This is the correctness
	// delta that softmax amplifies when arg << 0 for masked positions.
	bias := archsimd.BroadcastInt32x16(127)
	zero := archsimd.BroadcastInt32x16(0)
	maxExp := archsimd.BroadcastInt32x16(254)
	nShifted := nInt.Add(bias).Max(zero).Min(maxExp)
	pow2 := nShifted.ShiftAllLeft(23).AsFloat32x16()

	return p.Mul(pow2)
}

// softmaxExpSum replaces scores[lo:hi] with exp(scores[lo:hi] - maxLogit)
// and returns the sum of the exp values. Used by GQAAttentionWithSinks
// to compute the softmax denominator in one vectorized pass.
func softmaxExpSum(scores []float32, lo, hi int, maxLogit float32) float32 {
	maxV := archsimd.BroadcastFloat32x16(maxLogit)
	accV := archsimd.BroadcastFloat32x16(0)
	j := lo
	for ; j+16 <= hi; j += 16 {
		sv := archsimd.LoadFloat32x16Slice(scores[j:])
		e := fastExp16(sv.Sub(maxV))
		e.StoreSlice(scores[j:])
		accV = accV.Add(e)
	}
	var lanes [16]float32
	accV.Store(&lanes)
	sum := (((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) +
		((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]))) +
		(((lanes[8] + lanes[9]) + (lanes[10] + lanes[11])) +
			((lanes[12] + lanes[13]) + (lanes[14] + lanes[15])))
	for ; j < hi; j++ {
		e := float32(math.Exp(float64(scores[j] - maxLogit)))
		scores[j] = e
		sum += e
	}
	return sum
}

// softmaxScale multiplies scores[lo:hi] by scalar factor.
func softmaxScale(scores []float32, lo, hi int, factor float32) {
	factorV := archsimd.BroadcastFloat32x16(factor)
	j := lo
	for ; j+16 <= hi; j += 16 {
		sv := archsimd.LoadFloat32x16Slice(scores[j:])
		sv.Mul(factorV).StoreSlice(scores[j:])
	}
	for ; j < hi; j++ {
		scores[j] *= factor
	}
}

// moeActivation computes the Quick-GELU-gated GLU activation used inside
// MoEExperts:
//
//	for i in [0, I):
//	    g = min(gateUp[i], limit)
//	    u = clamp(gateUp[I+i], -limit, limit)
//	    sig = 1 / (1 + exp(-g*alpha))
//	    gated[i] = (u + 1) * g * sig
//
// Vectorized with fastExp16 so a single Float32x16 pass covers what used
// to be 16 scalar math.archExp calls.
func moeActivation(gateUp, gated []float32, I int, limit, alpha float32) {
	if len(gateUp) < 2*I || len(gated) < I {
		panic("moeActivation: length mismatch")
	}
	limitV := archsimd.BroadcastFloat32x16(limit)
	negLimitV := archsimd.BroadcastFloat32x16(-limit)
	negAlphaV := archsimd.BroadcastFloat32x16(-alpha)
	oneV := archsimd.BroadcastFloat32x16(1.0)

	i := 0
	for ; i+16 <= I; i += 16 {
		g := archsimd.LoadFloat32x16Slice(gateUp[i:]).Min(limitV)
		u := archsimd.LoadFloat32x16Slice(gateUp[I+i:]).Max(negLimitV).Min(limitV)

		e := fastExp16(g.Mul(negAlphaV))
		sig := oneV.Div(oneV.Add(e))
		glu := g.Mul(sig)
		out := u.Add(oneV).Mul(glu)
		out.StoreSlice(gated[i:])
	}
	// Scalar tail (uncommon — I=640 is divisible by 16 on the current model).
	for ; i < I; i++ {
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

// dotBatch8 computes 8 dot products against the same w row, sharing the
// w loads across 8 parallel Float32x16 accumulators. Used by Linear to
// amortize wRow bandwidth across 8 adjacent tokens per output column.
func dotBatch8(w []float32, xs [8][]float32) [8]float32 {
	var acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7 archsimd.Float32x16
	acc0 = archsimd.BroadcastFloat32x16(0)
	acc1 = archsimd.BroadcastFloat32x16(0)
	acc2 = archsimd.BroadcastFloat32x16(0)
	acc3 = archsimd.BroadcastFloat32x16(0)
	acc4 = archsimd.BroadcastFloat32x16(0)
	acc5 = archsimd.BroadcastFloat32x16(0)
	acc6 = archsimd.BroadcastFloat32x16(0)
	acc7 = archsimd.BroadcastFloat32x16(0)
	n := len(w)
	i := 0
	for ; i+16 <= n; i += 16 {
		wv := archsimd.LoadFloat32x16Slice(w[i:])
		acc0 = archsimd.LoadFloat32x16Slice(xs[0][i:]).MulAdd(wv, acc0)
		acc1 = archsimd.LoadFloat32x16Slice(xs[1][i:]).MulAdd(wv, acc1)
		acc2 = archsimd.LoadFloat32x16Slice(xs[2][i:]).MulAdd(wv, acc2)
		acc3 = archsimd.LoadFloat32x16Slice(xs[3][i:]).MulAdd(wv, acc3)
		acc4 = archsimd.LoadFloat32x16Slice(xs[4][i:]).MulAdd(wv, acc4)
		acc5 = archsimd.LoadFloat32x16Slice(xs[5][i:]).MulAdd(wv, acc5)
		acc6 = archsimd.LoadFloat32x16Slice(xs[6][i:]).MulAdd(wv, acc6)
		acc7 = archsimd.LoadFloat32x16Slice(xs[7][i:]).MulAdd(wv, acc7)
	}
	accs := [8]archsimd.Float32x16{acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7}
	var result [8]float32
	for s := 0; s < 8; s++ {
		var lanes [16]float32
		accs[s].Store(&lanes)
		result[s] = (((lanes[0]+lanes[1])+(lanes[2]+lanes[3]))+
			((lanes[4]+lanes[5])+(lanes[6]+lanes[7]))) +
			(((lanes[8]+lanes[9])+(lanes[10]+lanes[11]))+
				((lanes[12]+lanes[13])+(lanes[14]+lanes[15])))
	}
	for ; i < n; i++ {
		wi := w[i]
		for s := 0; s < 8; s++ {
			result[s] += xs[s][i] * wi
		}
	}
	return result
}

// linearTile4x4 computes a 4 (token) × 4 (output) C-tile of Linear with
// K as the outer reduction dimension. Each of the 16 accumulators is a
// Float32x16 that sums partial products for one (token, output) pair;
// the K loop loads 4 x-vectors + 4 w-vectors per step and issues 16 FMAs
// against them (MN-reuse ratio 4 — each x and w load feeds 4 FMAs).
//
// This is tinyBLAS's Kazushige-Goto-style inner kernel adapted for our
// simd/archsimd stack. Compared to dotBatch8 (reuse ratio ≈ 0.89), this
// doubles arithmetic intensity per load, which is the lever whenever
// Linear is load-bandwidth bound (Q/O projections on our shape).
//
// Precondition: tOff+4 ≤ T, oOff+4 ≤ out, in%16 == 0 (checked at call site).
func linearTile4x4(x []float32, W []float32, y []float32, in, out, tOff, oOff int, bias []float32) {
	z := archsimd.BroadcastFloat32x16(0)
	a00, a01, a02, a03 := z, z, z, z
	a10, a11, a12, a13 := z, z, z, z
	a20, a21, a22, a23 := z, z, z, z
	a30, a31, a32, a33 := z, z, z, z

	xBase := tOff * in
	wBase := oOff * in
	k := 0
	for ; k+32 <= in; k += 32 {
		// First 16 K-step
		xv0 := archsimd.LoadFloat32x16Slice(x[xBase+0*in+k:])
		xv1 := archsimd.LoadFloat32x16Slice(x[xBase+1*in+k:])
		xv2 := archsimd.LoadFloat32x16Slice(x[xBase+2*in+k:])
		xv3 := archsimd.LoadFloat32x16Slice(x[xBase+3*in+k:])
		wv0 := archsimd.LoadFloat32x16Slice(W[wBase+0*in+k:])
		wv1 := archsimd.LoadFloat32x16Slice(W[wBase+1*in+k:])
		wv2 := archsimd.LoadFloat32x16Slice(W[wBase+2*in+k:])
		wv3 := archsimd.LoadFloat32x16Slice(W[wBase+3*in+k:])
		a00 = xv0.MulAdd(wv0, a00)
		a01 = xv0.MulAdd(wv1, a01)
		a02 = xv0.MulAdd(wv2, a02)
		a03 = xv0.MulAdd(wv3, a03)
		a10 = xv1.MulAdd(wv0, a10)
		a11 = xv1.MulAdd(wv1, a11)
		a12 = xv1.MulAdd(wv2, a12)
		a13 = xv1.MulAdd(wv3, a13)
		a20 = xv2.MulAdd(wv0, a20)
		a21 = xv2.MulAdd(wv1, a21)
		a22 = xv2.MulAdd(wv2, a22)
		a23 = xv2.MulAdd(wv3, a23)
		a30 = xv3.MulAdd(wv0, a30)
		a31 = xv3.MulAdd(wv1, a31)
		a32 = xv3.MulAdd(wv2, a32)
		a33 = xv3.MulAdd(wv3, a33)
		// Second 16 K-step
		xv0 = archsimd.LoadFloat32x16Slice(x[xBase+0*in+k+16:])
		xv1 = archsimd.LoadFloat32x16Slice(x[xBase+1*in+k+16:])
		xv2 = archsimd.LoadFloat32x16Slice(x[xBase+2*in+k+16:])
		xv3 = archsimd.LoadFloat32x16Slice(x[xBase+3*in+k+16:])
		wv0 = archsimd.LoadFloat32x16Slice(W[wBase+0*in+k+16:])
		wv1 = archsimd.LoadFloat32x16Slice(W[wBase+1*in+k+16:])
		wv2 = archsimd.LoadFloat32x16Slice(W[wBase+2*in+k+16:])
		wv3 = archsimd.LoadFloat32x16Slice(W[wBase+3*in+k+16:])
		a00 = xv0.MulAdd(wv0, a00)
		a01 = xv0.MulAdd(wv1, a01)
		a02 = xv0.MulAdd(wv2, a02)
		a03 = xv0.MulAdd(wv3, a03)
		a10 = xv1.MulAdd(wv0, a10)
		a11 = xv1.MulAdd(wv1, a11)
		a12 = xv1.MulAdd(wv2, a12)
		a13 = xv1.MulAdd(wv3, a13)
		a20 = xv2.MulAdd(wv0, a20)
		a21 = xv2.MulAdd(wv1, a21)
		a22 = xv2.MulAdd(wv2, a22)
		a23 = xv2.MulAdd(wv3, a23)
		a30 = xv3.MulAdd(wv0, a30)
		a31 = xv3.MulAdd(wv1, a31)
		a32 = xv3.MulAdd(wv2, a32)
		a33 = xv3.MulAdd(wv3, a33)
	}
	for ; k+16 <= in; k += 16 {
		xv0 := archsimd.LoadFloat32x16Slice(x[xBase+0*in+k:])
		xv1 := archsimd.LoadFloat32x16Slice(x[xBase+1*in+k:])
		xv2 := archsimd.LoadFloat32x16Slice(x[xBase+2*in+k:])
		xv3 := archsimd.LoadFloat32x16Slice(x[xBase+3*in+k:])
		wv0 := archsimd.LoadFloat32x16Slice(W[wBase+0*in+k:])
		wv1 := archsimd.LoadFloat32x16Slice(W[wBase+1*in+k:])
		wv2 := archsimd.LoadFloat32x16Slice(W[wBase+2*in+k:])
		wv3 := archsimd.LoadFloat32x16Slice(W[wBase+3*in+k:])

		a00 = xv0.MulAdd(wv0, a00)
		a01 = xv0.MulAdd(wv1, a01)
		a02 = xv0.MulAdd(wv2, a02)
		a03 = xv0.MulAdd(wv3, a03)
		a10 = xv1.MulAdd(wv0, a10)
		a11 = xv1.MulAdd(wv1, a11)
		a12 = xv1.MulAdd(wv2, a12)
		a13 = xv1.MulAdd(wv3, a13)
		a20 = xv2.MulAdd(wv0, a20)
		a21 = xv2.MulAdd(wv1, a21)
		a22 = xv2.MulAdd(wv2, a22)
		a23 = xv2.MulAdd(wv3, a23)
		a30 = xv3.MulAdd(wv0, a30)
		a31 = xv3.MulAdd(wv1, a31)
		a32 = xv3.MulAdd(wv2, a32)
		a33 = xv3.MulAdd(wv3, a33)
	}

	hsum := func(v archsimd.Float32x16) float32 {
		var l [16]float32
		v.Store(&l)
		return (((l[0]+l[1])+(l[2]+l[3]))+((l[4]+l[5])+(l[6]+l[7]))) +
			(((l[8]+l[9])+(l[10]+l[11]))+((l[12]+l[13])+(l[14]+l[15])))
	}
	var b0, b1, b2, b3 float32
	if bias != nil {
		b0 = bias[oOff+0]
		b1 = bias[oOff+1]
		b2 = bias[oOff+2]
		b3 = bias[oOff+3]
	}
	y[(tOff+0)*out+oOff+0] = hsum(a00) + b0
	y[(tOff+0)*out+oOff+1] = hsum(a01) + b1
	y[(tOff+0)*out+oOff+2] = hsum(a02) + b2
	y[(tOff+0)*out+oOff+3] = hsum(a03) + b3
	y[(tOff+1)*out+oOff+0] = hsum(a10) + b0
	y[(tOff+1)*out+oOff+1] = hsum(a11) + b1
	y[(tOff+1)*out+oOff+2] = hsum(a12) + b2
	y[(tOff+1)*out+oOff+3] = hsum(a13) + b3
	y[(tOff+2)*out+oOff+0] = hsum(a20) + b0
	y[(tOff+2)*out+oOff+1] = hsum(a21) + b1
	y[(tOff+2)*out+oOff+2] = hsum(a22) + b2
	y[(tOff+2)*out+oOff+3] = hsum(a23) + b3
	y[(tOff+3)*out+oOff+0] = hsum(a30) + b0
	y[(tOff+3)*out+oOff+1] = hsum(a31) + b1
	y[(tOff+3)*out+oOff+2] = hsum(a32) + b2
	y[(tOff+3)*out+oOff+3] = hsum(a33) + b3
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
