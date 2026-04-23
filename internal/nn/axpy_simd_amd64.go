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
