// Package nn provides the small set of tensor primitives used by the
// privacy-filter forward pass. All math is done in float32, row-major, with
// flat `[]float32` buffers.
package nn

import "math"

// RMSNorm applies `rsqrt(mean(x^2) + eps) * x * weight` row-wise over the
// last dimension, matching the HuggingFace gpt_oss layer-norm:
//
//	variance = x.pow(2).mean(-1, keepdim=True)
//	x = x * rsqrt(variance + eps)
//	return weight * x
//
// All math is in fp32. The weight vector has length D.
//
//	x:      [T, D] flat row-major input
//	weight: [D]
//	T, D:   the two dims of x (T rows, D columns)
//	eps:    variance epsilon (1e-5 for this model)
//
// Returns a freshly allocated [T, D] slice.
func RMSNorm(x []float32, weight []float32, T, D int, eps float32) []float32 {
	if len(x) != T*D {
		panic("RMSNorm: x length mismatch")
	}
	if len(weight) != D {
		panic("RMSNorm: weight length mismatch")
	}
	out := make([]float32, T*D)
	epsd := float64(eps)
	for t := 0; t < T; t++ {
		row := x[t*D : (t+1)*D]
		// Mean of squares in fp64 to keep the reduction precise; the final
		// per-element multiply is still in fp32.
		var ss float64
		for i := 0; i < D; i++ {
			v := float64(row[i])
			ss += v * v
		}
		rms := 1.0 / math.Sqrt(ss/float64(D)+epsd)
		dst := out[t*D : (t+1)*D]
		for i := 0; i < D; i++ {
			dst[i] = float32(float64(row[i]) * rms * float64(weight[i]))
		}
	}
	return out
}

// Add returns a freshly allocated element-wise sum of a and b. Both slices
// must have the same length. Panics on length mismatch.
func Add(a, b []float32) []float32 {
	if len(a) != len(b) {
		panic("Add: length mismatch")
	}
	out := make([]float32, len(a))
	copy(out, a)
	addInPlace(out, b)
	return out
}

// EmbeddingLookup gathers rows of an embedding table by id.
//
//	embed: [V, D] flat row-major
//	V, D:  table dimensions
//	ids:   token ids, length T
//
// Returns a freshly allocated [T, D] slice. Panics if any id is out of range.
func EmbeddingLookup(embed []float32, V, D int, ids []int) []float32 {
	if len(embed) != V*D {
		panic("EmbeddingLookup: embed length mismatch")
	}
	T := len(ids)
	out := make([]float32, T*D)
	for t, id := range ids {
		if id < 0 || id >= V {
			panic("EmbeddingLookup: id out of range")
		}
		copy(out[t*D:(t+1)*D], embed[id*D:(id+1)*D])
	}
	return out
}

// YarnParams configures the yarn-scaled RoPE inverse-frequency computation.
// The values used by the privacy-filter model are:
//
//	HeadDim=64, Theta=150000, OriginalMaxPositions=4096,
//	Factor=32, BetaFast=32, BetaSlow=1.
type YarnParams struct {
	HeadDim              int
	Theta                float64
	OriginalMaxPositions int
	Factor               float64
	BetaFast             float64
	BetaSlow             float64
}

// YarnRoPETables returns yarn-scaled cos/sin tables for positions 0..T-1,
// following transformers.modeling_rope_utils._compute_yarn_parameters with
// `truncate=false`. The returned slices are both row-major [T, HeadDim/2].
//
// The "ramp" function blends between:
//   - inv_freq_extrapolation = 1 / theta^(2i/dim)           (for high-freq, small i)
//   - inv_freq_interpolation = 1 / (factor * theta^(2i/dim)) (for low-freq, large i)
//
// A yarn attention scaling factor `mscale = 0.1*ln(factor) + 1` is applied to
// both cos and sin.
func YarnRoPETables(T int, p YarnParams) (cos, sin []float32) {
	if p.HeadDim <= 0 || p.HeadDim%2 != 0 {
		panic("YarnRoPETables: HeadDim must be positive and even")
	}
	dim := p.HeadDim
	half := dim / 2
	base := p.Theta
	factor := p.Factor
	maxPos := float64(p.OriginalMaxPositions)

	// pos_freqs[i] = base ** (2i / dim) for i in 0..half-1.
	posFreqs := make([]float64, half)
	invExtrap := make([]float64, half)
	invInterp := make([]float64, half)
	for i := 0; i < half; i++ {
		pf := math.Pow(base, float64(2*i)/float64(dim))
		posFreqs[i] = pf
		invExtrap[i] = 1.0 / pf
		invInterp[i] = 1.0 / (factor * pf)
	}

	// Correction range (`truncate=false`, matching the model config).
	// low = max(findCorrDim(beta_fast), 0)
	// high = min(findCorrDim(beta_slow), dim-1)
	findCorr := func(numRot float64) float64 {
		return float64(dim) * math.Log(maxPos/(numRot*2*math.Pi)) / (2 * math.Log(base))
	}
	low := findCorr(p.BetaFast)
	high := findCorr(p.BetaSlow)
	if low < 0 {
		low = 0
	}
	if high > float64(dim-1) {
		high = float64(dim - 1)
	}
	if low == high {
		high = low + 0.001
	}

	// Linear ramp over i=0..half-1 (note: dim//2, not dim, per the reference).
	// ramp(i) = clip((i - low) / (high - low), 0, 1)
	// extrapFactor(i) = 1 - ramp(i) -- 1 at low i (extrapolate), 0 at high i
	// inv_freq(i) = interp * (1 - extrapFactor) + extrap * extrapFactor
	invFreq := make([]float64, half)
	for i := 0; i < half; i++ {
		ramp := (float64(i) - low) / (high - low)
		if ramp < 0 {
			ramp = 0
		} else if ramp > 1 {
			ramp = 1
		}
		extrapFactor := 1 - ramp
		invFreq[i] = invInterp[i]*(1-extrapFactor) + invExtrap[i]*extrapFactor
	}

	// Yarn attention scaling (applied to cos and sin).
	// attention_factor = 0.1 * ln(factor) + 1, assuming factor > 1.
	mscale := 1.0
	if factor > 1 {
		mscale = 0.1*math.Log(factor) + 1.0
	}

	cos = make([]float32, T*half)
	sin = make([]float32, T*half)
	for pos := 0; pos < T; pos++ {
		row := pos * half
		fpos := float64(pos)
		for i := 0; i < half; i++ {
			a := fpos * invFreq[i]
			cos[row+i] = float32(math.Cos(a) * mscale)
			sin[row+i] = float32(math.Sin(a) * mscale)
		}
	}
	return cos, sin
}
