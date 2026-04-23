package nn

import (
	"math"
	"runtime"
	"sync"
)

// Linear applies y[t] = W @ x[t] + b per token, where W is stored row-major
// with shape [out, in] (the HuggingFace nn.Linear.weight convention) and b has
// shape [out]. Pass b == nil to skip the bias term.
//
//	x:      [T, in]  row-major
//	W:      [out, in] row-major
//	b:      [out] or nil
//	Returns [T, out] row-major.
func Linear(x, W, b []float32, T, in, out int) []float32 {
	if len(x) != T*in {
		panic("Linear: x length mismatch")
	}
	if len(W) != out*in {
		panic("Linear: W length mismatch")
	}
	if b != nil && len(b) != out {
		panic("Linear: b length mismatch")
	}
	y := make([]float32, T*out)

	processRow := func(t int) {
		xrow := x[t*in : (t+1)*in]
		yrow := y[t*out : (t+1)*out]
		for o := 0; o < out; o++ {
			wrow := W[o*in : (o+1)*in]
			s := dot(xrow, wrow)
			if b != nil {
				s += b[o]
			}
			yrow[o] = s
		}
	}

	// Each output row y[t*out:(t+1)*out] is written by only one worker, so no
	// synchronization is needed. Linear is on the critical path for Q/K/V +
	// output projections (x per layer) and the router/classifier head, all of
	// which run serially with respect to MoEExperts, so we have full cores
	// available here.
	nWorkers := runtime.GOMAXPROCS(0)
	if nWorkers > T {
		nWorkers = T
	}
	if nWorkers <= 1 {
		for t := 0; t < T; t++ {
			processRow(t)
		}
		return y
	}

	var wg sync.WaitGroup
	chunk := (T + nWorkers - 1) / nWorkers
	for w := 0; w < nWorkers; w++ {
		start := w * chunk
		if start >= T {
			break
		}
		end := start + chunk
		if end > T {
			end = T
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for t := start; t < end; t++ {
				processRow(t)
			}
		}(start, end)
	}
	wg.Wait()
	return y
}

// ApplyRoPE applies rotary positional embeddings in-place to q and k using
// the openai_privacy_filter "interleaving" layout (as implemented by
// `_apply_rotary_emb` in transformers.models.openai_privacy_filter):
//
//	first_half  = x[..., 0::2]
//	second_half = x[..., 1::2]
//	first_out   = first_half  * cos - second_half * sin
//	second_out  = second_half * cos + first_half  * sin
//	out         = stack([first_out, second_out], dim=-1).flatten(-2)
//
// Concretely the output element at index 2i is `first_out[i]` and the element
// at index 2i+1 is `second_out[i]`.
//
// q: [T, numQ, headDim] row-major; k: [T, numKV, headDim] row-major.
// cos, sin: [T, headDim/2] row-major. With cos=1, sin=0 the rotation is the
// identity (required by the smoke test).
func ApplyRoPE(q, k, cos, sin []float32, T, numQ, numKV, headDim int) {
	if headDim <= 0 || headDim%2 != 0 {
		panic("ApplyRoPE: headDim must be positive and even")
	}
	half := headDim / 2
	if len(q) != T*numQ*headDim {
		panic("ApplyRoPE: q length mismatch")
	}
	if len(k) != T*numKV*headDim {
		panic("ApplyRoPE: k length mismatch")
	}
	if len(cos) != T*half || len(sin) != T*half {
		panic("ApplyRoPE: cos/sin length mismatch")
	}

	rotate := func(buf []float32, heads int) {
		for t := 0; t < T; t++ {
			cr := cos[t*half : (t+1)*half]
			sr := sin[t*half : (t+1)*half]
			base := t * heads * headDim
			for h := 0; h < heads; h++ {
				off := base + h*headDim
				for i := 0; i < half; i++ {
					a := buf[off+2*i]   // first_half[i]
					b := buf[off+2*i+1] // second_half[i]
					c := cr[i]
					s := sr[i]
					buf[off+2*i] = a*c - b*s
					buf[off+2*i+1] = b*c + a*s
				}
			}
		}
	}
	rotate(q, numQ)
	rotate(k, numKV)
}

// GQAAttentionWithSinks computes a single grouped-query attention block with
// attention sinks and a symmetric bidirectional sliding-window mask. See the
// docstring in the accompanying test file (and the architecture notes) for the
// exact reference semantics.
//
//	x         : [T, hidden]
//	wq/bq     : [numQ*headDim, hidden] / [numQ*headDim]
//	wk/bk     : [numKV*headDim, hidden] / [numKV*headDim]
//	wv/bv     : [numKV*headDim, hidden] / [numKV*headDim]
//	wo/bo     : [hidden, numQ*headDim] / [hidden]
//	sinks     : [numQ] fp32, one learnable logit per query head
//	cos, sin  : [T, headDim/2] yarn-scaled RoPE tables
//	window    : half-width of the bidirectional window, as
//	            `sliding_window + 1`. A pair (i, j) is allowed iff
//	            `|i-j| <= window - 1`.
//
// Returns [T, hidden].
func GQAAttentionWithSinks(
	x []float32,
	wq, bq, wk, bk, wv, bv, wo, bo []float32,
	sinks []float32,
	cos, sin []float32,
	T, hidden, headDim, numQ, numKV, window int,
) []float32 {
	if numKV <= 0 || numQ%numKV != 0 {
		panic("GQAAttentionWithSinks: numQ must be a positive multiple of numKV")
	}
	if len(sinks) != numQ {
		panic("GQAAttentionWithSinks: sinks length mismatch")
	}
	group := numQ / numKV

	qDim := numQ * headDim
	kvDim := numKV * headDim

	// Project.
	//   q: [T, numQ*headDim]  (row-major; head-major within row)
	//   k: [T, numKV*headDim]
	//   v: [T, numKV*headDim]
	q := Linear(x, wq, bq, T, hidden, qDim)
	k := Linear(x, wk, bk, T, hidden, kvDim)
	v := Linear(x, wv, bv, T, hidden, kvDim)

	// Apply RoPE. The projection layout is [T, heads, headDim] with heads
	// packed inside each token row; ApplyRoPE treats it that way.
	ApplyRoPE(q, k, cos, sin, T, numQ, numKV, headDim)

	// Scale Q and K individually by headDim^{-0.25} so that q @ k^T ends up
	// scaled by headDim^{-0.5}.
	scale := float32(math.Pow(float64(headDim), -0.25))
	for i := range q {
		q[i] *= scale
	}
	for i := range k {
		k[i] *= scale
	}

	// Attention per head. We don't materialize a repeated K/V — query head h
	// simply consumes kv head h/group.
	//
	// attnOut[h, t, :] accumulates into a [numQ, T, headDim] buffer, laid out
	// as [numQ, T, headDim] row-major.
	attnOut := make([]float32, numQ*T*headDim)

	// Each head writes to its own [T, headDim] slice of attnOut, so fan-out
	// across heads is lock-free. scores is a per-goroutine scratch buffer.
	//
	// With a symmetric sliding window of radius (window-1), valid keys for
	// query i live in [jLo, jHi). Hoisting those bounds out of the inner
	// loops skips both the per-j abs/mask branch and the IsInf check in the
	// softmax pass, and shrinks the exp/v-accumulate loops from T to ≤2*window.
	processHead := func(h int, scores []float32) {
		kvIdx := h / group
		sinkLogit := sinks[h]
		for i := 0; i < T; i++ {
			// q[i, h, :] at offset (i*numQ + h)*headDim
			qOff := (i*numQ + h) * headDim
			qRow := q[qOff : qOff+headDim]

			jLo := i - window + 1
			if jLo < 0 {
				jLo = 0
			}
			jHi := i + window
			if jHi > T {
				jHi = T
			}

			// Raw logits s[j] = <q_row, k[j, kvIdx, :]>, only for valid j.
			// Eight parallel accumulators — headDim is 64 on the current
			// model (divisible by 8); tail loop handles other shapes.
			maxLogit := float32(math.Inf(-1))
			for j := jLo; j < jHi; j++ {
				kOff := (j*numKV + kvIdx) * headDim
				kRow := k[kOff : kOff+headDim]
				var s0, s1, s2, s3, s4, s5, s6, s7 float32
				d2 := 0
				for ; d2+8 <= headDim; d2 += 8 {
					s0 += qRow[d2] * kRow[d2]
					s1 += qRow[d2+1] * kRow[d2+1]
					s2 += qRow[d2+2] * kRow[d2+2]
					s3 += qRow[d2+3] * kRow[d2+3]
					s4 += qRow[d2+4] * kRow[d2+4]
					s5 += qRow[d2+5] * kRow[d2+5]
					s6 += qRow[d2+6] * kRow[d2+6]
					s7 += qRow[d2+7] * kRow[d2+7]
				}
				s := ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7))
				for ; d2 < headDim; d2++ {
					s += qRow[d2] * kRow[d2]
				}
				scores[j] = s
				if s > maxLogit {
					maxLogit = s
				}
			}

			// The sink "column" is sinks[h] for every (h, i); include it in
			// the max reduction and the softmax denominator.
			if sinkLogit > maxLogit {
				maxLogit = sinkLogit
			}
			// Windowed attention with a sink logit always has at least one
			// finite candidate, so the -inf fallback from the original is
			// only reachable when T==0; defensive check is unnecessary here.

			// Softmax in fp32 over [jLo, jHi); fp64 denom to minimize roundoff.
			var denom float64
			for j := jLo; j < jHi; j++ {
				e := math.Exp(float64(scores[j] - maxLogit))
				scores[j] = float32(e)
				denom += e
			}
			denom += math.Exp(float64(sinkLogit - maxLogit))
			// Normalize — the sink column is intentionally dropped after
			// normalization, leaving it as "lost mass".
			inv := 1.0 / denom
			for j := jLo; j < jHi; j++ {
				scores[j] = float32(float64(scores[j]) * inv)
			}

			// attnOut[h, i, :] = sum_{j in [jLo,jHi)} scores[j] * v[j, kvIdx, :]
			dstOff := (h*T + i) * headDim
			dst := attnOut[dstOff : dstOff+headDim]
			for d2 := 0; d2 < headDim; d2++ {
				dst[d2] = 0
			}
			for j := jLo; j < jHi; j++ {
				w := scores[j]
				if w == 0 {
					continue
				}
				vOff := (j*numKV + kvIdx) * headDim
				vRow := v[vOff : vOff+headDim]
				d2 := 0
				for ; d2+8 <= headDim; d2 += 8 {
					dst[d2] += w * vRow[d2]
					dst[d2+1] += w * vRow[d2+1]
					dst[d2+2] += w * vRow[d2+2]
					dst[d2+3] += w * vRow[d2+3]
					dst[d2+4] += w * vRow[d2+4]
					dst[d2+5] += w * vRow[d2+5]
					dst[d2+6] += w * vRow[d2+6]
					dst[d2+7] += w * vRow[d2+7]
				}
				for ; d2 < headDim; d2++ {
					dst[d2] += w * vRow[d2]
				}
			}
		}
	}

	nWorkers := runtime.GOMAXPROCS(0)
	if nWorkers > numQ {
		nWorkers = numQ
	}
	if nWorkers <= 1 {
		scores := make([]float32, T)
		for h := 0; h < numQ; h++ {
			processHead(h, scores)
		}
	} else {
		var wg sync.WaitGroup
		chunk := (numQ + nWorkers - 1) / nWorkers
		for wk := 0; wk < nWorkers; wk++ {
			start := wk * chunk
			if start >= numQ {
				break
			}
			end := start + chunk
			if end > numQ {
				end = numQ
			}
			wg.Add(1)
			go func(start, end int) {
				defer wg.Done()
				scores := make([]float32, T)
				for h := start; h < end; h++ {
					processHead(h, scores)
				}
			}(start, end)
		}
		wg.Wait()
	}

	// Transpose from [numQ, T, headDim] back to [T, numQ*headDim] row-major
	// and feed through the output projection.
	concat := make([]float32, T*qDim)
	for h := 0; h < numQ; h++ {
		for i := 0; i < T; i++ {
			src := attnOut[(h*T+i)*headDim : (h*T+i+1)*headDim]
			dst := concat[i*qDim+h*headDim : i*qDim+(h+1)*headDim]
			copy(dst, src)
		}
	}
	return Linear(concat, wo, bo, T, qDim, hidden)
}
