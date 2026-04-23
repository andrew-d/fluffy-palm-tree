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

	// Each output column y[:, o] is written by only one worker, so no
	// synchronization is needed. Parallelize across out so each worker
	// streams out/nWorkers weight rows (~2.3 MB / 8 ≈ 290 KB for the Q
	// projection) instead of the full W. The shared x (780 KB for
	// T=306, in=640) stays hot in L3 across workers.
	nWorkers := runtime.GOMAXPROCS(0)
	if nWorkers > out {
		nWorkers = out
	}

	processChunk := func(oStart, oEnd int) {
		// Inner loop uses a 4×4 C-tile (16 Float32x16 accumulators, K-inner
		// reduction) so each x/w load feeds 4 FMAs. Falls through to
		// dotBatch8 then scalar dot for non-multiple-of-4 tails.
		o := oStart
		for ; o+4 <= oEnd; o += 4 {
			t := 0
			for ; t+4 <= T; t += 4 {
				linearTile4x4(x, W, y, in, out, t, o, b)
			}
			// Token tail: fill remaining t with single-token dots.
			for ; t < T; t++ {
				for oo := o; oo < o+4; oo++ {
					var bias float32
					if b != nil {
						bias = b[oo]
					}
					y[t*out+oo] = dot(x[t*in:(t+1)*in], W[oo*in:(oo+1)*in]) + bias
				}
			}
		}
		// Output tail (oEnd - o < 4): original per-o scheme.
		for ; o < oEnd; o++ {
			wrow := W[o*in : (o+1)*in]
			var bias float32
			if b != nil {
				bias = b[o]
			}
			t := 0
			for ; t+8 <= T; t += 8 {
				var xs [8][]float32
				for s := 0; s < 8; s++ {
					ts := t + s
					xs[s] = x[ts*in : (ts+1)*in]
				}
				r := dotBatch8(wrow, xs)
				for s := 0; s < 8; s++ {
					y[(t+s)*out+o] = r[s] + bias
				}
			}
			for ; t < T; t++ {
				y[t*out+o] = dot(x[t*in:(t+1)*in], wrow) + bias
			}
		}
	}

	if nWorkers <= 1 {
		processChunk(0, out)
		return y
	}

	var wg sync.WaitGroup
	chunk := (out + nWorkers - 1) / nWorkers
	for w := 0; w < nWorkers; w++ {
		start := w * chunk
		if start >= out {
			break
		}
		end := start + chunk
		if end > out {
			end = out
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			processChunk(start, end)
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
			maxLogit := float32(math.Inf(-1))
			for j := jLo; j < jHi; j++ {
				kOff := (j*numKV + kvIdx) * headDim
				s := dot(qRow, k[kOff:kOff+headDim])
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

			// Softmax in fp32 over [jLo, jHi). The sink column is included
			// in the denominator for "lost mass" semantics but never
			// multiplied back into attnOut.
			sum := softmaxExpSum(scores, jLo, jHi, maxLogit)
			sum += float32(math.Exp(float64(sinkLogit - maxLogit)))
			softmaxScale(scores, jLo, jHi, 1.0/sum)

			// attnOut[h, i, :] = sum_{j in [jLo,jHi)} scores[j] * v[j, kvIdx, :]
			// Batch 16 j's per axpyBatch16 call: one shared dst, 16
			// different (alpha, vRow) pairs per call, replacing 16 axpy
			// calls with 1. Saves 15/16 of the dst load/store traffic.
			dstOff := (h*T + i) * headDim
			dst := attnOut[dstOff : dstOff+headDim]
			for d2 := 0; d2 < headDim; d2++ {
				dst[d2] = 0
			}
			var alphasScratch [16]float32
			var alphas [16][]float32
			var vRows [16][]float32
			j := jLo
			for ; j+16 <= jHi; j += 16 {
				for s := 0; s < 16; s++ {
					alphasScratch[s] = scores[j+s]
					alphas[s] = alphasScratch[s : s+1]
					vOff := ((j + s)*numKV + kvIdx) * headDim
					vRows[s] = v[vOff : vOff+headDim]
				}
				axpyBatch16(alphas, vRows, dst, 0)
			}
			for ; j < jHi; j++ {
				w := scores[j]
				if w == 0 {
					continue
				}
				vOff := (j*numKV + kvIdx) * headDim
				axpy(w, v[vOff:vOff+headDim], dst)
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
