package nn

import (
	"math"
	"runtime"
	"sync"
)

// TopKRouter picks the top-k experts for each token and returns normalized
// scores (already divided by topK to exactly match the Python reference,
// OpenAIPrivacyFilterTopKRouter.forward):
//
//	router_logits = linear(hidden, wRouter, bRouter)        # [T, numExperts] fp32
//	top_vals, top_idx = topk(router_logits, topK)           # [T, topK]
//	scores = softmax(top_vals, dim=-1)                      # [T, topK] fp32
//	scores = scores / topK                                  # <-- still divided here
//
//	hidden:     [T, D]
//	wRouter:    [numExperts, D]
//	bRouter:    [numExperts] or nil
//	topK:       number of experts picked per token (e.g. 4)
//	numExperts: total expert count (e.g. 128)
//
// Returns:
//
//	scores:  [T, topK] fp32 — softmax(top-k logits) / topK
//	indices: [T, topK] int  — expert indices in [0, numExperts)
//
// All arithmetic is done in fp32. Ties in top-k select the earlier index,
// matching torch.topk's deterministic behaviour on CPU.
func TopKRouter(hidden, wRouter, bRouter []float32, T, D, numExperts, topK int) (scores []float32, indices []int) {
	if topK <= 0 || topK > numExperts {
		panic("TopKRouter: invalid topK")
	}
	if len(hidden) != T*D {
		panic("TopKRouter: hidden length mismatch")
	}
	if len(wRouter) != numExperts*D {
		panic("TopKRouter: wRouter length mismatch")
	}
	if bRouter != nil && len(bRouter) != numExperts {
		panic("TopKRouter: bRouter length mismatch")
	}

	// Compute all router logits in fp32. `Linear` already operates in fp32.
	logits := Linear(hidden, wRouter, bRouter, T, D, numExperts)

	scores = make([]float32, T*topK)
	indices = make([]int, T*topK)

	// Scratch for each row.
	topVals := make([]float32, topK)
	topIdx := make([]int, topK)

	for t := 0; t < T; t++ {
		row := logits[t*numExperts : (t+1)*numExperts]

		// Initialize top-k with -inf.
		for j := 0; j < topK; j++ {
			topVals[j] = float32(math.Inf(-1))
			topIdx[j] = -1
		}

		// Find top-k greatest values, preferring the smaller index on ties
		// (matches torch.topk with sorted=True on CPU, which is stable in
		// practice for our fixture).
		for e := 0; e < numExperts; e++ {
			v := row[e]
			// If v is strictly greater than the smallest kept value, insert.
			// topVals is maintained in descending order.
			if v > topVals[topK-1] {
				// Find insertion point from the end.
				pos := topK - 1
				for pos > 0 && v > topVals[pos-1] {
					topVals[pos] = topVals[pos-1]
					topIdx[pos] = topIdx[pos-1]
					pos--
				}
				topVals[pos] = v
				topIdx[pos] = e
			}
		}

		// Softmax over the topK values, in fp32 (but fp64 reduction for the
		// denominator to minimize rounding — the single-precision result is
		// then cast back).
		maxVal := topVals[0] // topVals is sorted descending
		var denom float64
		for j := 0; j < topK; j++ {
			denom += math.Exp(float64(topVals[j] - maxVal))
		}
		inv := 1.0 / denom
		invTopK := 1.0 / float64(topK)
		for j := 0; j < topK; j++ {
			e := math.Exp(float64(topVals[j] - maxVal))
			// Divide by topK right here, matching the Python reference.
			scores[t*topK+j] = float32(e * inv * invTopK)
			indices[t*topK+j] = topIdx[j]
		}
	}
	return scores, indices
}

// MoEExperts runs the mixture-of-experts block for a single layer:
//
//	For each (t, k_pos) pair with expert e = routerIndices[t, k_pos]:
//	  state    = hidden[t]                                 # [D]
//	  gate_up  = state @ gateUpProj[e] + gateUpBias[e]     # [2*I]
//	  gate, up = chunk(gate_up, 2, dim=-1)                 # each [I]
//	  gate     = min(gate, limit)
//	  up       = clamp(up, -limit, +limit)
//	  glu      = gate * sigmoid(gate * alpha)
//	  gated    = (up + 1) * glu
//	  out      = gated @ downProj[e] + downBias[e]         # [D]
//	  accum[t] += out * routerScores[t, k_pos]
//
//	After summing, accum *= topK (matches OpenAIPrivacyFilterMLP.forward,
//	which multiplies by `self.num_experts` == `num_experts_per_tok`).
//
// The routerScores fed in here are expected to be already divided by topK
// (as returned by TopKRouter above), so the net factor on each expert output
// is `softmax_score / topK * topK = softmax_score`, but we keep the division
// and multiplication explicit to reproduce the fp32 round-off of the
// reference implementation.
//
// Shapes:
//
//	hidden:        [T, D] fp32
//	gateUpProj:    [numExperts, D, 2*I]   — row-major, gate cols then up cols
//	gateUpBias:    [numExperts, 2*I]
//	downProj:      [numExperts, I, D]
//	downBias:      [numExperts, D]
//	routerScores:  [T, topK] (from TopKRouter, already divided by topK)
//	routerIndices: [T, topK]
//
// Returns: [T, D] fp32.
func MoEExperts(
	hidden []float32,
	gateUpProj, gateUpBias, downProj, downBias []float32,
	routerScores []float32, routerIndices []int,
	T, D, I, numExperts, topK int,
	limit, alpha float32,
) []float32 {
	if len(hidden) != T*D {
		panic("MoEExperts: hidden length mismatch")
	}
	if len(gateUpProj) != numExperts*D*(2*I) {
		panic("MoEExperts: gateUpProj length mismatch")
	}
	if len(gateUpBias) != numExperts*(2*I) {
		panic("MoEExperts: gateUpBias length mismatch")
	}
	if len(downProj) != numExperts*I*D {
		panic("MoEExperts: downProj length mismatch")
	}
	if len(downBias) != numExperts*D {
		panic("MoEExperts: downBias length mismatch")
	}
	if len(routerScores) != T*topK {
		panic("MoEExperts: routerScores length mismatch")
	}
	if len(routerIndices) != T*topK {
		panic("MoEExperts: routerIndices length mismatch")
	}

	accum := make([]float32, T*D)

	twoI := 2 * I
	expertStride := D * twoI
	downStride := I * D

	// Parallelize across tokens. Each token's accumulator row accum[t*D:(t+1)*D]
	// is written by only one worker, so no locking is needed. The gateUp/gated
	// scratch buffers are per-worker.
	nWorkers := runtime.GOMAXPROCS(0)
	if nWorkers > T {
		nWorkers = T
	}
	if nWorkers < 1 {
		nWorkers = 1
	}

	processToken := func(t int, gateUp, gated []float32) {
		state := hidden[t*D : (t+1)*D]
		accumRow := accum[t*D : (t+1)*D]
		for kPos := 0; kPos < topK; kPos++ {
			e := routerIndices[t*topK+kPos]
			if e < 0 || e >= numExperts {
				// Skip masked / invalid indices (reference code also skips
				// expert_idx == num_experts used as a mask class).
				continue
			}
			weight := routerScores[t*topK+kPos]

			// gate_up = state @ gate_up_proj[e] + gate_up_bias[e]   # [2*I]
			projBase := e * expertStride
			biasBase := e * twoI
			copy(gateUp, gateUpBias[biasBase:biasBase+twoI])
			for d := 0; d < D; d++ {
				s := state[d]
				if s == 0 {
					continue
				}
				rowBase := projBase + d*twoI
				row := gateUpProj[rowBase : rowBase+twoI]
				// Unroll by 8 so the compiler can emit independent FMA chains
				// and more easily vectorize (AVX2 = 8 fp32 per op). twoI is
				// 1280 on the current model (divisible by 8); the tail loop
				// handles any residual for other shapes.
				j := 0
				for ; j <= twoI-8; j += 8 {
					gateUp[j] += s * row[j]
					gateUp[j+1] += s * row[j+1]
					gateUp[j+2] += s * row[j+2]
					gateUp[j+3] += s * row[j+3]
					gateUp[j+4] += s * row[j+4]
					gateUp[j+5] += s * row[j+5]
					gateUp[j+6] += s * row[j+6]
					gateUp[j+7] += s * row[j+7]
				}
				for ; j < twoI; j++ {
					gateUp[j] += s * row[j]
				}
			}

			// gate = gateUp[:I], up = gateUp[I:]  (concatenated layout)
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

			// out = gated @ downProj[e] + downBias[e]  # [D]
			downBase := e * downStride
			dbBase := e * D
			for i := 0; i < I; i++ {
				gi := gated[i]
				if gi == 0 {
					continue
				}
				w := weight * gi
				rowBase := downBase + i*D
				row := downProj[rowBase : rowBase+D]
				d := 0
				for ; d <= D-8; d += 8 {
					accumRow[d] += w * row[d]
					accumRow[d+1] += w * row[d+1]
					accumRow[d+2] += w * row[d+2]
					accumRow[d+3] += w * row[d+3]
					accumRow[d+4] += w * row[d+4]
					accumRow[d+5] += w * row[d+5]
					accumRow[d+6] += w * row[d+6]
					accumRow[d+7] += w * row[d+7]
				}
				for ; d < D; d++ {
					accumRow[d] += w * row[d]
				}
			}
			for d := 0; d < D; d++ {
				accumRow[d] += weight * downBias[dbBase+d]
			}
		}
	}

	if nWorkers == 1 {
		gateUp := make([]float32, twoI)
		gated := make([]float32, I)
		for t := 0; t < T; t++ {
			processToken(t, gateUp, gated)
		}
	} else {
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
				gateUp := make([]float32, twoI)
				gated := make([]float32, I)
				for t := start; t < end; t++ {
					processToken(t, gateUp, gated)
				}
			}(start, end)
		}
		wg.Wait()
	}

	// Post-scale by topK, matching OpenAIPrivacyFilterMLP.forward:
	//   hidden_states = hidden_states * self.num_experts  (== topK)
	scale := float32(topK)
	for i := range accum {
		accum[i] *= scale
	}
	return accum
}

// MoELayer is a convenience wrapper that runs the router followed by the
// expert mixture for a single layer.
//
// Returns: [T, D] fp32.
func MoELayer(
	hidden, wRouter, bRouter []float32,
	gateUpProj, gateUpBias, downProj, downBias []float32,
	T, D, I, numExperts, topK int,
	limit, alpha float32,
) []float32 {
	scores, indices := TopKRouter(hidden, wRouter, bRouter, T, D, numExperts, topK)
	return MoEExperts(
		hidden,
		gateUpProj, gateUpBias, downProj, downBias,
		scores, indices,
		T, D, I, numExperts, topK,
		limit, alpha,
	)
}
