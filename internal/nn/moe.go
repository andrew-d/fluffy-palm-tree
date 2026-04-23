package nn

import (
	"math"
	"runtime"
	"sort"
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

	// Expert-major dispatch. The per-token loop streamed each expert's
	// ~5 MB weight set once per assigned token; batching tokens-per-expert
	// streams it once per expert and reuses it across all N assigned
	// tokens (~10 on average for T=306, topK=4, 128 experts). Profiling
	// showed we were DRAM-bandwidth-bound on Wu/Wd reads — this brings the
	// per-layer weight traffic from ~5 GB down to ~0.6 GB.

	// Gather (tokenIdx, weight) pairs per expert. Two-pass so each
	// perExpert[e] is allocated exactly once at the right capacity and
	// no grow-and-copy happens in the append loop — was responsible for
	// ~1K allocs/forward.
	type pair struct {
		t      int
		weight float32
	}
	counts := make([]int, numExperts)
	for t := 0; t < T; t++ {
		for kPos := 0; kPos < topK; kPos++ {
			e := routerIndices[t*topK+kPos]
			if e >= 0 && e < numExperts {
				counts[e]++
			}
		}
	}
	perExpert := make([][]pair, numExperts)
	for e, c := range counts {
		if c > 0 {
			perExpert[e] = make([]pair, 0, c)
		}
	}
	for t := 0; t < T; t++ {
		for kPos := 0; kPos < topK; kPos++ {
			e := routerIndices[t*topK+kPos]
			if e < 0 || e >= numExperts {
				continue
			}
			perExpert[e] = append(perExpert[e], pair{t, routerScores[t*topK+kPos]})
		}
	}

	nWorkers := runtime.GOMAXPROCS(0)
	if nWorkers > numExperts {
		nWorkers = numExperts
	}
	if nWorkers < 1 {
		nWorkers = 1
	}

	// Each token's accum row can be written by up to topK different experts,
	// each potentially on a different worker. Shard accum per worker and
	// reduce at the end to avoid scatter-add contention.
	shards := make([][]float32, nWorkers)
	for w := range shards {
		shards[w] = make([]float32, T*D)
	}

	processAssigned := func(workerID int, experts []int) {
		shard := shards[workerID]
		// Per-worker batch scratch, grown on demand to the largest N we see.
		var gateUpBatch, gatedBatch, outBatch []float32
		// Scratch for the per-iteration alpha vectors fed to axpyBatch.
		var gateAlphas, downAlphas []float32
		for _, e := range experts {
			pairs := perExpert[e]
			n := len(pairs)
			if n == 0 {
				continue
			}
			need := n * twoI
			if cap(gateUpBatch) < need {
				gateUpBatch = make([]float32, need)
			} else {
				gateUpBatch = gateUpBatch[:need]
			}
			need = n * I
			if cap(gatedBatch) < need {
				gatedBatch = make([]float32, need)
			} else {
				gatedBatch = gatedBatch[:need]
			}
			need = n * D
			if cap(outBatch) < need {
				outBatch = make([]float32, need)
			} else {
				outBatch = outBatch[:need]
			}
			if cap(gateAlphas) < n {
				gateAlphas = make([]float32, n)
			} else {
				gateAlphas = gateAlphas[:n]
			}
			if cap(downAlphas) < n {
				downAlphas = make([]float32, n)
			} else {
				downAlphas = downAlphas[:n]
			}

			projBase := e * expertStride
			biasBase := e * twoI
			downBase := e * downStride
			dbBase := e * D

			// Seed each row of gateUpBatch with the expert's bias.
			for k := 0; k < n; k++ {
				copy(gateUpBatch[k*twoI:(k+1)*twoI],
					gateUpBias[biasBase:biasBase+twoI])
			}

			// Batched gate-up: each d loads wRow once and fans out over the
			// N assigned tokens via axpyBatch (saves N function calls per d
			// and hoists the wRow load to the outer iteration).
			for d := 0; d < D; d++ {
				wRow := gateUpProj[projBase+d*twoI : projBase+(d+1)*twoI]
				for k, p := range pairs {
					gateAlphas[k] = hidden[p.t*D+d]
				}
				axpyBatch(gateAlphas, wRow, gateUpBatch, twoI)
			}

			// Activation per batched token.
			for k := 0; k < n; k++ {
				moeActivation(
					gateUpBatch[k*twoI:(k+1)*twoI],
					gatedBatch[k*I:(k+1)*I],
					I, limit, alpha,
				)
			}

			// Seed outBatch with downBias, then batched down matmul.
			for k := 0; k < n; k++ {
				copy(outBatch[k*D:(k+1)*D], downBias[dbBase:dbBase+D])
			}
			for i := 0; i < I; i++ {
				wRow := downProj[downBase+i*D : downBase+(i+1)*D]
				for k := 0; k < n; k++ {
					downAlphas[k] = gatedBatch[k*I+i]
				}
				axpyBatch(downAlphas, wRow, outBatch, D)
			}

			// Scatter-add weighted outputs into this worker's shard.
			for k, p := range pairs {
				axpy(p.weight, outBatch[k*D:(k+1)*D],
					shard[p.t*D:(p.t+1)*D])
			}
		}
	}

	if nWorkers == 1 {
		assignments := make([][]int, 1)
		assignments[0] = make([]int, 0, numExperts)
		for e := 0; e < numExperts; e++ {
			if counts[e] > 0 {
				assignments[0] = append(assignments[0], e)
			}
		}
		processAssigned(0, assignments[0])
	} else {
		// Sort experts by assigned-token count (desc) and round-robin
		// them across workers. Prevents one worker from getting all the
		// heavy experts while another gets empty ones — the naive
		// contiguous-slab split happened to leave the last worker idle on
		// skewed router distributions.
		order := make([]int, 0, numExperts)
		for e := 0; e < numExperts; e++ {
			if counts[e] > 0 {
				order = append(order, e)
			}
		}
		sort.Slice(order, func(i, j int) bool {
			return counts[order[i]] > counts[order[j]]
		})
		assignments := make([][]int, nWorkers)
		for i, e := range order {
			w := i % nWorkers
			assignments[w] = append(assignments[w], e)
		}
		var wg sync.WaitGroup
		for w := 0; w < nWorkers; w++ {
			if len(assignments[w]) == 0 {
				continue
			}
			wg.Add(1)
			go func(workerID int, experts []int) {
				defer wg.Done()
				processAssigned(workerID, experts)
			}(w, assignments[w])
		}
		wg.Wait()
	}

	// Reduce worker shards into accum.
	for _, shard := range shards {
		for i := range accum {
			accum[i] += shard[i]
		}
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
