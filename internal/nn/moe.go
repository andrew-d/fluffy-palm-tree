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
			if cap(gateAlphas) < 8*n {
				gateAlphas = make([]float32, 8*n)
			} else {
				gateAlphas = gateAlphas[:8*n]
			}
			if cap(downAlphas) < 8*n {
				downAlphas = make([]float32, 8*n)
			} else {
				downAlphas = downAlphas[:8*n]
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

			// Batched gate-up: fuse up to eight d-iterations per gateUp
			// block (one load/store per 8 d-steps). Tail falls through
			// axpyBatch4 → axpyBatch2 → axpyBatch.
			ga0 := gateAlphas[:n]
			ga1 := gateAlphas[n : 2*n]
			ga2 := gateAlphas[2*n : 3*n]
			ga3 := gateAlphas[3*n : 4*n]
			ga4 := gateAlphas[4*n : 5*n]
			ga5 := gateAlphas[5*n : 6*n]
			ga6 := gateAlphas[6*n : 7*n]
			ga7 := gateAlphas[7*n : 8*n]
			gwRow := func(d int) []float32 {
				return gateUpProj[projBase+d*twoI : projBase+(d+1)*twoI]
			}
			d := 0
			for ; d+8 <= D; d += 8 {
				for k, p := range pairs {
					base := p.t * D
					ga0[k] = hidden[base+d]
					ga1[k] = hidden[base+d+1]
					ga2[k] = hidden[base+d+2]
					ga3[k] = hidden[base+d+3]
					ga4[k] = hidden[base+d+4]
					ga5[k] = hidden[base+d+5]
					ga6[k] = hidden[base+d+6]
					ga7[k] = hidden[base+d+7]
				}
				axpyBatch8(ga0, ga1, ga2, ga3, ga4, ga5, ga6, ga7,
					gwRow(d), gwRow(d+1), gwRow(d+2), gwRow(d+3),
					gwRow(d+4), gwRow(d+5), gwRow(d+6), gwRow(d+7),
					gateUpBatch, twoI)
			}
			for ; d+4 <= D; d += 4 {
				for k, p := range pairs {
					base := p.t * D
					ga0[k] = hidden[base+d]
					ga1[k] = hidden[base+d+1]
					ga2[k] = hidden[base+d+2]
					ga3[k] = hidden[base+d+3]
				}
				axpyBatch4(ga0, ga1, ga2, ga3,
					gwRow(d), gwRow(d+1), gwRow(d+2), gwRow(d+3),
					gateUpBatch, twoI)
			}
			for ; d+2 <= D; d += 2 {
				for k, p := range pairs {
					ga0[k] = hidden[p.t*D+d]
					ga1[k] = hidden[p.t*D+d+1]
				}
				axpyBatch2(ga0, ga1, gwRow(d), gwRow(d+1), gateUpBatch, twoI)
			}
			for ; d < D; d++ {
				for k, p := range pairs {
					ga0[k] = hidden[p.t*D+d]
				}
				axpyBatch(ga0, gwRow(d), gateUpBatch, twoI)
			}

			// Activation per batched token.
			for k := 0; k < n; k++ {
				moeActivation(
					gateUpBatch[k*twoI:(k+1)*twoI],
					gatedBatch[k*I:(k+1)*I],
					I, limit, alpha,
				)
			}

			// Seed outBatch with downBias, then batched down matmul with
			// the same pair-d fusion.
			for k := 0; k < n; k++ {
				copy(outBatch[k*D:(k+1)*D], downBias[dbBase:dbBase+D])
			}
			da0 := downAlphas[:n]
			da1 := downAlphas[n : 2*n]
			da2 := downAlphas[2*n : 3*n]
			da3 := downAlphas[3*n : 4*n]
			da4 := downAlphas[4*n : 5*n]
			da5 := downAlphas[5*n : 6*n]
			da6 := downAlphas[6*n : 7*n]
			da7 := downAlphas[7*n : 8*n]
			dwRow := func(i int) []float32 {
				return downProj[downBase+i*D : downBase+(i+1)*D]
			}
			i := 0
			for ; i+8 <= I; i += 8 {
				for k := 0; k < n; k++ {
					base := k * I
					da0[k] = gatedBatch[base+i]
					da1[k] = gatedBatch[base+i+1]
					da2[k] = gatedBatch[base+i+2]
					da3[k] = gatedBatch[base+i+3]
					da4[k] = gatedBatch[base+i+4]
					da5[k] = gatedBatch[base+i+5]
					da6[k] = gatedBatch[base+i+6]
					da7[k] = gatedBatch[base+i+7]
				}
				axpyBatch8(da0, da1, da2, da3, da4, da5, da6, da7,
					dwRow(i), dwRow(i+1), dwRow(i+2), dwRow(i+3),
					dwRow(i+4), dwRow(i+5), dwRow(i+6), dwRow(i+7),
					outBatch, D)
			}
			for ; i+4 <= I; i += 4 {
				for k := 0; k < n; k++ {
					base := k * I
					da0[k] = gatedBatch[base+i]
					da1[k] = gatedBatch[base+i+1]
					da2[k] = gatedBatch[base+i+2]
					da3[k] = gatedBatch[base+i+3]
				}
				axpyBatch4(da0, da1, da2, da3,
					dwRow(i), dwRow(i+1), dwRow(i+2), dwRow(i+3),
					outBatch, D)
			}
			for ; i+2 <= I; i += 2 {
				for k := 0; k < n; k++ {
					da0[k] = gatedBatch[k*I+i]
					da1[k] = gatedBatch[k*I+i+1]
				}
				axpyBatch2(da0, da1, dwRow(i), dwRow(i+1), outBatch, D)
			}
			for ; i < I; i++ {
				for k := 0; k < n; k++ {
					da0[k] = gatedBatch[k*I+i]
				}
				axpyBatch(da0, dwRow(i), outBatch, D)
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

	// Reduce worker shards into accum and apply the topK post-scale in a
	// single fan-out so the output buffer is touched exactly once.
	scale := float32(topK)
	reduceChunk := (T*D + nWorkers - 1) / nWorkers
	var rwg sync.WaitGroup
	for w := 0; w < nWorkers; w++ {
		start := w * reduceChunk
		if start >= T*D {
			break
		}
		end := start + reduceChunk
		if end > T*D {
			end = T*D
		}
		rwg.Add(1)
		go func(start, end int) {
			defer rwg.Done()
			for i := start; i < end; i++ {
				var s float32
				for _, shard := range shards {
					s += shard[i]
				}
				accum[i] = s * scale
			}
		}(start, end)
	}
	rwg.Wait()
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
