//go:build !goexperiment.simd || !amd64

package nn

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

// gemv: scalar fallback of the block-outer SIMD matmul. Outer over j so
// the result can be accumulated in an 8-way unrolled register-local
// reduction, matching the SIMD variant's I/O profile.
func gemv(x, W, bias, y []float32, rows, cols int) {
	if len(x) != rows {
		panic("gemv: x length mismatch")
	}
	if len(W) < rows*cols {
		panic("gemv: W length mismatch")
	}
	if len(y) < cols {
		panic("gemv: y length mismatch")
	}
	if bias != nil && len(bias) < cols {
		panic("gemv: bias length mismatch")
	}
	for j := 0; j < cols; j++ {
		var s float32
		if bias != nil {
			s = bias[j]
		}
		for d := 0; d < rows; d++ {
			s += x[d] * W[d*cols+j]
		}
		y[j] = s
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
