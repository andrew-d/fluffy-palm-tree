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
