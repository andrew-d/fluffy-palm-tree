package nn

import (
	"path/filepath"
	"testing"

	"github.com/andrew-d/openai-privacy/internal/safetensors"
)

// TestLinearSmoke exercises the basic Linear primitive with a tiny
// hand-checked example.
//
//	x = [[1, 2, 3],
//	     [4, 5, 6]]                 shape [T=2, in=3]
//	W = [[1, 0, -1],
//	     [2, 1,  0]]                shape [out=2, in=3]
//	b = [10, -5]                    shape [out=2]
//	y[t, o] = sum_i x[t,i] * W[o,i] + b[o]
//	       -> y = [[1*1 + 2*0 + 3*-1 + 10, 1*2 + 2*1 + 3*0 + -5],
//	               [4*1 + 5*0 + 6*-1 + 10, 4*2 + 5*1 + 6*0 + -5]]
//	            = [[ 8, -1],
//	               [ 8,  8]]
func TestLinearSmoke(t *testing.T) {
	x := []float32{1, 2, 3, 4, 5, 6}
	w := []float32{1, 0, -1, 2, 1, 0}
	b := []float32{10, -5}
	got := Linear(x, w, b, 2, 3, 2)
	want := []float32{8, -1, 8, 8}
	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		d := got[i] - want[i]
		if d < 0 {
			d = -d
		}
		if d > 1e-6 {
			t.Errorf("idx %d: got %v, want %v", i, got[i], want[i])
		}
	}

	// Also sanity-check nil bias.
	got2 := Linear(x, w, nil, 2, 3, 2)
	want2 := []float32{want[0] - 10, want[1] - -5, want[2] - 10, want[3] - -5}
	for i := range want2 {
		d := got2[i] - want2[i]
		if d < 0 {
			d = -d
		}
		if d > 1e-6 {
			t.Errorf("nil-bias idx %d: got %v, want %v", i, got2[i], want2[i])
		}
	}
}

// TestApplyRoPESmoke applies RoPE with cos=1 / sin=0; both q and k must
// come back unchanged (the rotation is the identity).
func TestApplyRoPESmoke(t *testing.T) {
	const T = 3
	const numQ = 2
	const numKV = 1
	const headDim = 4
	const half = headDim / 2

	// Fill q and k with recognisable values.
	q := make([]float32, T*numQ*headDim)
	k := make([]float32, T*numKV*headDim)
	for i := range q {
		q[i] = float32(i) + 0.5
	}
	for i := range k {
		k[i] = float32(-i) - 0.25
	}
	origQ := append([]float32(nil), q...)
	origK := append([]float32(nil), k...)

	cos := make([]float32, T*half)
	sin := make([]float32, T*half)
	for i := range cos {
		cos[i] = 1
		sin[i] = 0
	}

	ApplyRoPE(q, k, cos, sin, T, numQ, numKV, headDim)

	for i := range q {
		if q[i] != origQ[i] {
			t.Fatalf("q[%d] changed: got %v, want %v", i, q[i], origQ[i])
		}
	}
	for i := range k {
		if k[i] != origK[i] {
			t.Fatalf("k[%d] changed: got %v, want %v", i, k[i], origK[i])
		}
	}
}

// TestLayer0AttentionAgainstFixture exercises GQAAttentionWithSinks end-to-end
// against reference outputs dumped from the Python reference implementation.
func TestLayer0AttentionAgainstFixture(t *testing.T) {
	meta, root := loadFixtureMeta(t)
	T := meta.SeqLen
	if T == 0 {
		t.Fatal("fixture seq_len is zero")
	}
	const (
		hidden  = 640
		headDim = 64
		numQ    = 14
		numKV   = 2
		window  = 129
	)

	stPath := filepath.Join(root, "model", "model.safetensors")
	r, err := safetensors.Open(stPath)
	if err != nil {
		t.Fatalf("open safetensors: %v", err)
	}
	defer r.Close()

	load := func(name string) []float32 {
		tn, err := r.Tensor(name)
		if err != nil {
			t.Fatalf("get %s: %v", name, err)
		}
		return tn.Float32s()
	}

	wq := load("model.layers.0.self_attn.q_proj.weight")
	bq := load("model.layers.0.self_attn.q_proj.bias")
	wk := load("model.layers.0.self_attn.k_proj.weight")
	bk := load("model.layers.0.self_attn.k_proj.bias")
	wv := load("model.layers.0.self_attn.v_proj.weight")
	bv := load("model.layers.0.self_attn.v_proj.bias")
	wo := load("model.layers.0.self_attn.o_proj.weight")
	bo := load("model.layers.0.self_attn.o_proj.bias")
	sinks := load("model.layers.0.self_attn.sinks")

	if got, want := len(wq), numQ*headDim*hidden; got != want {
		t.Fatalf("wq len = %d, want %d", got, want)
	}
	if got, want := len(bq), numQ*headDim; got != want {
		t.Fatalf("bq len = %d, want %d", got, want)
	}
	if got, want := len(wk), numKV*headDim*hidden; got != want {
		t.Fatalf("wk len = %d, want %d", got, want)
	}
	if got, want := len(wo), hidden*numQ*headDim; got != want {
		t.Fatalf("wo len = %d, want %d", got, want)
	}
	if got, want := len(sinks), numQ; got != want {
		t.Fatalf("sinks len = %d, want %d", got, want)
	}

	// Pre-attention normalized input.
	x := loadF32(t, filepath.Join(root, "fixtures", "layer0_pre_attn_norm.f32.bin"))
	if len(x) != T*hidden {
		t.Fatalf("pre-attn-norm len = %d, want %d", len(x), T*hidden)
	}

	cos, sin := YarnRoPETables(T, YarnParams{
		HeadDim:              headDim,
		Theta:                150000,
		OriginalMaxPositions: 4096,
		Factor:               32,
		BetaFast:             32,
		BetaSlow:             1,
	})

	got := GQAAttentionWithSinks(x, wq, bq, wk, bk, wv, bv, wo, bo, sinks, cos, sin,
		T, hidden, headDim, numQ, numKV, window)

	want := loadF32(t, filepath.Join(root, "fixtures", "layer0_attn_out.f32.bin"))
	if len(got) != len(want) {
		t.Fatalf("len mismatch: got %d want %d", len(got), len(want))
	}
	errv, idx := maxAbsErr(got, want)
	if errv >= 1e-2 {
		t.Fatalf("max abs err = %v at idx %d (got %v want %v)", errv, idx, got[idx], want[idx])
	}
	// Verify per-element tolerance (< 1e-3) by walking the arrays.
	var failCount int
	for i := range want {
		d := got[i] - want[i]
		if d < 0 {
			d = -d
		}
		if d >= 1e-3 {
			if failCount < 5 {
				t.Errorf("idx %d: got %v want %v (abs err %v)", i, got[i], want[i], d)
			}
			failCount++
		}
	}
	if failCount > 0 {
		t.Errorf("%d elements exceed 1e-3 tolerance", failCount)
	}
	t.Logf("layer0 attention max abs err = %v at idx %d", errv, idx)
}
