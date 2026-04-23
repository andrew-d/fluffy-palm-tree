package nn

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/andrew-d/openai-privacy/internal/safetensors"
)

// repoRoot returns the absolute path to the repository root, located by
// walking up from the test file until we find go.mod.
func repoRoot(t *testing.T) string {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	dir := filepath.Dir(file)
	for i := 0; i < 10; i++ {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	t.Fatal("could not locate repo root (no go.mod ancestor)")
	return ""
}

// loadF32 reads a little-endian float32 binary file produced by the fixture
// dump scripts. The caller is responsible for knowing the expected length.
func loadF32(t *testing.T, path string) []float32 {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	if len(b)%4 != 0 {
		t.Fatalf("%s: size %d not a multiple of 4", path, len(b))
	}
	n := len(b) / 4
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out
}

// fixtureMeta is a minimal subset of fixtures.json that these tests use.
type fixtureMeta struct {
	InputIDs []int `json:"input_ids"`
	SeqLen   int   `json:"seq_len"`
}

func loadFixtureMeta(t *testing.T) (fixtureMeta, string) {
	t.Helper()
	root := repoRoot(t)
	p := filepath.Join(root, "fixtures", "fixtures.json")
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatalf("read %s: %v", p, err)
	}
	var m fixtureMeta
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("parse %s: %v", p, err)
	}
	return m, root
}

func maxAbsErr(a, b []float32) (maxErr float32, idx int) {
	if len(a) != len(b) {
		return float32(math.Inf(1)), -1
	}
	maxErr = 0
	idx = -1
	for i := range a {
		d := a[i] - b[i]
		if d < 0 {
			d = -d
		}
		if d > maxErr {
			maxErr = d
			idx = i
		}
	}
	return
}

func TestRMSNorm(t *testing.T) {
	// x=[1,2,3,4], weight=[1,1,1,1], eps=1e-5. mean(x^2)=(1+4+9+16)/4=7.5.
	// norm = x / sqrt(7.5 + 1e-5).
	x := []float32{1, 2, 3, 4}
	w := []float32{1, 1, 1, 1}
	got := RMSNorm(x, w, 1, 4, 1e-5)
	scale := 1.0 / math.Sqrt(7.5+1e-5)
	want := []float32{
		float32(1 * scale),
		float32(2 * scale),
		float32(3 * scale),
		float32(4 * scale),
	}
	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		d := got[i] - want[i]
		if d < 0 {
			d = -d
		}
		if d > 1e-6 {
			t.Errorf("idx %d: got %v, want %v (abs err %v)", i, got[i], want[i], d)
		}
	}
}

func TestRMSNormAgainstFixture(t *testing.T) {
	meta, root := loadFixtureMeta(t)
	T := meta.SeqLen
	if T == 0 {
		t.Fatal("fixture seq_len is zero")
	}
	const D = 640

	// Load the input_layernorm.weight for layer 0.
	stPath := filepath.Join(root, "model", "model.safetensors")
	r, err := safetensors.Open(stPath)
	if err != nil {
		t.Fatalf("open safetensors: %v", err)
	}
	defer r.Close()

	ln, err := r.Tensor("model.layers.0.input_layernorm.weight")
	if err != nil {
		t.Fatalf("get input_layernorm: %v", err)
	}
	w := ln.Float32s()
	if len(w) != D {
		t.Fatalf("weight len = %d, want %d", len(w), D)
	}

	// Input is the embedding output [1, 19, 640] => treat as [19, 640].
	x := loadF32(t, filepath.Join(root, "fixtures", "embedding.f32.bin"))
	if len(x) != T*D {
		t.Fatalf("embedding len = %d, want %d", len(x), T*D)
	}
	want := loadF32(t, filepath.Join(root, "fixtures", "layer0_pre_attn_norm.f32.bin"))
	if len(want) != T*D {
		t.Fatalf("want len = %d, want %d", len(want), T*D)
	}

	got := RMSNorm(x, w, T, D, 1e-5)
	errv, idx := maxAbsErr(got, want)
	if errv > 1e-4 {
		t.Errorf("max abs err = %v at idx %d (got %v want %v)", errv, idx, got[idx], want[idx])
	}
}

func TestEmbeddingLookupAgainstFixture(t *testing.T) {
	meta, root := loadFixtureMeta(t)
	T := meta.SeqLen
	const D = 640
	const V = 200064

	stPath := filepath.Join(root, "model", "model.safetensors")
	r, err := safetensors.Open(stPath)
	if err != nil {
		t.Fatalf("open safetensors: %v", err)
	}
	defer r.Close()

	et, err := r.Tensor("model.embed_tokens.weight")
	if err != nil {
		t.Fatalf("get embed_tokens: %v", err)
	}
	table := et.Float32s()
	if len(table) != V*D {
		t.Fatalf("embed table len = %d, want %d", len(table), V*D)
	}

	got := EmbeddingLookup(table, V, D, meta.InputIDs)
	want := loadF32(t, filepath.Join(root, "fixtures", "embedding.f32.bin"))
	if len(got) != len(want) {
		t.Fatalf("len mismatch: got %d want %d", len(got), len(want))
	}
	// bf16 -> f32 expansion is bit-exact; expect exact equality.
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("idx %d: got %v want %v", i, got[i], want[i])
		}
	}
	_ = T
}

func TestYarnRoPETablesAgainstFixture(t *testing.T) {
	meta, root := loadFixtureMeta(t)
	T := meta.SeqLen
	const headDim = 64
	const half = headDim / 2

	p := YarnParams{
		HeadDim:              headDim,
		Theta:                150000,
		OriginalMaxPositions: 4096,
		Factor:               32,
		BetaFast:             32,
		BetaSlow:             1,
	}
	cos, sin := YarnRoPETables(T, p)
	if len(cos) != T*half || len(sin) != T*half {
		t.Fatalf("table size: cos=%d sin=%d want %d each", len(cos), len(sin), T*half)
	}

	wantCos := loadF32(t, filepath.Join(root, "fixtures", "rope_cos.f32.bin"))
	wantSin := loadF32(t, filepath.Join(root, "fixtures", "rope_sin.f32.bin"))
	if len(wantCos) != T*half {
		t.Fatalf("wantCos len = %d, want %d", len(wantCos), T*half)
	}

	errC, idxC := maxAbsErr(cos, wantCos)
	if errC > 1e-5 {
		t.Errorf("cos max abs err = %v at %d (got %v want %v)", errC, idxC, cos[idxC], wantCos[idxC])
	}
	errS, idxS := maxAbsErr(sin, wantSin)
	if errS > 1e-5 {
		t.Errorf("sin max abs err = %v at %d (got %v want %v)", errS, idxS, sin[idxS], wantSin[idxS])
	}
}
