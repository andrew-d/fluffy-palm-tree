package safetensors_test

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/andrew-d/openai-privacy/internal/safetensors"
)

// modelPath returns the absolute path to the repo's model.safetensors,
// computed relative to this test file so the test can be run from any cwd.
func modelPath(t *testing.T) string {
	t.Helper()
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	// internal/safetensors/safetensors_test.go -> repo root is three up.
	repo := filepath.Clean(filepath.Join(filepath.Dir(thisFile), "..", ".."))
	p := filepath.Join(repo, "model", "model.safetensors")
	if _, err := os.Stat(p); err != nil {
		t.Fatalf("missing model weights at %s: %v", p, err)
	}
	return p
}

// fixturePath returns the absolute path to a file under fixtures/.
func fixturePath(t *testing.T, name string) string {
	t.Helper()
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	repo := filepath.Clean(filepath.Join(filepath.Dir(thisFile), "..", ".."))
	return filepath.Join(repo, "fixtures", name)
}

func TestReaderNames(t *testing.T) {
	r, err := safetensors.Open(modelPath(t))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer r.Close()

	names := r.Names()
	if got, want := len(names), 140; got != want {
		t.Errorf("Names() length = %d, want %d", got, want)
	}

	set := make(map[string]struct{}, len(names))
	for _, n := range names {
		set[n] = struct{}{}
	}
	for _, want := range []string{
		"model.embed_tokens.weight",
		"model.layers.0.self_attn.q_proj.weight",
		"score.bias",
	} {
		if _, ok := set[want]; !ok {
			t.Errorf("Names() missing %q", want)
		}
	}
}

func TestReaderEmbedTokens(t *testing.T) {
	r, err := safetensors.Open(modelPath(t))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer r.Close()

	tensor, err := r.Tensor("model.embed_tokens.weight")
	if err != nil {
		t.Fatalf("Tensor: %v", err)
	}
	if tensor.DType != safetensors.BF16 {
		t.Errorf("dtype = %v, want BF16", tensor.DType)
	}
	wantShape := []int{200064, 640}
	if got := tensor.Shape; len(got) != len(wantShape) || got[0] != wantShape[0] || got[1] != wantShape[1] {
		t.Errorf("shape = %v, want %v", got, wantShape)
	}
	if got, want := len(tensor.Bytes), 200064*640*2; got != want {
		t.Errorf("len(Bytes) = %d, want %d", got, want)
	}
}

func TestReaderAttnSinks(t *testing.T) {
	r, err := safetensors.Open(modelPath(t))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer r.Close()

	tensor, err := r.Tensor("model.layers.0.self_attn.sinks")
	if err != nil {
		t.Fatalf("Tensor: %v", err)
	}
	if tensor.DType != safetensors.F32 {
		t.Errorf("dtype = %v, want F32", tensor.DType)
	}
	wantShape := []int{14}
	if len(tensor.Shape) != 1 || tensor.Shape[0] != wantShape[0] {
		t.Errorf("shape = %v, want %v", tensor.Shape, wantShape)
	}
}

func TestReaderScoreBiasFloat32s(t *testing.T) {
	r, err := safetensors.Open(modelPath(t))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer r.Close()

	tensor, err := r.Tensor("score.bias")
	if err != nil {
		t.Fatalf("Tensor: %v", err)
	}
	if tensor.DType != safetensors.BF16 {
		t.Errorf("dtype = %v, want BF16", tensor.DType)
	}
	if len(tensor.Shape) != 1 || tensor.Shape[0] != 33 {
		t.Errorf("shape = %v, want [33]", tensor.Shape)
	}

	got := tensor.Float32s()
	if len(got) != 33 {
		t.Fatalf("Float32s length = %d, want 33", len(got))
	}

	// Cross-check against the reference dump from dump_fixtures.py, which
	// reads the same tensor from the model and casts bf16 -> f32.
	raw, err := os.ReadFile(fixturePath(t, "score_bias.f32.bin"))
	if err != nil {
		t.Fatalf("read score_bias fixture: %v", err)
	}
	if len(raw) != 33*4 {
		t.Fatalf("score_bias fixture = %d bytes, want %d", len(raw), 33*4)
	}
	want := make([]float32, 33)
	for i := range want {
		want[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}

	const eps = 1e-5
	for i := range want {
		if diff := float64(got[i] - want[i]); diff < -eps || diff > eps {
			t.Errorf("Float32s[%d] = %v, want %v (diff %v)", i, got[i], want[i], diff)
		}
	}
}

func TestReaderMissingTensor(t *testing.T) {
	r, err := safetensors.Open(modelPath(t))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer r.Close()

	if _, err := r.Tensor("does.not.exist"); err == nil {
		t.Error("Tensor(nonexistent) returned nil error, want non-nil")
	}
}
