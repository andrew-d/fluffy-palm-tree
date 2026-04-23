package privatemodel

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// repoRoot locates the repository root by walking up from this test file
// until we find go.mod.
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
	t.Fatal("could not locate repo root")
	return ""
}

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

// fixtureMeta is the subset of fixtures.json that these tests use.
type fixtureMeta struct {
	InputIDs     []int `json:"input_ids"`
	SeqLen       int   `json:"seq_len"`
	PredLabelIDs []int `json:"pred_label_ids"`
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

// TestLoadFromSafetensorsSmoke simply confirms that the loader can read the
// full model from disk and that the layer count matches the config.
func TestLoadFromSafetensorsSmoke(t *testing.T) {
	root := repoRoot(t)
	modelDir := filepath.Join(root, "model")
	m, err := LoadFromSafetensors(modelDir)
	if err != nil {
		t.Fatalf("LoadFromSafetensors: %v", err)
	}
	if m.Config == nil {
		t.Fatal("Config is nil")
	}
	if got, want := len(m.Layers), m.Config.NumHiddenLayers; got != want {
		t.Fatalf("len(Layers) = %d, want %d", got, want)
	}
	if m.Config.NumHiddenLayers != 8 {
		t.Errorf("expected 8 layers, got %d", m.Config.NumHiddenLayers)
	}
	// Basic sanity checks on weight shapes.
	if got, want := len(m.EmbedTokens), m.Config.VocabSize*m.Config.HiddenSize; got != want {
		t.Errorf("EmbedTokens len = %d, want %d", got, want)
	}
	if got, want := len(m.Norm), m.Config.HiddenSize; got != want {
		t.Errorf("Norm len = %d, want %d", got, want)
	}
	if got, want := len(m.ScoreW), m.Config.NumLabels*m.Config.HiddenSize; got != want {
		t.Errorf("ScoreW len = %d, want %d", got, want)
	}
	if got, want := len(m.ScoreB), m.Config.NumLabels; got != want {
		t.Errorf("ScoreB len = %d, want %d", got, want)
	}
}

// TestForwardLayer0Attn checks that seeding with the fixture embedding,
// running layer 0's input-norm + attention + residual produces something
// close to fixtures/layer0_out.f32.bin's attention-residual component. We
// actually compare against (embedding + layer0_attn_out) which is the
// post-residual state before the MLP.
func TestForwardLayer0Attn(t *testing.T) {
	meta, root := loadFixtureMeta(t)
	if meta.SeqLen == 0 {
		t.Fatal("fixture seq_len is zero")
	}
	modelDir := filepath.Join(root, "model")
	m, err := LoadFromSafetensors(modelDir)
	if err != nil {
		t.Fatalf("LoadFromSafetensors: %v", err)
	}

	// Seed from the fixture embedding so we isolate the attention stage.
	embed := loadF32(t, filepath.Join(root, "fixtures", "embedding.f32.bin"))

	got := m.ForwardFirstLayerAttn(embed, meta.SeqLen)

	// Expected: embedding + layer0_attn_out.
	attnOut := loadF32(t, filepath.Join(root, "fixtures", "layer0_attn_out.f32.bin"))
	if len(embed) != len(attnOut) {
		t.Fatalf("embedding/attn_out length mismatch: %d vs %d", len(embed), len(attnOut))
	}
	want := make([]float32, len(embed))
	for i := range embed {
		want[i] = embed[i] + attnOut[i]
	}

	if len(got) != len(want) {
		t.Fatalf("len mismatch: got %d want %d", len(got), len(want))
	}
	errv, idx := maxAbsErr(got, want)
	if errv > 1e-3 {
		t.Fatalf("max abs err = %v at %d (got %v want %v)", errv, idx, got[idx], want[idx])
	}
	t.Logf("layer0 attn+residual max abs err = %v at idx %d", errv, idx)
}

// TestFinalHiddenAgainstFixture runs the full forward pass and compares
// the final-norm hidden states against the fixture.
func TestFinalHiddenAgainstFixture(t *testing.T) {
	meta, root := loadFixtureMeta(t)
	if meta.SeqLen == 0 {
		t.Fatal("fixture seq_len is zero")
	}
	modelDir := filepath.Join(root, "model")
	m, err := LoadFromSafetensors(modelDir)
	if err != nil {
		t.Fatalf("LoadFromSafetensors: %v", err)
	}

	hidden, T := m.ForwardFinalHidden(meta.InputIDs)
	if T != meta.SeqLen {
		t.Fatalf("T = %d, want %d", T, meta.SeqLen)
	}

	want := loadF32(t, filepath.Join(root, "fixtures", "final_norm.f32.bin"))
	if len(hidden) != len(want) {
		t.Fatalf("len mismatch: got %d want %d", len(hidden), len(want))
	}
	errv, idx := maxAbsErr(hidden, want)
	if errv > 2e-3 {
		t.Fatalf("final hidden max abs err = %v at %d (got %v want %v)",
			errv, idx, hidden[idx], want[idx])
	}
	t.Logf("final hidden max abs err = %v at idx %d", errv, idx)
}

// TestLogitsAgainstFixture runs Forward end-to-end and checks that the
// logits match the fixture AND the per-token argmax is identical to the
// reference's pred_label_ids.
func TestLogitsAgainstFixture(t *testing.T) {
	meta, root := loadFixtureMeta(t)
	if meta.SeqLen == 0 {
		t.Fatal("fixture seq_len is zero")
	}
	modelDir := filepath.Join(root, "model")
	m, err := LoadFromSafetensors(modelDir)
	if err != nil {
		t.Fatalf("LoadFromSafetensors: %v", err)
	}

	logits, T := m.Forward(meta.InputIDs)
	if T != meta.SeqLen {
		t.Fatalf("T = %d, want %d", T, meta.SeqLen)
	}

	want := loadF32(t, filepath.Join(root, "fixtures", "logits.f32.bin"))
	if len(logits) != len(want) {
		t.Fatalf("len mismatch: got %d want %d", len(logits), len(want))
	}
	errv, idx := maxAbsErr(logits, want)
	if errv > 5e-3 {
		t.Fatalf("logits max abs err = %v at %d (got %v want %v)",
			errv, idx, logits[idx], want[idx])
	}
	t.Logf("logits max abs err = %v at idx %d", errv, idx)

	// Check per-token argmax. The classifier head has 33 labels.
	const nLabels = 33
	if got := len(meta.PredLabelIDs); got != T {
		t.Fatalf("fixture pred_label_ids has length %d, want %d", got, T)
	}
	for t_ := 0; t_ < T; t_++ {
		row := logits[t_*nLabels : (t_+1)*nLabels]
		best := 0
		for k := 1; k < nLabels; k++ {
			if row[k] > row[best] {
				best = k
			}
		}
		if best != meta.PredLabelIDs[t_] {
			t.Errorf("token %d: argmax = %d, want %d", t_, best, meta.PredLabelIDs[t_])
		}
	}
}
