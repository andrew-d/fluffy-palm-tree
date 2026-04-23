package nn

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/andrew-d/openai-privacy/internal/safetensors"
)

// TestTopKRouterAgainstFixture loads the layer 0 router weights and the
// pre-MLP-norm hidden states, then verifies that TopKRouter produces the
// expert scores and indices recorded in the fixture.
//
// The fixture scores (`layer0_router_scores.f32.bin`) are already post
// `/top_k` (see `OpenAIPrivacyFilterTopKRouter.forward`), so TopKRouter
// returns scores already divided by topK to match.
func TestTopKRouterAgainstFixture(t *testing.T) {
	meta, root := loadFixtureMeta(t)
	T := meta.SeqLen
	if T == 0 {
		t.Fatal("fixture seq_len is zero")
	}
	const (
		D          = 640
		numExperts = 128
		topK       = 4
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

	wRouter := load("model.layers.0.mlp.router.weight")
	bRouter := load("model.layers.0.mlp.router.bias")
	if got, want := len(wRouter), numExperts*D; got != want {
		t.Fatalf("wRouter len = %d, want %d", got, want)
	}
	if got, want := len(bRouter), numExperts; got != want {
		t.Fatalf("bRouter len = %d, want %d", got, want)
	}

	hidden := loadF32(t, filepath.Join(root, "fixtures", "layer0_pre_mlp_norm.f32.bin"))
	if len(hidden) != T*D {
		t.Fatalf("pre-mlp-norm len = %d, want %d", len(hidden), T*D)
	}

	scores, indices := TopKRouter(hidden, wRouter, bRouter, T, D, numExperts, topK)
	if len(scores) != T*topK {
		t.Fatalf("scores len = %d, want %d", len(scores), T*topK)
	}
	if len(indices) != T*topK {
		t.Fatalf("indices len = %d, want %d", len(indices), T*topK)
	}

	wantScores := loadF32(t, filepath.Join(root, "fixtures", "layer0_router_scores.f32.bin"))
	wantIndicesF := loadF32(t, filepath.Join(root, "fixtures", "layer0_router_indices.f32.bin"))

	if len(wantScores) != T*topK {
		t.Fatalf("wantScores len = %d, want %d", len(wantScores), T*topK)
	}
	if len(wantIndicesF) != T*topK {
		t.Fatalf("wantIndices len = %d, want %d", len(wantIndicesF), T*topK)
	}

	// Verify indices match exactly (fixtures store ints as float32).
	for i := range wantIndicesF {
		wantIdx := int(wantIndicesF[i])
		if indices[i] != wantIdx {
			t.Errorf("indices[%d] = %d, want %d", i, indices[i], wantIdx)
		}
	}

	// Verify scores (already divided by topK in both cases).
	errv, idx := maxAbsErr(scores, wantScores)
	if errv > 1e-5 {
		t.Errorf("scores max abs err = %v at idx %d (got %v want %v)",
			errv, idx, scores[idx], wantScores[idx])
	}
	t.Logf("router scores max abs err = %v at idx %d", errv, idx)
}

// TestMoELayerAgainstFixture loads all layer 0 MoE weights and the pre-MLP
// hidden states, runs the full MoE MLP (router + experts), and compares
// against layer0_mlp_out.
func TestMoELayerAgainstFixture(t *testing.T) {
	meta, root := loadFixtureMeta(t)
	T := meta.SeqLen
	if T == 0 {
		t.Fatal("fixture seq_len is zero")
	}
	const (
		D          = 640
		I          = 640
		numExperts = 128
		topK       = 4
		limit      = float32(7.0)
		alpha      = float32(1.702)
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

	wRouter := load("model.layers.0.mlp.router.weight")
	bRouter := load("model.layers.0.mlp.router.bias")
	gateUpProj := load("model.layers.0.mlp.experts.gate_up_proj")
	gateUpBias := load("model.layers.0.mlp.experts.gate_up_proj_bias")
	downProj := load("model.layers.0.mlp.experts.down_proj")
	downBias := load("model.layers.0.mlp.experts.down_proj_bias")

	if got, want := len(gateUpProj), numExperts*D*(2*I); got != want {
		t.Fatalf("gateUpProj len = %d, want %d", got, want)
	}
	if got, want := len(gateUpBias), numExperts*(2*I); got != want {
		t.Fatalf("gateUpBias len = %d, want %d", got, want)
	}
	if got, want := len(downProj), numExperts*I*D; got != want {
		t.Fatalf("downProj len = %d, want %d", got, want)
	}
	if got, want := len(downBias), numExperts*D; got != want {
		t.Fatalf("downBias len = %d, want %d", got, want)
	}

	hidden := loadF32(t, filepath.Join(root, "fixtures", "layer0_pre_mlp_norm.f32.bin"))
	if len(hidden) != T*D {
		t.Fatalf("pre-mlp-norm len = %d, want %d", len(hidden), T*D)
	}

	got := MoELayer(
		hidden, wRouter, bRouter,
		gateUpProj, gateUpBias, downProj, downBias,
		T, D, I, numExperts, topK,
		limit, alpha,
	)

	want := loadF32(t, filepath.Join(root, "fixtures", "layer0_mlp_out.f32.bin"))
	if len(got) != len(want) {
		t.Fatalf("len mismatch: got %d want %d", len(got), len(want))
	}
	errv, idx := maxAbsErr(got, want)
	if errv >= 1e-3 {
		t.Errorf("max abs err = %v at idx %d (got %v want %v)",
			errv, idx, got[idx], want[idx])
	}
	// Per-element check for any wild outliers.
	var failCount int
	for i := range want {
		d := got[i] - want[i]
		if d < 0 {
			d = -d
		}
		if math.IsNaN(float64(d)) || d >= 1e-3 {
			if failCount < 5 {
				t.Errorf("idx %d: got %v want %v (abs err %v)", i, got[i], want[i], d)
			}
			failCount++
		}
	}
	if failCount > 0 {
		t.Errorf("%d elements exceed 1e-3 tolerance", failCount)
	}
	t.Logf("layer0 MoE max abs err = %v at idx %d", errv, idx)
}
