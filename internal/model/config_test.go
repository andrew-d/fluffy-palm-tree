package model

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func repoRoot(t *testing.T) string {
	t.Helper()
	_, file, _, _ := runtime.Caller(0)
	dir := filepath.Dir(file)
	for i := 0; i < 10; i++ {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		dir = filepath.Dir(dir)
	}
	t.Fatal("no repo root found")
	return ""
}

func TestLoadConfig(t *testing.T) {
	root := repoRoot(t)
	cfg, err := LoadConfig(filepath.Join(root, "model"))
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}

	// Spot-check fields that should come straight from the known model.
	if cfg.HiddenSize != 640 {
		t.Errorf("HiddenSize = %d, want 640", cfg.HiddenSize)
	}
	if cfg.NumHiddenLayers != 8 {
		t.Errorf("NumHiddenLayers = %d, want 8", cfg.NumHiddenLayers)
	}
	if cfg.NumAttentionHeads != 14 {
		t.Errorf("NumAttentionHeads = %d, want 14", cfg.NumAttentionHeads)
	}
	if cfg.NumKVHeads != 2 {
		t.Errorf("NumKVHeads = %d, want 2", cfg.NumKVHeads)
	}
	if cfg.HeadDim != 64 {
		t.Errorf("HeadDim = %d, want 64", cfg.HeadDim)
	}
	if cfg.NumExperts != 128 {
		t.Errorf("NumExperts = %d, want 128", cfg.NumExperts)
	}
	if cfg.NumExpertsPerTok != 4 {
		t.Errorf("NumExpertsPerTok = %d, want 4", cfg.NumExpertsPerTok)
	}
	if cfg.IntermediateSize != 640 {
		t.Errorf("IntermediateSize = %d, want 640", cfg.IntermediateSize)
	}
	if cfg.VocabSize != 200064 {
		t.Errorf("VocabSize = %d, want 200064", cfg.VocabSize)
	}
	if cfg.NumLabels != 33 {
		t.Errorf("NumLabels = %d, want 33", cfg.NumLabels)
	}
	// sliding_window in file is 128; we bake in the +1 used by attention.
	if cfg.SlidingWindow != 129 {
		t.Errorf("SlidingWindow = %d, want 129", cfg.SlidingWindow)
	}
	if cfg.RMSNormEps != 1e-5 {
		t.Errorf("RMSNormEps = %v, want 1e-5", cfg.RMSNormEps)
	}
	if cfg.RopeTheta != 150000 {
		t.Errorf("RopeTheta = %v, want 150000", cfg.RopeTheta)
	}
	if cfg.RopeFactor != 32 {
		t.Errorf("RopeFactor = %v, want 32", cfg.RopeFactor)
	}
	if cfg.RopeBetaFast != 32 {
		t.Errorf("RopeBetaFast = %v, want 32", cfg.RopeBetaFast)
	}
	if cfg.RopeBetaSlow != 1 {
		t.Errorf("RopeBetaSlow = %v, want 1", cfg.RopeBetaSlow)
	}
	if cfg.OriginalMaxPositions != 4096 {
		t.Errorf("OriginalMaxPositions = %d, want 4096", cfg.OriginalMaxPositions)
	}

	// id2label[0] must be "O".
	if cfg.ID2Label[0] != "O" {
		t.Errorf("ID2Label[0] = %q, want \"O\"", cfg.ID2Label[0])
	}
	if cfg.ID2Label[17] != "B-private_person" {
		t.Errorf("ID2Label[17] = %q, want \"B-private_person\"", cfg.ID2Label[17])
	}
	if cfg.ID2Label[15] != "E-private_email" {
		t.Errorf("ID2Label[15] = %q, want \"E-private_email\"", cfg.ID2Label[15])
	}
}
