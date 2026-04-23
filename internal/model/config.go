// Package model loads the HuggingFace config.json of the openai/privacy-filter
// model into a typed Go struct. Only the fields used by the Go port are
// extracted; unknown fields are ignored.
package model

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
)

// Config captures the subset of `config.json` needed by the Go inference
// implementation.
type Config struct {
	HiddenSize        int
	NumHiddenLayers   int
	NumAttentionHeads int
	NumKVHeads        int
	HeadDim           int

	NumExperts       int
	NumExpertsPerTok int
	IntermediateSize int

	VocabSize int
	NumLabels int

	// SlidingWindow is the effective window value `sliding_window + 1` per the
	// gpt_oss attention code (+1 for flash-attention symmetry). The model
	// config stores 128; this field is 129.
	SlidingWindow int

	RMSNormEps float32

	RopeTheta            float64
	RopeFactor           float64
	RopeBetaFast         float64
	RopeBetaSlow         float64
	OriginalMaxPositions int

	// ID2Label is indexed by label id, i.e. ID2Label[k] is the label name for
	// class k. The length equals NumLabels.
	ID2Label []string
}

// rawConfig mirrors the JSON layout of config.json for the fields we care
// about. We decode the id2label map from string keys ("0", "1", ...) and
// repack it into a dense slice after reading.
type rawConfig struct {
	HiddenSize        int `json:"hidden_size"`
	NumHiddenLayers   int `json:"num_hidden_layers"`
	NumAttentionHeads int `json:"num_attention_heads"`
	NumKVHeads        int `json:"num_key_value_heads"`
	HeadDim           int `json:"head_dim"`

	NumLocalExperts  int `json:"num_local_experts"`
	NumExpertsPerTok int `json:"num_experts_per_tok"`
	IntermediateSize int `json:"intermediate_size"`

	VocabSize int `json:"vocab_size"`

	SlidingWindow int `json:"sliding_window"`

	RMSNormEps float32 `json:"rms_norm_eps"`

	RopeParameters struct {
		RopeTheta                     float64 `json:"rope_theta"`
		Factor                        float64 `json:"factor"`
		BetaFast                      float64 `json:"beta_fast"`
		BetaSlow                      float64 `json:"beta_slow"`
		OriginalMaxPositionEmbeddings int     `json:"original_max_position_embeddings"`
	} `json:"rope_parameters"`

	ID2Label map[string]string `json:"id2label"`
}

// LoadConfig reads config.json from dir and parses it into a Config.
func LoadConfig(dir string) (*Config, error) {
	p := filepath.Join(dir, "config.json")
	b, err := os.ReadFile(p)
	if err != nil {
		return nil, fmt.Errorf("model: read %s: %w", p, err)
	}
	var raw rawConfig
	if err := json.Unmarshal(b, &raw); err != nil {
		return nil, fmt.Errorf("model: parse %s: %w", p, err)
	}

	// Repack id2label (keyed by decimal string) into a dense slice, checking
	// for gaps as we go.
	n := len(raw.ID2Label)
	labels := make([]string, n)
	for k, v := range raw.ID2Label {
		id, err := strconv.Atoi(k)
		if err != nil {
			return nil, fmt.Errorf("model: id2label key %q: %w", k, err)
		}
		if id < 0 || id >= n {
			return nil, fmt.Errorf("model: id2label key %q out of range [0,%d)", k, n)
		}
		labels[id] = v
	}
	for i, s := range labels {
		if s == "" {
			return nil, fmt.Errorf("model: id2label missing entry for id %d", i)
		}
	}

	return &Config{
		HiddenSize:        raw.HiddenSize,
		NumHiddenLayers:   raw.NumHiddenLayers,
		NumAttentionHeads: raw.NumAttentionHeads,
		NumKVHeads:        raw.NumKVHeads,
		HeadDim:           raw.HeadDim,

		NumExperts:       raw.NumLocalExperts,
		NumExpertsPerTok: raw.NumExpertsPerTok,
		IntermediateSize: raw.IntermediateSize,

		VocabSize: raw.VocabSize,
		NumLabels: n,

		// The gpt_oss attention code adds +1 to the configured window for
		// flash-attention symmetry; we pre-bake that here.
		SlidingWindow: raw.SlidingWindow + 1,

		RMSNormEps: raw.RMSNormEps,

		RopeTheta:            raw.RopeParameters.RopeTheta,
		RopeFactor:           raw.RopeParameters.Factor,
		RopeBetaFast:         raw.RopeParameters.BetaFast,
		RopeBetaSlow:         raw.RopeParameters.BetaSlow,
		OriginalMaxPositions: raw.RopeParameters.OriginalMaxPositionEmbeddings,

		ID2Label: labels,
	}, nil
}
