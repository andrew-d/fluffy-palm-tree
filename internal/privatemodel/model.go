// Package privatemodel assembles the per-layer weights of the
// openai/privacy-filter model and provides an end-to-end forward pass
// (encoder + classifier head) that reproduces the reference outputs.
//
// All weights are decoded to fp32 at load time; the forward pass is a
// straight-line translation of the Python reference described in
// docs/architecture.md. Only the standard library is used.
package privatemodel

import (
	"fmt"
	"path/filepath"

	"github.com/andrew-d/openai-privacy/internal/model"
	"github.com/andrew-d/openai-privacy/internal/nn"
	"github.com/andrew-d/openai-privacy/internal/safetensors"
)

// Layer holds the weights for a single transformer block.
//
// Shapes (using the config's HiddenSize = D, IntermediateSize = I,
// NumAttentionHeads = Hq, NumKVHeads = Hkv, HeadDim = Dh, NumExperts = E):
//
//	InputNormW, PostAttnNormW: [D]
//	QW:  [Hq*Dh, D]    QB: [Hq*Dh]
//	KW:  [Hkv*Dh, D]   KB: [Hkv*Dh]
//	VW:  [Hkv*Dh, D]   VB: [Hkv*Dh]
//	OW:  [D, Hq*Dh]    OB: [D]
//	Sinks:  [Hq]       fp32, one per query head
//	RouterW: [E, D]    RouterB: [E]
//	ExpertsGateUpW:   [E, D, 2*I]    ExpertsGateUpB: [E, 2*I]
//	ExpertsDownW:     [E, I, D]      ExpertsDownB:   [E, D]
type Layer struct {
	InputNormW    []float32
	PostAttnNormW []float32

	QW, QB []float32
	KW, KB []float32
	VW, VB []float32
	OW, OB []float32

	Sinks []float32

	RouterW, RouterB []float32

	ExpertsGateUpW, ExpertsGateUpB []float32
	ExpertsDownW, ExpertsDownB     []float32
}

// Model is the full encoder + classifier head, with every tensor decoded to
// fp32.
type Model struct {
	Config *model.Config

	EmbedTokens []float32 // [vocab, hidden]
	Norm        []float32 // [hidden] final rms-norm weight
	ScoreW      []float32 // [labels, hidden]
	ScoreB      []float32 // [labels]

	Layers []*Layer
}

// LoadFromSafetensors reads modelDir/model.safetensors and modelDir/config.json
// and returns a fully-loaded Model ready for Forward.
//
// Every tensor is decoded to fp32 up-front; the peak memory footprint is
// roughly 2x the on-disk bf16 size (~5.6 GB for the 2.8 GB weights file).
func LoadFromSafetensors(modelDir string) (*Model, error) {
	cfg, err := model.LoadConfig(modelDir)
	if err != nil {
		return nil, err
	}

	stPath := filepath.Join(modelDir, "model.safetensors")
	r, err := safetensors.Open(stPath)
	if err != nil {
		return nil, err
	}
	defer r.Close()

	loadF32 := func(name string, wantLen int) ([]float32, error) {
		tn, err := r.Tensor(name)
		if err != nil {
			return nil, fmt.Errorf("load %s: %w", name, err)
		}
		v := tn.Float32s()
		if wantLen > 0 && len(v) != wantLen {
			return nil, fmt.Errorf("%s: got %d elements, want %d", name, len(v), wantLen)
		}
		return v, nil
	}

	D := cfg.HiddenSize
	Hq := cfg.NumAttentionHeads
	Hkv := cfg.NumKVHeads
	Dh := cfg.HeadDim
	I := cfg.IntermediateSize
	E := cfg.NumExperts
	V := cfg.VocabSize
	L := cfg.NumLabels

	embed, err := loadF32("model.embed_tokens.weight", V*D)
	if err != nil {
		return nil, err
	}
	finalNorm, err := loadF32("model.norm.weight", D)
	if err != nil {
		return nil, err
	}
	scoreW, err := loadF32("score.weight", L*D)
	if err != nil {
		return nil, err
	}
	scoreB, err := loadF32("score.bias", L)
	if err != nil {
		return nil, err
	}

	layers := make([]*Layer, cfg.NumHiddenLayers)
	qDim := Hq * Dh
	kvDim := Hkv * Dh
	for li := 0; li < cfg.NumHiddenLayers; li++ {
		base := fmt.Sprintf("model.layers.%d", li)

		inLN, err := loadF32(base+".input_layernorm.weight", D)
		if err != nil {
			return nil, err
		}
		postLN, err := loadF32(base+".post_attention_layernorm.weight", D)
		if err != nil {
			return nil, err
		}

		qw, err := loadF32(base+".self_attn.q_proj.weight", qDim*D)
		if err != nil {
			return nil, err
		}
		qb, err := loadF32(base+".self_attn.q_proj.bias", qDim)
		if err != nil {
			return nil, err
		}
		kw, err := loadF32(base+".self_attn.k_proj.weight", kvDim*D)
		if err != nil {
			return nil, err
		}
		kb, err := loadF32(base+".self_attn.k_proj.bias", kvDim)
		if err != nil {
			return nil, err
		}
		vw, err := loadF32(base+".self_attn.v_proj.weight", kvDim*D)
		if err != nil {
			return nil, err
		}
		vb, err := loadF32(base+".self_attn.v_proj.bias", kvDim)
		if err != nil {
			return nil, err
		}
		ow, err := loadF32(base+".self_attn.o_proj.weight", D*qDim)
		if err != nil {
			return nil, err
		}
		ob, err := loadF32(base+".self_attn.o_proj.bias", D)
		if err != nil {
			return nil, err
		}
		sinks, err := loadF32(base+".self_attn.sinks", Hq)
		if err != nil {
			return nil, err
		}

		routerW, err := loadF32(base+".mlp.router.weight", E*D)
		if err != nil {
			return nil, err
		}
		routerB, err := loadF32(base+".mlp.router.bias", E)
		if err != nil {
			return nil, err
		}
		guW, err := loadF32(base+".mlp.experts.gate_up_proj", E*D*(2*I))
		if err != nil {
			return nil, err
		}
		guB, err := loadF32(base+".mlp.experts.gate_up_proj_bias", E*(2*I))
		if err != nil {
			return nil, err
		}
		dwW, err := loadF32(base+".mlp.experts.down_proj", E*I*D)
		if err != nil {
			return nil, err
		}
		dwB, err := loadF32(base+".mlp.experts.down_proj_bias", E*D)
		if err != nil {
			return nil, err
		}

		layers[li] = &Layer{
			InputNormW:     inLN,
			PostAttnNormW:  postLN,
			QW:             qw,
			QB:             qb,
			KW:             kw,
			KB:             kb,
			VW:             vw,
			VB:             vb,
			OW:             ow,
			OB:             ob,
			Sinks:          sinks,
			RouterW:        routerW,
			RouterB:        routerB,
			ExpertsGateUpW: guW,
			ExpertsGateUpB: guB,
			ExpertsDownW:   dwW,
			ExpertsDownB:   dwB,
		}
	}

	return &Model{
		Config:      cfg,
		EmbedTokens: embed,
		Norm:        finalNorm,
		ScoreW:      scoreW,
		ScoreB:      scoreB,
		Layers:      layers,
	}, nil
}

// yarnParams extracts the yarn-RoPE parameters from the Config.
func (m *Model) yarnParams() nn.YarnParams {
	return nn.YarnParams{
		HeadDim:              m.Config.HeadDim,
		Theta:                m.Config.RopeTheta,
		OriginalMaxPositions: m.Config.OriginalMaxPositions,
		Factor:               m.Config.RopeFactor,
		BetaFast:             m.Config.RopeBetaFast,
		BetaSlow:             m.Config.RopeBetaSlow,
	}
}

// runLayerAttn applies the attention sub-block of a single transformer
// layer: RMSNorm(x, InputNormW) -> GQAAttentionWithSinks -> residual.
// Returns the post-residual hidden state.
func (m *Model) runLayerAttn(l *Layer, x, cos, sin []float32, T int) []float32 {
	cfg := m.Config
	D := cfg.HiddenSize
	h := nn.RMSNorm(x, l.InputNormW, T, D, cfg.RMSNormEps)
	y := nn.GQAAttentionWithSinks(
		h,
		l.QW, l.QB, l.KW, l.KB, l.VW, l.VB, l.OW, l.OB,
		l.Sinks,
		cos, sin,
		T, D, cfg.HeadDim, cfg.NumAttentionHeads, cfg.NumKVHeads, cfg.SlidingWindow,
	)
	return nn.Add(x, y)
}

// runLayerMLP applies the MLP sub-block of a single transformer layer:
// RMSNorm(x, PostAttnNormW) -> MoELayer -> residual.
func (m *Model) runLayerMLP(l *Layer, x []float32, T int) []float32 {
	cfg := m.Config
	D := cfg.HiddenSize
	h := nn.RMSNorm(x, l.PostAttnNormW, T, D, cfg.RMSNormEps)
	y := nn.MoELayer(
		h,
		l.RouterW, l.RouterB,
		l.ExpertsGateUpW, l.ExpertsGateUpB,
		l.ExpertsDownW, l.ExpertsDownB,
		T, D, cfg.IntermediateSize, cfg.NumExperts, cfg.NumExpertsPerTok,
		7.0, 1.702,
	)
	return nn.Add(x, y)
}

// ForwardFirstLayerAttn runs only `RMSNorm -> attention -> residual` for
// layer 0, seeded with the provided [T, D] hidden state. Used by tests to
// isolate the attention+residual stage; the returned slice has shape [T, D].
func (m *Model) ForwardFirstLayerAttn(x []float32, T int) []float32 {
	cos, sin := nn.YarnRoPETables(T, m.yarnParams())
	return m.runLayerAttn(m.Layers[0], x, cos, sin, T)
}

// ForwardFinalHidden runs the full encoder (embedding + all layers + final
// norm) and returns the [T, D] hidden state fed into the classifier head.
func (m *Model) ForwardFinalHidden(ids []int) (hidden []float32, T int) {
	cfg := m.Config
	T = len(ids)
	x := nn.EmbeddingLookup(m.EmbedTokens, cfg.VocabSize, cfg.HiddenSize, ids)
	cos, sin := nn.YarnRoPETables(T, m.yarnParams())

	for _, l := range m.Layers {
		x = m.runLayerAttn(l, x, cos, sin, T)
		x = m.runLayerMLP(l, x, T)
	}
	x = nn.RMSNorm(x, m.Norm, T, cfg.HiddenSize, cfg.RMSNormEps)
	return x, T
}

// Forward runs the full encoder + classifier head and returns logits of
// shape [T, NumLabels] in row-major fp32.
func (m *Model) Forward(ids []int) (logits []float32, T int) {
	cfg := m.Config
	hidden, T := m.ForwardFinalHidden(ids)
	logits = nn.Linear(hidden, m.ScoreW, m.ScoreB, T, cfg.HiddenSize, cfg.NumLabels)
	return logits, T
}
