// Package privacyfilter provides a pure-Go inference implementation of the
// openai/privacy-filter token-classification model.
//
// Only the Go standard library is used; the model weights and tokenizer are
// read directly from the files shipped in the HuggingFace repository.
package privacyfilter

import (
	"path/filepath"

	"github.com/andrew-d/openai-privacy/internal/privatemodel"
	"github.com/andrew-d/openai-privacy/internal/tokenizer"
)

// Entity is a single PII span detected in the input text.
type Entity struct {
	// EntityGroup is the PII category, e.g. "private_person".
	EntityGroup string
	// Score is the mean softmax probability (at each token's argmax label) of
	// the constituent tokens.
	Score float32
	// Word is the substring of the input text covered by the span. It is
	// extracted by character (rune) offsets and therefore includes any
	// leading whitespace that the tokenizer pre-tokenized into the first
	// token of the span (e.g. " Harry Potter" for the Harry Potter example).
	Word string
	// Start and End are CHARACTER (rune) offsets into the input text. This
	// matches the offsets returned by HuggingFace's fast tokenizer and by
	// the tokenizer package in this module. End is exclusive.
	Start int
	End   int
}

// Model holds the loaded weights, configuration, and tokenizer for the
// openai/privacy-filter model.
type Model struct {
	tok   *tokenizer.Tokenizer
	inner *privatemodel.Model
}

// LoadModel reads the privacy-filter model from dir.
//
// dir must contain the files shipped in the openai/privacy-filter HuggingFace
// repository: config.json, tokenizer.json, and model.safetensors.
func LoadModel(dir string) (*Model, error) {
	tok, err := tokenizer.Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		return nil, err
	}
	inner, err := privatemodel.LoadFromSafetensors(dir)
	if err != nil {
		return nil, err
	}
	return &Model{tok: tok, inner: inner}, nil
}

// TokenCount returns the number of model tokens that would be produced by
// tokenizing text. Useful for estimating throughput or cost.
func (m *Model) TokenCount(text string) int {
	return len(m.tok.Encode(text))
}

// Classify runs the model on text and returns the detected PII spans,
// aggregated from the per-token BIOES label predictions.
func (m *Model) Classify(text string) ([]Entity, error) {
	toks := m.tok.Encode(text)
	if len(toks) == 0 {
		return nil, nil
	}
	ids := make([]int, len(toks))
	for i, t := range toks {
		ids[i] = t.ID
	}

	logits, T := m.inner.Forward(ids)
	numLabels := m.inner.Config.NumLabels
	id2label := m.inner.Config.ID2Label

	labels := make([]string, T)
	scores := make([]float32, T)
	for t := 0; t < T; t++ {
		row := logits[t*numLabels : (t+1)*numLabels]
		arg, prob := softmaxArgmaxAndProb(row)
		labels[t] = id2label[arg]
		scores[t] = prob
	}
	return aggregateBIOES(toks, labels, scores, text), nil
}
