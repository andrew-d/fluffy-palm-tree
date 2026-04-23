// Package privacyfilter provides a pure-Go inference implementation of the
// openai/privacy-filter token-classification model.
//
// Only the Go standard library is used; the model weights and tokenizer are
// read directly from the files shipped in the HuggingFace repository.
package privacyfilter

import "errors"

// Entity is a single PII span detected in the input text.
type Entity struct {
	// EntityGroup is the PII category, e.g. "private_person".
	EntityGroup string
	// Score is the mean softmax probability of the constituent tokens.
	Score float32
	// Word is the substring of the input text covered by the span.
	Word string
	// Start and End are byte offsets into the original input text.
	Start int
	End   int
}

// Model holds loaded weights and configuration for the privacy-filter model.
// Not yet implemented; LoadModel will return an error until the library is
// complete.
type Model struct{}

// LoadModel reads the privacy-filter model from dir.
//
// dir must contain the files shipped in the openai/privacy-filter HuggingFace
// repository: config.json, tokenizer.json, and model.safetensors.
func LoadModel(dir string) (*Model, error) {
	return nil, errors.New("privacyfilter: not implemented")
}

// Classify runs the model on text and returns the detected PII spans.
func (m *Model) Classify(text string) ([]Entity, error) {
	return nil, errors.New("privacyfilter: not implemented")
}
