package privacyfilter

import (
	"math"
	"strings"

	"github.com/andrew-d/openai-privacy/internal/tokenizer"
)

// softmaxArgmaxAndProb returns the index of the largest logit and the softmax
// probability at that index. It uses the standard numerically-stable form:
// subtract the max, exponentiate, and normalize. The intermediate sum is kept
// in float64 so very peaky logits do not lose precision when we divide.
func softmaxArgmaxAndProb(logits []float32) (int, float32) {
	if len(logits) == 0 {
		return -1, 0
	}
	maxV := logits[0]
	maxI := 0
	for i, v := range logits {
		if v > maxV {
			maxV = v
			maxI = i
		}
	}
	var sum float64
	for _, v := range logits {
		sum += math.Exp(float64(v - maxV))
	}
	// exp(logits[maxI]-maxV) == exp(0) == 1.
	return maxI, float32(1.0 / sum)
}

// aggregateBIOES walks per-token BIOES labels left-to-right and emits one
// Entity per completed span (strict BIOES: B+I*+E or a lone S).
//
// See the package doc / architecture.md "Aggregation" section for the exact
// transitions we implement. The function is defensive about mismatches (a
// stray I-X or E-X with no matching open span starts/emits a fresh one) so a
// misbehaving model never silently drops a high-confidence prediction.
func aggregateBIOES(tokens []tokenizer.Token, labels []string, scores []float32, text string) []Entity {
	if len(tokens) == 0 {
		return nil
	}
	runes := []rune(text)

	var out []Entity
	// Open span state: openType == "" means no span is open.
	openType := ""
	var bufIdx []int

	// emit flushes the currently-open span (if any) to the output slice.
	emit := func() {
		if openType == "" {
			return
		}
		first, last := bufIdx[0], bufIdx[len(bufIdx)-1]
		rs, re := tokens[first].Start, tokens[last].End
		// Mean score over the span. scores already reflects the argmax-class
		// softmax probability per token, so this matches the HuggingFace
		// reference behaviour.
		var sum float64
		for _, i := range bufIdx {
			sum += float64(scores[i])
		}
		out = append(out, Entity{
			EntityGroup: openType,
			Score:       float32(sum / float64(len(bufIdx))),
			Word:        sliceRunes(runes, rs, re),
			Start:       rs,
			End:         re,
		})
		openType = ""
		bufIdx = bufIdx[:0]
	}

	// open starts a new span of type typ containing only token t.
	open := func(typ string, t int) {
		openType = typ
		bufIdx = append(bufIdx[:0], t)
	}

	for t, lbl := range labels {
		tag, typ := splitBIOES(lbl)
		switch tag {
		case "O":
			emit()
		case "S":
			emit()
			open(typ, t)
			emit()
		case "B":
			emit()
			open(typ, t)
		case "I":
			if openType == typ {
				bufIdx = append(bufIdx, t)
			} else {
				// Mismatch: treat as the start of a new span.
				emit()
				open(typ, t)
			}
		case "E":
			if openType == typ {
				bufIdx = append(bufIdx, t)
				emit()
			} else {
				// Mismatch: emit the previous span (if any) and then emit a
				// single-token span for this stray E.
				emit()
				open(typ, t)
				emit()
			}
		default:
			// Unknown label format — close out any open span and skip.
			emit()
		}
	}
	emit()
	return out
}

// splitBIOES splits a label like "B-private_person" into ("B", "private_person").
// "O" returns ("O", ""). An unknown or malformed label returns ("", "") so the
// caller can treat it as an O.
func splitBIOES(lbl string) (tag, typ string) {
	if lbl == "O" {
		return "O", ""
	}
	i := strings.IndexByte(lbl, '-')
	if i <= 0 || i == len(lbl)-1 {
		return "", ""
	}
	tag = lbl[:i]
	typ = lbl[i+1:]
	switch tag {
	case "B", "I", "E", "S":
		return tag, typ
	}
	return "", ""
}

// sliceRunes returns text[start:end] where start and end are character (rune)
// offsets into text. The HuggingFace tokenizer and fixtures use character
// offsets, so this matches the reference implementation.
func sliceRunes(runes []rune, start, end int) string {
	if start < 0 {
		start = 0
	}
	if end > len(runes) {
		end = len(runes)
	}
	if start >= end {
		return ""
	}
	return string(runes[start:end])
}
