package privacyfilter

import (
	"math"
	"testing"

	"github.com/andrew-d/openai-privacy/internal/tokenizer"
)

// mkTok builds a tokenizer.Token with the given character offsets. The ID is
// irrelevant for aggregation, so it is always 0.
func mkTok(start, end int) tokenizer.Token {
	return tokenizer.Token{ID: 0, Start: start, End: end}
}

// floatClose reports whether a and b are within an absolute tolerance.
func floatClose(a, b, tol float32) bool {
	d := float64(a - b)
	return math.Abs(d) <= float64(tol)
}

// TestAggregateBIOES covers the walk-through cases enumerated in the spec.
func TestAggregateBIOES(t *testing.T) {
	type wantEntity struct {
		group      string
		word       string
		start, end int
		score      float32
	}
	tests := []struct {
		name   string
		text   string
		tokens []tokenizer.Token
		labels []string
		scores []float32
		want   []wantEntity
	}{
		{
			name:   "B-E pair",
			text:   "My name is Sherlock Holmes.",
			tokens: []tokenizer.Token{mkTok(0, 2), mkTok(2, 7), mkTok(7, 10), mkTok(10, 19), mkTok(19, 26), mkTok(26, 27)},
			labels: []string{"O", "O", "O", "B-private_person", "E-private_person", "O"},
			scores: []float32{0.9, 0.9, 0.9, 0.8, 1.0, 0.9},
			want: []wantEntity{
				{group: "private_person", word: " Sherlock Holmes", start: 10, end: 26, score: 0.9},
			},
		},
		{
			name:   "B-I-E triple",
			text:   "X aaa bbb ccc.",
			tokens: []tokenizer.Token{mkTok(0, 1), mkTok(1, 5), mkTok(5, 9), mkTok(9, 13), mkTok(13, 14)},
			labels: []string{"O", "B-private_email", "I-private_email", "E-private_email", "O"},
			scores: []float32{0.9, 1.0, 1.0, 1.0, 0.9},
			want: []wantEntity{
				{group: "private_email", word: " aaa bbb ccc", start: 1, end: 13, score: 1.0},
			},
		},
		{
			name:   "S singleton",
			text:   "Hi Bob.",
			tokens: []tokenizer.Token{mkTok(0, 2), mkTok(2, 6), mkTok(6, 7)},
			labels: []string{"O", "S-private_person", "O"},
			scores: []float32{0.9, 0.95, 0.9},
			want: []wantEntity{
				{group: "private_person", word: " Bob", start: 2, end: 6, score: 0.95},
			},
		},
		{
			name:   "back-to-back spans of different types",
			text:   "ab cd ef gh",
			tokens: []tokenizer.Token{mkTok(0, 2), mkTok(2, 5), mkTok(5, 8), mkTok(8, 11)},
			labels: []string{"B-private_person", "E-private_person", "B-private_email", "E-private_email"},
			scores: []float32{1.0, 1.0, 1.0, 1.0},
			want: []wantEntity{
				{group: "private_person", word: "ab cd", start: 0, end: 5, score: 1.0},
				{group: "private_email", word: " ef gh", start: 5, end: 11, score: 1.0},
			},
		},
		{
			name: "mismatched I self-heals to a new span",
			// An I-X with no matching open span becomes a fresh span starting
			// at that token.
			text:   "a b c",
			tokens: []tokenizer.Token{mkTok(0, 1), mkTok(1, 3), mkTok(3, 5)},
			labels: []string{"O", "I-private_person", "E-private_person"},
			scores: []float32{0.9, 0.8, 1.0},
			want: []wantEntity{
				{group: "private_person", word: " b c", start: 1, end: 5, score: 0.9},
			},
		},
		{
			name: "trailing B with no E still emits",
			// A B-X at end-of-sequence (or not closed by E/O) still emits as a
			// one-token span.
			text:   "hi Bob",
			tokens: []tokenizer.Token{mkTok(0, 2), mkTok(2, 6)},
			labels: []string{"O", "B-private_person"},
			scores: []float32{0.9, 0.77},
			want: []wantEntity{
				{group: "private_person", word: " Bob", start: 2, end: 6, score: 0.77},
			},
		},
		{
			name: "stray E self-heals to a single-token span",
			// E-X with no matching open span should still produce an entity so
			// we never lose a high-confidence prediction.
			text:   "ab",
			tokens: []tokenizer.Token{mkTok(0, 2)},
			labels: []string{"E-private_person"},
			scores: []float32{0.91},
			want: []wantEntity{
				{group: "private_person", word: "ab", start: 0, end: 2, score: 0.91},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := aggregateBIOES(tt.tokens, tt.labels, tt.scores, tt.text)
			if len(got) != len(tt.want) {
				t.Fatalf("got %d entities, want %d: %+v", len(got), len(tt.want), got)
			}
			for i, e := range got {
				w := tt.want[i]
				if e.EntityGroup != w.group {
					t.Errorf("entity %d: group %q, want %q", i, e.EntityGroup, w.group)
				}
				if e.Word != w.word {
					t.Errorf("entity %d: word %q, want %q", i, e.Word, w.word)
				}
				if e.Start != w.start || e.End != w.end {
					t.Errorf("entity %d: [%d,%d], want [%d,%d]", i, e.Start, e.End, w.start, w.end)
				}
				if !floatClose(e.Score, w.score, 1e-5) {
					t.Errorf("entity %d: score %.5f, want %.5f", i, e.Score, w.score)
				}
			}
		})
	}
}

// TestSoftmaxArgmaxAndProb exercises the tiny numerically-stable softmax
// helper used during per-token label assignment.
func TestSoftmaxArgmaxAndProb(t *testing.T) {
	// Uniform logits: all classes tie at 1/N; the implementation returns the
	// first index on ties and prob ~= 1/N.
	idx, p := softmaxArgmaxAndProb([]float32{0, 0, 0, 0})
	if idx != 0 {
		t.Errorf("uniform: argmax %d, want 0", idx)
	}
	if !floatClose(p, 0.25, 1e-6) {
		t.Errorf("uniform: prob %.6f, want 0.25", p)
	}

	// Sharp peak: large gap should give near-1.0 probability.
	idx, p = softmaxArgmaxAndProb([]float32{0, 20, 0})
	if idx != 1 {
		t.Errorf("peak: argmax %d, want 1", idx)
	}
	if p < 0.999 {
		t.Errorf("peak: prob %.6f, want >= 0.999", p)
	}

	// Two-way tie: first index wins, prob is ~0.5.
	idx, p = softmaxArgmaxAndProb([]float32{1, 1, -10})
	if idx != 0 {
		t.Errorf("tie: argmax %d, want 0", idx)
	}
	if !floatClose(p, 0.5, 1e-4) {
		t.Errorf("tie: prob %.6f, want ~0.5", p)
	}
}
