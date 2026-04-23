// Package integration cross-checks the Go privacyfilter against the Python
// HuggingFace reference implementation.
//
// Each pair of `testdata/<name>.txt` (raw input) and `testdata/<name>.json`
// (expected entities, produced by scripts/generate_testdata.py) is run through
// Go's privacyfilter.Classify and compared against the stored entities.
package integration_test

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/andrew-d/openai-privacy"
)

type expectedEntity struct {
	EntityGroup string  `json:"entity_group"`
	Word        string  `json:"word"`
	Start       int     `json:"start"`
	End         int     `json:"end"`
	Score       float64 `json:"score"`
}

type testCase struct {
	Text     string           `json:"text"`
	Entities []expectedEntity `json:"entities"`
}

// scoreTolerance bounds |go - python| on the mean-softmax entity score. For
// tokens where the model is nearly saturated this should be tiny; fp32
// accumulation drift across 8 layers explains most of what we do see.
const scoreTolerance = 2e-3

var (
	loadOnce      sync.Once
	sharedModel   *privacyfilter.Model
	sharedLoadErr error
)

func loadModel(t *testing.T) *privacyfilter.Model {
	t.Helper()
	loadOnce.Do(func() {
		sharedModel, sharedLoadErr = privacyfilter.LoadModel("../model")
	})
	if sharedLoadErr != nil {
		t.Fatalf("LoadModel: %v", sharedLoadErr)
	}
	return sharedModel
}

func TestAgainstPythonReference(t *testing.T) {
	inputs, err := filepath.Glob("testdata/*.txt")
	if err != nil {
		t.Fatalf("glob: %v", err)
	}
	if len(inputs) == 0 {
		t.Skip("no testdata/*.txt cases yet — run scripts/generate_testdata.py after adding them")
	}

	model := loadModel(t)

	for _, txtPath := range inputs {
		name := strings.TrimSuffix(filepath.Base(txtPath), ".txt")
		t.Run(name, func(t *testing.T) {
			raw, err := os.ReadFile(txtPath)
			if err != nil {
				t.Fatalf("read %s: %v", txtPath, err)
			}
			text := string(raw)
			text = strings.TrimSuffix(text, "\n")

			jsonPath := strings.TrimSuffix(txtPath, ".txt") + ".json"
			jsonBytes, err := os.ReadFile(jsonPath)
			if err != nil {
				t.Fatalf("read %s: %v (did you run scripts/generate_testdata.py?)", jsonPath, err)
			}
			var want testCase
			if err := json.Unmarshal(jsonBytes, &want); err != nil {
				t.Fatalf("parse %s: %v", jsonPath, err)
			}
			if want.Text != text {
				t.Fatalf("%s: text in JSON does not match .txt contents", name)
			}

			got, err := model.Classify(text)
			if err != nil {
				t.Fatalf("Classify: %v", err)
			}

			if len(got) != len(want.Entities) {
				t.Errorf("entity count mismatch: got %d, want %d", len(got), len(want.Entities))
				t.Logf("  got:  %s", formatEntities(got))
				t.Logf("  want: %s", formatExpected(want.Entities))
				return
			}

			for i, g := range got {
				w := want.Entities[i]
				if g.EntityGroup != w.EntityGroup {
					t.Errorf("entity %d: group got %q, want %q", i, g.EntityGroup, w.EntityGroup)
				}
				if g.Word != w.Word {
					t.Errorf("entity %d: word got %q, want %q", i, g.Word, w.Word)
				}
				if g.Start != w.Start || g.End != w.End {
					t.Errorf("entity %d: offsets got [%d,%d], want [%d,%d]", i, g.Start, g.End, w.Start, w.End)
				}
				if diff := math.Abs(float64(g.Score) - w.Score); diff > scoreTolerance {
					t.Errorf("entity %d: score got %.6f, want %.6f (diff %.2e > %.2e)", i, g.Score, w.Score, diff, scoreTolerance)
				}
			}
		})
	}
}

func formatEntities(es []privacyfilter.Entity) string {
	parts := make([]string, len(es))
	for i, e := range es {
		parts[i] = formatOne(e.EntityGroup, e.Word, e.Start, e.End, float64(e.Score))
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

func formatExpected(es []expectedEntity) string {
	parts := make([]string, len(es))
	for i, e := range es {
		parts[i] = formatOne(e.EntityGroup, e.Word, e.Start, e.End, e.Score)
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

func formatOne(group, word string, start, end int, score float64) string {
	return fmt.Sprintf("%s(%q)[%d:%d]@%.4f", group, word, start, end, score)
}
