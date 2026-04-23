package privacyfilter_test

import (
	"testing"

	"github.com/andrew-d/openai-privacy"
)

// TestHarryPotterEndToEnd is the top-level integration test: it loads the
// model and runs it on the example from the model card. The expected output
// comes from the transformers Python reference implementation (with BIOES
// aggregation) captured in fixtures/reference_outputs.json.
func TestHarryPotterEndToEnd(t *testing.T) {
	model, err := privacyfilter.LoadModel("./model")
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	text := "My name is Harry Potter and my email is harry.potter@hogwarts.edu."
	entities, err := model.Classify(text)
	if err != nil {
		t.Fatalf("Classify: %v", err)
	}

	want := []privacyfilter.Entity{
		{EntityGroup: "private_person", Word: " Harry Potter", Start: 10, End: 23},
		{EntityGroup: "private_email", Word: " harry.potter@hogwarts.edu", Start: 39, End: 65},
	}
	if len(entities) != len(want) {
		t.Fatalf("got %d entities, want %d: %+v", len(entities), len(want), entities)
	}
	for i, e := range entities {
		w := want[i]
		if e.EntityGroup != w.EntityGroup {
			t.Errorf("entity %d: got group %q, want %q", i, e.EntityGroup, w.EntityGroup)
		}
		if e.Word != w.Word {
			t.Errorf("entity %d: got word %q, want %q", i, e.Word, w.Word)
		}
		if e.Start != w.Start || e.End != w.End {
			t.Errorf("entity %d: got [%d,%d], want [%d,%d]", i, e.Start, e.End, w.Start, w.End)
		}
		if e.Score < 0.99 {
			t.Errorf("entity %d: score %.4f < 0.99", i, e.Score)
		}
	}
}
