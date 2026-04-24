package privacyfilter_test

import (
	"sync"
	"testing"

	"github.com/andrew-d/openai-privacy"
)

// loadOnce loads the model exactly once per test binary invocation so the two
// end-to-end tests do not each pay the ~5s weight-decode cost.
var (
	loadOnce      sync.Once
	sharedModel   *privacyfilter.Model
	sharedLoadErr error
)

func loadSharedModel(t *testing.T) *privacyfilter.Model {
	t.Helper()
	loadOnce.Do(func() {
		sharedModel, sharedLoadErr = privacyfilter.LoadModel("./model")
	})
	if sharedLoadErr != nil {
		t.Fatalf("LoadModel: %v", sharedLoadErr)
	}
	return sharedModel
}

// TestSherlockHolmesEndToEnd is the top-level integration test: it loads the
// model and runs it on the canonical demo sentence. The expected output comes
// from the transformers Python reference implementation (with BIOES
// aggregation) captured in fixtures/reference_outputs.json.
func TestSherlockHolmesEndToEnd(t *testing.T) {
	model := loadSharedModel(t)

	text := "My name is Sherlock Holmes and my email is sherlock.holmes@scotlandyard.uk."
	entities, err := model.Classify(text)
	if err != nil {
		t.Fatalf("Classify: %v", err)
	}

	want := []privacyfilter.Entity{
		{EntityGroup: "private_person", Word: " Sherlock Holmes", Start: 10, End: 26},
		{EntityGroup: "private_email", Word: " sherlock.holmes@scotlandyard.uk", Start: 42, End: 74},
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

// TestAliceSmithEndToEnd is a second end-to-end case: a single merged
// private_person span (the reference "simple" aggregation splits this into
// two because E-private_person closes it, but our strict BIOES walk keeps
// B+E together).
func TestAliceSmithEndToEnd(t *testing.T) {
	model := loadSharedModel(t)

	text := "My name is Alice Smith"
	entities, err := model.Classify(text)
	if err != nil {
		t.Fatalf("Classify: %v", err)
	}

	want := []privacyfilter.Entity{
		{EntityGroup: "private_person", Word: " Alice Smith", Start: 10, End: 22},
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
