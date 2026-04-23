package tokenizer_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/andrew-d/openai-privacy/internal/tokenizer"
)

// repoPath returns the absolute path of a file under the repo, computed
// relative to this test source so tests work from any cwd.
func repoPath(t *testing.T, rel ...string) string {
	t.Helper()
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	repo := filepath.Clean(filepath.Join(filepath.Dir(thisFile), "..", ".."))
	parts := append([]string{repo}, rel...)
	p := filepath.Join(parts...)
	if _, err := os.Stat(p); err != nil {
		t.Fatalf("missing %s: %v", p, err)
	}
	return p
}

func loadTokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	tok, err := tokenizer.Load(repoPath(t, "model", "tokenizer.json"))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	return tok
}

type fixtureCase struct {
	Text    string   `json:"text"`
	IDs     []int    `json:"ids"`
	Offsets [][2]int `json:"offsets"`
}

func loadFixtures(t *testing.T) []fixtureCase {
	t.Helper()
	raw, err := os.ReadFile(repoPath(t, "fixtures", "tokenizer_cases.json"))
	if err != nil {
		t.Fatalf("read fixtures: %v", err)
	}
	var cases []fixtureCase
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatalf("unmarshal fixtures: %v", err)
	}
	if len(cases) == 0 {
		t.Fatal("no fixture cases")
	}
	return cases
}

func TestEncodeEmpty(t *testing.T) {
	tok := loadTokenizer(t)
	if got := tok.Encode(""); len(got) != 0 {
		t.Errorf("Encode(\"\") = %v, want []", got)
	}
}

func TestEncodeHello(t *testing.T) {
	tok := loadTokenizer(t)
	got := tok.Encode("Hello")
	want := []tokenizer.Token{{ID: 13225, Start: 0, End: 5}}
	if len(got) != len(want) {
		t.Fatalf("Encode(\"Hello\") length = %d, want %d (got %v)", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("Encode(\"Hello\")[%d] = %+v, want %+v", i, got[i], want[i])
		}
	}
}

func TestEncodeFixtures(t *testing.T) {
	tok := loadTokenizer(t)
	cases := loadFixtures(t)
	for _, c := range cases {
		c := c
		name := c.Text
		if name == "" {
			name = "<empty>"
		}
		t.Run(name, func(t *testing.T) {
			got := tok.Encode(c.Text)
			if len(got) != len(c.IDs) {
				t.Fatalf("Encode(%q) produced %d tokens, want %d\n  got = %+v\n  ids = %v\n  off = %v",
					c.Text, len(got), len(c.IDs), got, c.IDs, c.Offsets)
			}
			for i, tk := range got {
				if tk.ID != c.IDs[i] {
					t.Errorf("Encode(%q) token[%d].ID = %d, want %d (full got=%+v)", c.Text, i, tk.ID, c.IDs[i], got)
				}
				if tk.Start != c.Offsets[i][0] || tk.End != c.Offsets[i][1] {
					t.Errorf("Encode(%q) token[%d] offsets = [%d,%d), want [%d,%d)", c.Text, i, tk.Start, tk.End, c.Offsets[i][0], c.Offsets[i][1])
				}
			}
		})
	}
}
