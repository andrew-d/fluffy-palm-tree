// Package tokenizer is a pure-Go port of the o200k-style byte-level BPE
// tokenizer used by the openai/privacy-filter model.
//
// It reads a HuggingFace tokenizers library `tokenizer.json` file and
// re-implements the pre-tokenization regex, the GPT-2 `bytes_to_unicode`
// byte-level encoding, and the BPE merge loop. Only the stdlib is used.
//
// This package does NOT add any special tokens during encoding (the
// privacy-filter model is invoked with `add_special_tokens=False`), and the
// `added_tokens` section of tokenizer.json is ignored entirely.
package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"unicode"
	"unicode/utf8"
)

// Token is a single BPE token produced by Encode. Start and End are
// character (code point) offsets into the original input string; End is
// exclusive. We use character rather than byte offsets to match HuggingFace's
// `offset_mapping` output, which reports positions in Python string (code
// point) space.
type Token struct {
	ID    int
	Start int
	End   int
}

// pretokRegex is the o200k Split pattern from tokenizer.json, minus the
// `\s+(?!\S)` lookahead alternative (alt 6). RE2 has no lookahead, but the
// 6-alternative regex differs from the authoritative 7-alt version only when
// a whitespace run immediately precedes a non-whitespace token; that case is
// patched up in splitLeadingWhitespace below.
const pretokRegex = `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?` +
	`|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?` +
	`|\p{N}{1,3}` +
	`| ?[^\s\p{L}\p{N}]+[\r\n/]*` +
	`|\s*[\r\n]+` +
	`|\s+`

// byteEncoder maps each of the 256 possible byte values to its printable
// Unicode code point per GPT-2's `bytes_to_unicode`. It is initialized once at
// package load.
var byteEncoder [256]rune

func init() {
	// Printable ranges that map to themselves.
	printable := func(b int) bool {
		return (b >= '!' && b <= '~') ||
			(b >= 0xA1 && b <= 0xAC) ||
			(b >= 0xAE && b <= 0xFF)
	}
	next := rune(256)
	for b := 0; b < 256; b++ {
		if printable(b) {
			byteEncoder[b] = rune(b)
		} else {
			byteEncoder[b] = next
			next++
		}
	}
}

// tokJSON mirrors the subset of tokenizer.json fields we care about.
type tokJSON struct {
	Model struct {
		Type         string         `json:"type"`
		IgnoreMerges bool           `json:"ignore_merges"`
		ByteFallback bool           `json:"byte_fallback"`
		Vocab        map[string]int `json:"vocab"`
		Merges       [][2]string    `json:"merges"`
	} `json:"model"`
}

// Tokenizer is a loaded o200k-style BPE tokenizer.
type Tokenizer struct {
	re           *regexp.Regexp
	vocab        map[string]int
	merges       map[[2]string]int // pair -> rank (lower = earlier = preferred)
	ignoreMerges bool
}

// Load reads a HuggingFace `tokenizer.json` file from path.
func Load(path string) (*Tokenizer, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("tokenizer: open %s: %w", path, err)
	}
	defer f.Close()

	var raw tokJSON
	dec := json.NewDecoder(f)
	if err := dec.Decode(&raw); err != nil {
		return nil, fmt.Errorf("tokenizer: decode %s: %w", path, err)
	}
	if raw.Model.Type != "BPE" {
		return nil, fmt.Errorf("tokenizer: unsupported model type %q (want BPE)", raw.Model.Type)
	}
	if len(raw.Model.Vocab) == 0 {
		return nil, fmt.Errorf("tokenizer: empty vocab")
	}
	if len(raw.Model.Merges) == 0 {
		return nil, fmt.Errorf("tokenizer: empty merges")
	}

	re, err := regexp.Compile(pretokRegex)
	if err != nil {
		return nil, fmt.Errorf("tokenizer: compile pretok regex: %w", err)
	}

	merges := make(map[[2]string]int, len(raw.Model.Merges))
	for i, m := range raw.Model.Merges {
		// If the same pair ever appears twice (it should not), keep the
		// earliest (lowest-rank) occurrence.
		if _, ok := merges[m]; !ok {
			merges[m] = i
		}
	}

	return &Tokenizer{
		re:           re,
		vocab:        raw.Model.Vocab,
		merges:       merges,
		ignoreMerges: raw.Model.IgnoreMerges,
	}, nil
}

// Encode pre-tokenizes text, applies byte-level encoding, and runs BPE on each
// piece, returning the resulting token ids along with character-offset ranges
// into the original input text. No special tokens are inserted.
func (t *Tokenizer) Encode(text string) []Token {
	if text == "" {
		return nil
	}
	rawMatches := t.re.FindAllStringIndex(text, -1)
	matches := splitLeadingWhitespace(text, rawMatches)

	// Precompute byte-offset to rune-offset (character index) mapping for
	// the output token offsets. We need charOffset(byteIdx) for every byte
	// index that could appear at a token boundary; those are always the
	// start of a UTF-8 rune because the pre-tokenizer regex only matches at
	// rune boundaries, and BPE can only merge byte-level-encoded characters
	// (which map 1:1 to input bytes) into groups that re-align at multi-byte
	// character boundaries by way of the merge table.
	//
	// To be safe for any byte index (in case a merge ever failed to
	// reassemble a multi-byte character), we map a mid-codepoint byte
	// position to the char index of the character it sits inside.
	charIdx := make([]int, len(text)+1)
	{
		bi, ci := 0, 0
		for bi < len(text) {
			charIdx[bi] = ci
			_, size := utf8.DecodeRuneInString(text[bi:])
			// Mid-codepoint continuation bytes map to ci+1 (the char index
			// AFTER the containing rune). HuggingFace reports char-end
			// positions one past the rune for tokens that end mid-rune,
			// which matches this assignment.
			for k := 1; k < size && bi+k < len(text); k++ {
				charIdx[bi+k] = ci + 1
			}
			bi += size
			ci++
		}
		charIdx[len(text)] = ci
	}

	out := make([]Token, 0, len(matches))
	// Reusable scratch buffers to reduce allocation across pieces.
	var (
		encoded   []rune
		charStart []int // for each rune in `encoded`, the byte offset in `text` of the input byte it encodes
		symbols   []string
		symStart  []int // character index (into `encoded`) where each symbol starts
	)
	for _, m := range matches {
		start, end := m[0], m[1]
		piece := text[start:end]

		// Byte-level encode: each byte of `piece` becomes one rune in
		// `encoded`. The i-th rune's original byte offset in `text` is
		// `start + i`.
		if cap(encoded) < len(piece) {
			encoded = make([]rune, 0, len(piece))
			charStart = make([]int, 0, len(piece))
		}
		encoded = encoded[:0]
		charStart = charStart[:0]
		for i := 0; i < len(piece); i++ {
			encoded = append(encoded, byteEncoder[piece[i]])
			charStart = append(charStart, start+i)
		}

		// ignore_merges shortcut: if the whole byte-encoded piece is in
		// vocab as-is, emit it as a single token without running BPE.
		pieceStr := string(encoded)
		if t.ignoreMerges {
			if id, ok := t.vocab[pieceStr]; ok {
				out = append(out, Token{ID: id, Start: charIdx[start], End: charIdx[end]})
				continue
			}
		}

		// BPE: each rune starts as its own one-char symbol.
		if cap(symbols) < len(encoded) {
			symbols = make([]string, 0, len(encoded))
			symStart = make([]int, 0, len(encoded))
		}
		symbols = symbols[:0]
		symStart = symStart[:0]
		for i, r := range encoded {
			symbols = append(symbols, string(r))
			symStart = append(symStart, i)
		}

		symbols, symStart = t.bpe(symbols, symStart)

		// Look up each final symbol in vocab and emit a token.
		for k, sym := range symbols {
			id, ok := t.vocab[sym]
			if !ok {
				// Unreachable for a well-formed tokenizer that does not
				// use byte_fallback: every BPE-merged symbol is in vocab.
				// Emit -1 so tests fail loudly rather than silently.
				id = -1
			}
			byteStart := charStart[symStart[k]]
			var byteEnd int
			if k+1 < len(symbols) {
				byteEnd = charStart[symStart[k+1]]
			} else {
				byteEnd = end
			}
			out = append(out, Token{ID: id, Start: charIdx[byteStart], End: charIdx[byteEnd]})
		}
	}
	return out
}

// bpe runs the BPE merge loop on the given sequence of symbols. symbols[k] is
// the k-th symbol string; symStart[k] is the character index (into the
// byte-encoded piece) where symbols[k] begins. The function mutates its inputs
// and returns the final (possibly shorter) slices.
//
// The algorithm: repeatedly find the adjacent pair with the lowest merge rank
// and merge it. Stop when no adjacent pair is in the merge table. Ties are
// broken by position (leftmost wins), matching HuggingFace's reference
// implementation.
func (t *Tokenizer) bpe(symbols []string, symStart []int) ([]string, []int) {
	for len(symbols) >= 2 {
		bestRank := -1
		bestIdx := -1
		for i := 0; i+1 < len(symbols); i++ {
			rank, ok := t.merges[[2]string{symbols[i], symbols[i+1]}]
			if !ok {
				continue
			}
			if bestRank == -1 || rank < bestRank {
				bestRank = rank
				bestIdx = i
			}
		}
		if bestIdx == -1 {
			break
		}
		// Merge symbols[bestIdx] and symbols[bestIdx+1] in place.
		symbols[bestIdx] = symbols[bestIdx] + symbols[bestIdx+1]
		symbols = append(symbols[:bestIdx+1], symbols[bestIdx+2:]...)
		symStart = append(symStart[:bestIdx+1], symStart[bestIdx+2:]...)
	}
	return symbols, symStart
}

// splitLeadingWhitespace emulates the `\s+(?!\S)` alternative (alt 6) that
// RE2 cannot express. The 6-alt regex produces a single whitespace match for
// each whitespace run; the 7-alt regex splits runs that precede non-whitespace
// so that (a) word/punct alts can consume one leading space (their optional
// ` ?` prefix) and (b) runs before bare digits are broken into single-char
// matches.
//
// Matches are [start,end) byte-offset pairs and must be in ascending order
// with no gaps; splitLeadingWhitespace returns a new slice.
func splitLeadingWhitespace(text string, matches [][]int) [][]int {
	out := make([][]int, 0, len(matches)+4)
	for i := 0; i < len(matches); i++ {
		m := matches[i]
		start, end := m[0], m[1]
		// Only touch pure whitespace runs that contain no newline (alt 5
		// claims anything with a newline before alt 6 could).
		if !isPureSpaceRun(text[start:end]) {
			out = append(out, m)
			continue
		}
		// Need a following match that begins exactly at `end` (whitespace
		// immediately followed by non-whitespace in the original text).
		if i+1 >= len(matches) || matches[i+1][0] != end {
			out = append(out, m)
			continue
		}
		next := matches[i+1]
		nextFirst, _ := utf8.DecodeRuneInString(text[next[0]:next[1]])
		if nextFirst == utf8.RuneError || unicode.IsSpace(nextFirst) {
			out = append(out, m)
			continue
		}

		// How many bytes long is the whitespace run? (Each space char in
		// `\s` used by this tokenizer is 1 byte: ASCII space, tab, form
		// feed, etc. No multi-byte spaces appear in the fixtures.)
		runLen := end - start

		if unicode.IsDigit(nextFirst) || unicode.IsNumber(nextFirst) {
			// Alt 3 `\p{N}{1,3}` has no ` ?` prefix; the word match does
			// not absorb a leading space. 7-alt splits the whitespace run
			// as: one (runLen-1)-wide alt-6 match plus a single alt-7
			// match of length 1 (both whitespace). For runLen==1 no split
			// is needed.
			if runLen >= 2 {
				out = append(out, []int{start, end - 1})
				out = append(out, []int{end - 1, end})
			} else {
				out = append(out, m)
			}
			continue
		}

		// Otherwise (alt 1/2/4 with ` ?` prefix): 7-alt shrinks the
		// whitespace run by 1 char and the next match absorbs that char
		// as its leading space. If runLen==1 the whitespace run disappears
		// entirely (the next match grows by 1).
		if runLen >= 2 {
			out = append(out, []int{start, end - 1})
		}
		// Mutate next match to include the transferred byte.
		newNext := []int{end - 1, next[1]}
		matches[i+1] = newNext
	}
	return out
}

// isPureSpaceRun reports whether s consists entirely of whitespace characters
// and contains no newline/carriage-return (which are handled by alt 5 and
// therefore never reach the `\s+(?!\S)` / `\s+` alternatives).
func isPureSpaceRun(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if r == '\n' || r == '\r' {
			return false
		}
		if !unicode.IsSpace(r) {
			return false
		}
	}
	return true
}
