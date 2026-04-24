package privacyfilter_test

import (
	"strings"
	"testing"

	"github.com/andrew-d/openai-privacy"
)

// benchModel caches the loaded model across benchmarks in this file so the
// ~5s weight-decode cost is paid only once per `go test` invocation.
var benchModel *privacyfilter.Model

func getBenchModel(tb testing.TB) *privacyfilter.Model {
	tb.Helper()
	if benchModel != nil {
		return benchModel
	}
	m, err := privacyfilter.LoadModel("./model")
	if err != nil {
		tb.Fatalf("LoadModel: %v", err)
	}
	benchModel = m
	return benchModel
}

const benchParagraph = "My name is Sherlock Holmes and my email is sherlock.holmes@scotlandyard.uk. " +
	"Please contact me at 555-123-4567 or at my home, 221B Baker Street, London. " +
	"My account number is 000123456789 and my password is hunter2. "

func benchInput(repeat int) string {
	return strings.Repeat(benchParagraph, repeat)
}

func benchmarkClassify(b *testing.B, repeat int) {
	m := getBenchModel(b)
	text := benchInput(repeat)

	// Warm up once so the first-iteration cost (go runtime cache fills,
	// allocator heating, any lazily-initialized state) does not skew N=1.
	if _, err := m.Classify(text); err != nil {
		b.Fatalf("warmup Classify: %v", err)
	}

	// Tokens/char counts for reporting; derived from the tokenizer by way of
	// one extra Classify — negligible cost relative to the benchmark body.
	tokens := m.TokenCount(text)
	chars := len([]rune(text))
	bytes := len(text)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := m.Classify(text)
		if err != nil {
			b.Fatalf("Classify: %v", err)
		}
	}
	b.StopTimer()

	secs := b.Elapsed().Seconds()
	if secs > 0 {
		b.ReportMetric(float64(tokens*b.N)/secs, "tokens/sec")
		b.ReportMetric(float64(chars*b.N)/secs, "chars/sec")
		b.ReportMetric(float64(bytes*b.N)/secs, "bytes/sec")
	}
	b.Logf("input: %d tokens, %d chars, %d bytes; %d iter(s) in %s",
		tokens, chars, bytes, b.N, b.Elapsed())
}

func BenchmarkClassifyShort(b *testing.B)  { benchmarkClassify(b, 1) }
func BenchmarkClassifyMedium(b *testing.B) { benchmarkClassify(b, 5) }
func BenchmarkClassifyLong(b *testing.B)   { benchmarkClassify(b, 20) }
