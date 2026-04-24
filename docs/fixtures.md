# Test fixtures and how to regenerate them

This repo has two kinds of test fixtures, both produced by running the
HuggingFace Python reference model on hand-picked inputs and saving the
outputs. The Go tests compare their own computations against these saved
outputs, so every layer of the pure-Go reimplementation is pinned to
numerically match the reference.

- `fixtures/` — intermediate tensors, tokenizer golden cases, and end-to-end
  reference outputs for a single small input. These drive the unit tests in
  `internal/nn`, `internal/privatemodel`, `internal/safetensors`,
  `internal/tokenizer`, and the top-level `privacyfilter_test.go`.
- `integration/testdata/` — `.txt` input + `.json` expected-output pairs for
  the black-box integration test in `integration/integration_test.go`.

The Go tests do **not** require Python or the HuggingFace libraries. Python
is only needed to (re)generate the fixtures from scratch.

## Setup

One-time, from the repo root:

```
python3 -m venv .venv
source .venv/bin/activate
pip install torch transformers numpy
```

Every regeneration command below assumes `source .venv/bin/activate` has been
run. The scripts expect the model files to be present at `./model/` (see the
top-level README for how to fetch them from HuggingFace).

## Generator scripts

All four live in `scripts/` and are independent — you only need to run the
ones whose outputs you invalidated.

### `scripts/reference_classify.py` → `fixtures/reference_outputs.json`

Runs the reference model on three short sentences and captures both the
per-token output (input_ids, offsets, label ids, softmax scores) and the
HuggingFace pipeline's aggregated spans. The top-level
`TestSherlockHolmesEndToEnd` / `TestAliceSmithEndToEnd` end-to-end tests
assert against these values after applying the Go package's BIOES aggregation.

```
python3 scripts/reference_classify.py
```

### `scripts/dump_fixtures.py` → `fixtures/fixtures.json` + `fixtures/*.f32.bin`

Runs the reference model on one short sentence and uses PyTorch forward hooks
to capture every intermediate tensor: embedding, per-layer pre-attention
norm, attention output, pre-MLP norm, router indices/scores/logits, MLP
output, layer output, final norm, logits, and probs. `fixtures.json` holds
the tokenization + tensor metadata (shape, filename); each `*.f32.bin` is a
raw little-endian float32 dump of its tensor.

The per-layer tests in `internal/nn/*_test.go` and
`internal/privatemodel/model_test.go` load these and assert the Go
implementation reproduces each tensor within a tight float tolerance. If you
change the input text you must regenerate these — the `.bin` shapes are
sequence-length-dependent and the tests check shapes explicitly.

```
python3 scripts/dump_fixtures.py
```

### `scripts/dump_tokenizer_fixtures.py` → `fixtures/tokenizer_cases.json`

Runs the HuggingFace fast tokenizer over a hand-picked set of edge cases
(empty, ASCII, punctuation, digits, whitespace/newlines, Unicode letters and
marks, URLs, the Sherlock Holmes sentence). Each case records the input
string plus the expected token IDs and character offsets.
`internal/tokenizer/tokenizer_test.go` replays every case against the Go
tokenizer and diffs the output.

```
python3 scripts/dump_tokenizer_fixtures.py
```

### `scripts/generate_testdata.py` → `integration/testdata/*.json`

Walks every `integration/testdata/*.txt`, runs the reference model, applies
the same BIOES aggregation the Go package uses, and writes a matching
`*.json` next to each `.txt`. This is the "black-box" test data the
integration test reads.

To add a new case: drop a `<name>.txt` in `integration/testdata/` and rerun
the script. To change an existing case: edit the `.txt` and rerun.

```
python3 scripts/generate_testdata.py
```

## When to regenerate

- **Changed the demo / reference input text** (e.g. swapped a fictional
  character across the codebase): regenerate all four, because the reference
  text appears in `reference_classify.py`, `dump_fixtures.py`, and
  `dump_tokenizer_fixtures.py`, and several integration `.txt` files use the
  same names.
- **Changed the model weights** at `./model/`: regenerate all four; tensor
  values and scores will have shifted.
- **Added an integration test case**: only `generate_testdata.py` needs
  rerunning.
- **Added a new tokenizer edge case to `CASES`** in
  `dump_tokenizer_fixtures.py`: just that one.

After regeneration, rerun the Go suite (single-package serial mode — see
`CLAUDE.md` for why):

```
go test -p=1 ./...
```

## Notes and gotchas

- The `fixtures/*.f32.bin` files are raw float32. Size on disk =
  `4 * product(shape)` bytes; `fixtures.json` carries the shape.
- `reference_outputs.json` uses HuggingFace's `aggregation_strategy="simple"`,
  which does **not** merge `B + I* + E` into a single span — that's the Go
  package's job. The per-token data is the source of truth; the aggregated
  block is a secondary cross-check.
- `scripts/benchmark_reference.py` (Python benchmark harness) isn't a fixture
  generator — it writes `fixtures/python_benchmark.json` for comparing Go
  vs. Python throughput, not for use by tests.
- If you change the demo text, also update the hand-written expected offsets
  in `privacyfilter_test.go` and the per-token walk example in
  `docs/architecture.md`; these aren't driven by a fixture and won't update
  themselves.
