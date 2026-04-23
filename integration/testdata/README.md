# Integration testdata

Each test case is a pair of files sharing a basename:

- `<name>.txt` — the raw input text fed into `Classify`.
- `<name>.json` — the expected output, generated from the HuggingFace Python
  reference model with the same BIOES aggregation used by the Go package.

JSON schema:

```json
{
  "text": "<input text verbatim>",
  "entities": [
    {
      "entity_group": "private_person",
      "word":         " Harry Potter",
      "start":        10,
      "end":          23,
      "score":        0.99998
    }
  ]
}
```

`start` and `end` are CHARACTER (rune) offsets into the input text, matching
HuggingFace fast tokenizers and the Go package's `Entity.Start/End` convention.

To regenerate, from the repo root:

```
source .venv/bin/activate && python3 scripts/generate_testdata.py
```

The Go integration test in `../integration_test.go` reads every pair and
compares Go's `Classify(text)` against the stored `entities`.
