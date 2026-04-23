"""Regenerate integration/testdata/*.json from .txt input files.

For every `integration/testdata/<name>.txt`, this script runs the HuggingFace
reference implementation, aggregates per-token BIOES labels into spans using
the same walk as the Go package (internal/privacyfilter aggregation.go), and
writes `integration/testdata/<name>.json`.

Usage:
    source .venv/bin/activate
    python3 scripts/generate_testdata.py

The BIOES aggregation mirrors the Go code exactly:
  - O:        closes any open span
  - S-X:      closes any open span, emits a one-token span of type X
  - B-X:      closes any open span, opens new span of type X
  - I-X:      if open type == X, extend; else open a fresh span of type X
  - E-X:      if open type == X, extend + close; else open + close a single span
  - EOF:      if any open span, close it
Entity score = mean of per-token softmax probability at each token's argmax.
"""
import json
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO / "model"
TESTDATA_DIR = REPO / "integration" / "testdata"


def aggregate_bioes(text, offsets, labels, scores):
    """Walk tokens once, emit a list of entity spans with start/end char offsets."""
    entities = []
    open_type = None
    buf_starts = []
    buf_ends = []
    buf_scores = []

    def close():
        nonlocal open_type, buf_starts, buf_ends, buf_scores
        if open_type is None:
            return
        s = buf_starts[0]
        e = buf_ends[-1]
        entities.append(
            {
                "entity_group": open_type,
                "word": text[s:e],
                "start": int(s),
                "end": int(e),
                "score": float(sum(buf_scores) / len(buf_scores)),
            }
        )
        open_type = None
        buf_starts = []
        buf_ends = []
        buf_scores = []

    def start(etype, off, sc):
        nonlocal open_type, buf_starts, buf_ends, buf_scores
        open_type = etype
        buf_starts = [off[0]]
        buf_ends = [off[1]]
        buf_scores = [sc]

    def extend(off, sc):
        buf_starts.append(off[0])
        buf_ends.append(off[1])
        buf_scores.append(sc)

    for off, lbl, sc in zip(offsets, labels, scores):
        if lbl == "O":
            close()
            continue
        prefix, etype = lbl.split("-", 1)
        if prefix == "S":
            close()
            start(etype, off, sc)
            close()
        elif prefix == "B":
            close()
            start(etype, off, sc)
        elif prefix == "I":
            if open_type == etype:
                extend(off, sc)
            else:
                close()
                start(etype, off, sc)
        elif prefix == "E":
            if open_type == etype:
                extend(off, sc)
                close()
            else:
                close()
                start(etype, off, sc)
                close()
        else:
            raise ValueError(f"unknown BIOES prefix: {prefix}")

    close()
    return entities


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR, dtype=torch.float32)
    model.eval()
    id2label = model.config.id2label

    input_files = sorted(TESTDATA_DIR.glob("*.txt"))
    if not input_files:
        print(f"no .txt cases in {TESTDATA_DIR}; add some and re-run")
        return

    for txt_path in input_files:
        text = txt_path.read_text()
        # Strip a single trailing newline: editors often add one; the test
        # case is the text content up to but not including that final '\n'.
        if text.endswith("\n"):
            text = text[:-1]

        enc = tok(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
        offsets = enc.pop("offset_mapping")[0].tolist()
        with torch.no_grad():
            out = model(**enc)
        logits = out.logits[0]
        probs = torch.softmax(logits, dim=-1)
        pred_ids = logits.argmax(dim=-1).tolist()
        labels = [id2label[i] for i in pred_ids]
        scores = [float(probs[i, pid]) for i, pid in enumerate(pred_ids)]

        entities = aggregate_bioes(text, offsets, labels, scores)

        out_path = txt_path.with_suffix(".json")
        out_path.write_text(
            json.dumps(
                {"text": text, "entities": entities},
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
        print(f"  {txt_path.name} -> {len(entities)} entities")


if __name__ == "__main__":
    main()
