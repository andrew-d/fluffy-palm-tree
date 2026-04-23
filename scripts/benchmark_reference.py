"""Benchmark the HuggingFace Python reference for openai/privacy-filter.

Mirrors the Go benchmark in `bench_test.go` exactly: same input paragraph,
same short/medium/long multipliers, same warmup policy, same reported metrics
(tokens/sec, chars/sec, bytes/sec).

The "Classify" equivalent includes tokenization + forward + BIOES aggregation,
just like the Go library's Classify. Tokenization is done via the HF
tokenizer; aggregation reuses the same walker that
`scripts/generate_testdata.py` uses (matching `aggregation.go`).

Usage:
    source .venv/bin/activate
    python3 scripts/benchmark_reference.py [--iterations N] [--threads T]

Prints one line per case in a format similar to `go test -bench` so results
are easy to eyeball against each other.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

# Reuse the aggregator so Python and Go agree on span merging cost shape.
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from generate_testdata import aggregate_bioes  # noqa: E402

REPO = SCRIPT_DIR.parent
MODEL_DIR = REPO / "model"

BENCH_PARAGRAPH = (
    "My name is Harry Potter and my email is harry.potter@hogwarts.edu. "
    "Please contact me at 555-123-4567 or at my home, 4 Privet Drive, Little Whinging. "
    "My account number is 000123456789 and my password is hunter2. "
)

CASES = [
    ("Short", 1),
    ("Medium", 5),
    ("Long", 20),
]


def classify_once(tok, model, id2label, text):
    """Tokenize, forward, argmax+softmax, BIOES-aggregate — the full
    end-to-end Classify path so we're comparing apples to apples."""
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc.pop("offset_mapping")[0].tolist()
    with torch.no_grad():
        out = model(**enc)
    logits = out.logits[0]
    probs = torch.softmax(logits, dim=-1)
    pred_ids = logits.argmax(dim=-1).tolist()
    labels = [id2label[i] for i in pred_ids]
    scores = [float(probs[i, pid]) for i, pid in enumerate(pred_ids)]
    return aggregate_bioes(text, offsets, labels, scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", "-n", type=int, default=5,
                        help="measured iterations per case (after 1 warmup)")
    parser.add_argument("--threads", "-t", type=int, default=None,
                        help="torch.set_num_threads (default: PyTorch's default)")
    args = parser.parse_args()

    if args.threads is not None:
        torch.set_num_threads(args.threads)

    print(f"device: cpu")
    print(f"torch threads: {torch.get_num_threads()}")

    print("loading tokenizer...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    print("loading model...", flush=True)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR, dtype=torch.float32)
    model.eval()
    id2label = model.config.id2label

    print()
    print(f"{'case':10s}  {'tokens':>7s}  {'chars':>7s}  {'bytes':>7s}  "
          f"{'iters':>5s}  {'wall(s)':>8s}  "
          f"{'ns/op':>12s}  {'tokens/s':>9s}  {'chars/s':>9s}  {'bytes/s':>9s}")

    results = []
    for name, repeat in CASES:
        text = BENCH_PARAGRAPH * repeat
        n_tokens = len(tok(text, add_special_tokens=False)["input_ids"])
        n_chars = len(text)
        n_bytes = len(text.encode("utf-8"))

        # Warmup once
        classify_once(tok, model, id2label, text)

        # Measure
        t0 = time.perf_counter()
        for _ in range(args.iterations):
            classify_once(tok, model, id2label, text)
        elapsed = time.perf_counter() - t0

        ns_per_op = int(elapsed * 1e9 / args.iterations)
        tps = n_tokens * args.iterations / elapsed
        cps = n_chars * args.iterations / elapsed
        bps = n_bytes * args.iterations / elapsed
        print(f"{name:10s}  {n_tokens:>7d}  {n_chars:>7d}  {n_bytes:>7d}  "
              f"{args.iterations:>5d}  {elapsed:>8.3f}  "
              f"{ns_per_op:>12d}  {tps:>9.2f}  {cps:>9.2f}  {bps:>9.2f}")

        results.append({
            "case": name,
            "tokens": n_tokens,
            "chars": n_chars,
            "bytes": n_bytes,
            "iterations": args.iterations,
            "elapsed_seconds": elapsed,
            "ns_per_op": ns_per_op,
            "tokens_per_sec": tps,
            "chars_per_sec": cps,
            "bytes_per_sec": bps,
        })

    out = REPO / "fixtures" / "python_benchmark.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
