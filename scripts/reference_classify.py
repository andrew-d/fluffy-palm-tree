"""Run the HuggingFace reference implementation to capture expected output."""
import json
import os
import sys

import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLES = [
    "My name is Sherlock Holmes and my email is sherlock.holmes@scotlandyard.uk.",
    "My name is Alice Smith",
    "Call me at 555-123-4567.",
]


def main() -> None:
    print("loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    print("loading model...", flush=True)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_DIR, dtype=torch.float32
    )
    model.eval()

    id2label = model.config.id2label

    results = []
    for text in SAMPLES:
        enc = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")[0].tolist()
        with torch.no_grad():
            out = model(**enc)
        logits = out.logits[0]  # [seq, num_labels]
        pred_ids = logits.argmax(dim=-1).tolist()
        input_ids = enc["input_ids"][0].tolist()
        results.append(
            {
                "text": text,
                "input_ids": input_ids,
                "offsets": offsets,
                "pred_label_ids": pred_ids,
                "pred_labels": [id2label[i] for i in pred_ids],
                "logits_argmax_scores": [
                    float(torch.softmax(logits[i], dim=-1)[pid])
                    for i, pid in enumerate(pred_ids)
                ],
            }
        )

    # Also run the pipeline with aggregation_strategy
    print("running pipeline for aggregation...", flush=True)
    clf = pipeline(
        task="token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
    )
    aggregated = []
    for text in SAMPLES:
        agg = clf(text)
        # convert numpy floats to python floats
        agg = [
            {k: (float(v) if hasattr(v, "item") else v) for k, v in item.items()}
            for item in agg
        ]
        aggregated.append({"text": text, "entities": agg})

    out_path = os.path.join(OUT_DIR, "reference_outputs.json")
    with open(out_path, "w") as fh:
        json.dump({"per_token": results, "aggregated": aggregated}, fh, indent=2)
    print(f"wrote {out_path}")

    for item in aggregated:
        print(item["text"])
        for e in item["entities"]:
            print(f"  {e['entity_group']}: {e['word']!r} (score={e['score']:.4f})")


if __name__ == "__main__":
    main()
