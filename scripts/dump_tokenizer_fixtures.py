"""Dump tokenizer fixtures: tokenize a range of inputs and record ids + offsets.

These act as golden cases for the pure-Go tokenizer. We pick inputs that
exercise: ASCII words, punctuation, digits, trailing whitespace, newlines,
Unicode letters and marks, and the long Sherlock Holmes example.
"""
import json
from pathlib import Path

from transformers import AutoTokenizer

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
OUT_DIR = Path(__file__).resolve().parent.parent / "fixtures"
OUT_DIR.mkdir(exist_ok=True)

CASES = [
    "",
    "Hello",
    "Hello world",
    "My name is Sherlock Holmes and my email is sherlock.holmes@scotlandyard.uk.",
    "My name is Alice Smith",
    "Call me at 555-123-4567.",
    "\n\nHello",
    "Hello\n",
    "Hello   ",
    "abc 12345 67",
    "I don't know",
    "Naïve résumé",
    "1.5 + 2.7 = 4.2",
    "http://example.com/path",
    "  leading spaces",
    "trailing spaces   ",
    "tab\there",
    "CamelCaseIdentifier",
    "UPPER_CASE_CONST",
    "unicode: αβγ δέλτα",
]


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    out = []
    for text in CASES:
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
        out.append(
            {
                "text": text,
                "ids": list(map(int, enc["input_ids"])),
                "offsets": [list(map(int, pair)) for pair in enc["offset_mapping"]],
            }
        )

    path = OUT_DIR / "tokenizer_cases.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"wrote {path} ({len(out)} cases)")

    for item in out:
        print(f"  {item['text']!r} -> {item['ids']}")


if __name__ == "__main__":
    main()
