#!/usr/bin/env -S uv run --with matplotlib --with pandas -s
# -*- coding: utf-8 -*-
"""Render a shareable PNG of the autoresearch perf-journey.

Reads ``autoresearch-results.tsv`` (project root, sibling of ``scripts/``) and
produces a chart telling the speedup story: the monotonically non-decreasing
best-so-far line, kept vs discarded experiments as differently-styled markers,
and annotations on the biggest jumps (pulled from the TSV ``description``).

Usage:
    # If uv is on PATH and the file is executable, just run it:
    ./scripts/graph_autoresearch.py
    # Or explicitly:
    uv run --with matplotlib --with pandas scripts/graph_autoresearch.py

Options:
    -i / --input   path to the TSV (default ./autoresearch-results.tsv)
    -o / --output  path to the PNG (default ./autoresearch-progress.png)

TSV format assumption: 3 leading ``#`` comment lines, a header row, then rows
with columns: iteration, commit, metric, delta, guard, guard-metric, status,
description. Status is one of ``baseline`` / ``keep`` / ``discard``.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


# Threshold for labelling a kept jump on the chart (tokens/sec).
LABEL_JUMP_THRESHOLD = 5.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("autoresearch-results.tsv"),
        help="Path to the TSV file (default: ./autoresearch-results.tsv)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("autoresearch-progress.png"),
        help="Path to the output PNG (default: ./autoresearch-progress.png)",
    )
    return p.parse_args()


def load_tsv(path: Path) -> pd.DataFrame:
    """Load the autoresearch TSV, skipping the 3 leading ``#`` comment lines."""
    if not path.exists():
        sys.exit(f"error: input TSV not found: {path}")
    try:
        df = pd.read_csv(path, sep="\t", comment="#")
    except Exception as exc:  # pd.errors.ParserError, UnicodeError, etc.
        sys.exit(f"error: failed to parse TSV {path}: {exc}")

    required = {
        "iteration",
        "commit",
        "metric",
        "delta",
        "guard",
        "guard-metric",
        "status",
        "description",
    }
    missing = required - set(df.columns)
    if missing:
        sys.exit(
            f"error: TSV missing expected columns: {sorted(missing)} "
            f"(got {list(df.columns)})"
        )
    if df.empty:
        sys.exit(f"error: TSV {path} has no data rows")

    # Coerce numerics; bail if any row is unparseable.
    for col in ("iteration", "metric"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df["iteration"].isna().any() or df["metric"].isna().any():
        bad = df[df["iteration"].isna() | df["metric"].isna()]
        sys.exit(
            f"error: non-numeric iteration/metric in rows: {bad.index.tolist()}"
        )
    df["iteration"] = df["iteration"].astype(int)
    df = df.sort_values("iteration").reset_index(drop=True)
    df["description"] = df["description"].fillna("").astype(str)
    df["status"] = df["status"].astype(str).str.strip()
    return df


def short_label(desc: str) -> str:
    """First clause of a description: split on '.', '—', or ';'."""
    if not desc:
        return ""
    # Split on the first of . — ; (em-dash or en-dash or hyphen-dash).
    m = re.split(r"[.—–;]", desc, maxsplit=1)
    first = m[0].strip()
    # Trim overly long labels.
    if len(first) > 60:
        first = first[:57].rstrip() + "..."
    return first


def render(df: pd.DataFrame, output: Path) -> None:
    # Compute the best-so-far line (monotonic non-decreasing).
    df = df.copy()
    df["best_so_far"] = df["metric"].cummax()

    baseline_rows = df[df["status"] == "baseline"]
    if baseline_rows.empty:
        # Fall back to iteration 0 or the minimum metric.
        baseline_metric = float(df.iloc[0]["metric"])
    else:
        baseline_metric = float(baseline_rows.iloc[0]["metric"])

    current_metric = float(df["metric"].iloc[-1])
    best_metric = float(df["best_so_far"].iloc[-1])
    multiplier = best_metric / baseline_metric if baseline_metric else float("nan")

    n_total = len(df)
    n_kept = int((df["status"] == "keep").sum())
    n_discard = int((df["status"] == "discard").sum())
    n_baseline = int((df["status"] == "baseline").sum())

    # Style.
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)

    # Best-so-far journey line.
    ax.plot(
        df["iteration"],
        df["best_so_far"],
        color="#1f77b4",
        linewidth=2.2,
        alpha=0.85,
        label="best-so-far",
        zorder=2,
    )

    keep_mask = df["status"] == "keep"
    discard_mask = df["status"] == "discard"
    baseline_mask = df["status"] == "baseline"

    # Discards: muted red, low opacity.
    ax.scatter(
        df.loc[discard_mask, "iteration"],
        df.loc[discard_mask, "metric"],
        marker="x",
        s=80,
        color="#d62728",
        alpha=0.45,
        linewidths=2,
        label=f"discard ({n_discard})",
        zorder=3,
    )
    # Keeps: green circle.
    ax.scatter(
        df.loc[keep_mask, "iteration"],
        df.loc[keep_mask, "metric"],
        marker="o",
        s=70,
        color="#2ca02c",
        edgecolors="white",
        linewidths=1.0,
        alpha=0.95,
        label=f"keep ({n_kept})",
        zorder=4,
    )
    # Baseline: big gold star.
    ax.scatter(
        df.loc[baseline_mask, "iteration"],
        df.loc[baseline_mask, "metric"],
        marker="*",
        s=320,
        color="#ffcc00",
        edgecolors="#8a6d00",
        linewidths=1.0,
        label=f"baseline ({n_baseline})",
        zorder=5,
    )

    # Annotate big kept jumps. A "jump" is a strictly-positive bump in the
    # best-so-far line — i.e. a new best. We label jumps whose delta over the
    # previous best exceeds LABEL_JUMP_THRESHOLD tokens/sec.
    prev_best = -float("inf")
    annotations = []
    for _, row in df.iterrows():
        best = float(row["best_so_far"])
        if best > prev_best and row["status"] in ("keep", "baseline"):
            jump = best - prev_best if prev_best != -float("inf") else 0.0
            if jump >= LABEL_JUMP_THRESHOLD or row["status"] == "baseline":
                annotations.append((int(row["iteration"]), best, short_label(row["description"]), jump))
            prev_best = best
        elif best > prev_best:
            prev_best = best

    # Stagger label y-offsets so they don't overlap too heavily.
    y_range = max(df["metric"].max() - df["metric"].min(), 1.0)
    for idx, (it, y, label, jump) in enumerate(annotations):
        # Zigzag above the point.
        offset_y = 18 + (idx % 3) * 22
        ax.annotate(
            label,
            xy=(it, y),
            xytext=(8, offset_y),
            textcoords="offset points",
            fontsize=9,
            color="#222222",
            ha="left",
            va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.25",
                fc="white",
                ec="#bbbbbb",
                alpha=0.85,
                linewidth=0.6,
            ),
            arrowprops=dict(
                arrowstyle="-",
                color="#888888",
                linewidth=0.6,
                connectionstyle="arc3,rad=0.0",
            ),
            zorder=6,
        )

    # Integer ticks on x-axis, capped to something reasonable.
    max_iter = int(df["iteration"].max())
    step = 1 if max_iter <= 40 else (2 if max_iter <= 80 else 5)
    ax.set_xticks(range(0, max_iter + 1, step))
    ax.set_xlim(-0.5, max_iter + 0.5)

    # Add a little headroom on the top so labels don't clip.
    ymin = min(df["metric"].min(), baseline_metric) * 0.92
    ymax = df["metric"].max() * 1.22
    ax.set_ylim(ymin, ymax)

    ax.set_xlabel("iteration")
    ax.set_ylabel("tokens/sec (BenchmarkClassifyMedium best-of-3)")

    # Secondary y-axis showing × vs baseline.
    ax2 = ax.twinx()
    ax2.set_ylim(ymin / baseline_metric, ymax / baseline_metric)
    ax2.set_ylabel("× vs baseline", rotation=270, labelpad=18)
    ax2.grid(False)

    title = (
        f"Autoresearch perf journey: {baseline_metric:.2f} "
        f"→ {best_metric:.2f} tokens/sec ({multiplier:.2f}×)"
    )
    ax.set_title(title, pad=14, loc="left", fontweight="bold")

    footer = (
        f"{n_total} iterations · {n_kept} kept · {n_discard} discarded "
        f"· current = {current_metric:.2f} tokens/sec · "
        f"best = {best_metric:.2f} tokens/sec"
    )
    fig.text(
        0.5,
        0.015,
        footer,
        ha="center",
        va="bottom",
        fontsize=10,
        color="#555555",
    )

    # Legend — include the best-so-far line and the marker types.
    handles = [
        Line2D([0], [0], color="#1f77b4", linewidth=2.2, label="best-so-far"),
        Line2D(
            [0], [0], marker="o", linestyle="",
            color="#2ca02c", markeredgecolor="white",
            markersize=9, label=f"keep ({n_kept})",
        ),
        Line2D(
            [0], [0], marker="x", linestyle="",
            color="#d62728", markersize=9, markeredgewidth=2,
            alpha=0.6, label=f"discard ({n_discard})",
        ),
        Line2D(
            [0], [0], marker="*", linestyle="",
            color="#ffcc00", markeredgecolor="#8a6d00",
            markersize=16, label=f"baseline ({n_baseline})",
        ),
    ]
    ax.legend(handles=handles, loc="upper left", framealpha=0.9)

    fig.tight_layout(rect=(0, 0.03, 1, 1))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = load_tsv(args.input)
    render(df, args.output)
    print(str(args.output.resolve()))


if __name__ == "__main__":
    main()
