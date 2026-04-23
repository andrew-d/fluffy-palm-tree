#!/usr/bin/env bash
# Autoresearch verify: runs BenchmarkClassifyMedium a few times and prints
# the best (highest) tokens/sec number on stdout. Direction: higher is better.
set -euo pipefail

cd "$(dirname "$0")/.."

# -count=3 -benchtime=3x gives us 3 independent samples of (3 in-loop iters).
# Best-of-3 filters out noise from co-tenant CPU jitter.
#
# systemd-run caps memory and disables swap so a runaway allocation fails fast
# inside the scope instead of dragging the whole box into swap thrash. Only
# MemoryMax is set (no MemoryHigh): exceeding MemoryHigh triggers slow kernel
# reclaim and thrashing; exceeding MemoryMax triggers an immediate OOM kill
# (exit 137), which is the signal we actually want — the experiment is too
# memory-hungry, discard it and move on. The cap sits well below total RAM
# (30G) so the user-scope parent has headroom.
#
# -p=1 serializes package test binaries so we never have two model loads in
# flight at once — the full test tree OOMs at parallel=N otherwise.
out=$(systemd-run --user --scope --quiet \
    -p MemoryMax=14G -p MemorySwapMax=0 \
    go test -p=1 -run='^$' -bench='^BenchmarkClassifyMedium$' \
    -benchtime=3x -count=3 . 2>&1)

# Each matching line looks like:
#   BenchmarkClassifyMedium-8  3  10303390406 ns/op  102.4 bytes/sec  102.4 chars/sec  29.70 tokens/sec  ...
# Extract the number immediately before "tokens/sec".
echo "$out" \
  | awk '{for(i=1;i<=NF;i++) if($i=="tokens/sec") print $(i-1)}' \
  | sort -g \
  | tail -1
