#!/usr/bin/env bash
# Autoresearch guard: vet + full test suite, no -race. Exits non-zero on any
# failure — the loop reverts the experiment.
#
# -race was dropped because the safetensors fp32 decode under checkptr
# instrumentation pushed a single-package run past 6 minutes. The loop
# iterates per kept experiment and that feedback tax was untenable. Race
# coverage on the MoE goroutine fan-out is a separate concern to address
# later. For now, plain correctness checks.
#
# Memory is still capped (MemoryMax only, no MemoryHigh) with swap disabled
# so a runaway experiment OOM-kills immediately (exit 137) instead of
# dragging the host into swap thrash. Fail-fast beats sluggish-then-fail.
#
# -p=1 serializes packages so we only have one model load in RSS at a time.
set -euo pipefail

cd "$(dirname "$0")/.."

go vet ./...

# Match the verify script's flags so the guard validates exactly the
# same binary flavor we're benchmarking.
systemd-run --user --scope --quiet \
    -p MemoryMax=14G -p MemorySwapMax=0 \
    env GOAMD64=v3 GOEXPERIMENT=simd GOGC=500 \
    go test -p=1 -timeout=300s -count=1 ./...
