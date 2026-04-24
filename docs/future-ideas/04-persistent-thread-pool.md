# Persistent thread pool with spin barriers

**Expected gain: 5–10% (moderate, and this idea actively fights the Go runtime).**

## What it is

A persistent thread pool pre-spawns N worker threads at model load time
(one per core, often pinned via affinity) and keeps them alive for the
process lifetime. Instead of `wg.Add(N); go f(); wg.Wait()` per op,
every op publishes its work descriptor to a shared struct and the
workers — already spinning at a barrier — each call
`atomic.AddInt32(&chunk, 1)` in a loop to claim the next unit until
the counter exceeds total chunks.

Completion is signaled by a two-phase barrier: each worker increments
`n_barrier`, then spins on `n_barrier_passed` using `_mm_pause` (x86
PAUSE hint) rather than a futex. No `pthread_create`, no kernel entry,
no condition variable, no allocation.

Work-stealing falls out naturally because any idle worker grabs the
next chunk, so stragglers don't stall fast workers. llama.cpp's
`ggml-cpu.c` implements exactly this; Justine Tunney's matmul writeup
explicitly notes "no futexes or semaphores, because kernel scheduling
would greatly reduce tokens/sec."

## Why it might help us

Our back-of-envelope (~5 ms on goroutine machinery, 0.3%) is probably
the *floor*, not the ceiling. Benchmarks cited in the dev.to
worker-pool comparison measure ~2 µs per `go f()` including scheduler
hand-off and stack allocation, not the ~500 ns microbench number — at
200 fan-outs × 40 goroutines × 2 µs ≈ 16 ms, closer to 1%.

The bigger lever is what happens *around* the fan-out: `wg.Wait()`
unblocks via futex (golang/go issue #34231 shows futex contention from
allocator interactions), and each wake is a scheduler round-trip. On
a loaded 8-core box, post-1.14 timer/preemption changes can push tail
latencies into milliseconds (#38860).

Compounding this: every fresh goroutine starts on a cold stack, may
land on a different P, and the GC mark-assist can hijack workers
mid-op. A persistent pool with hot caches, hot branch predictors, and
no wake-up path would eliminate all of those — likely worth a few
percent, plausibly 5–10% on a kernel-bound workload.

## What makes it hard

Go's runtime actively fights tight spin loops:

- Since 1.14, **asynchronous preemption** will signal a spinning
  goroutine after ~10 ms regardless of `runtime.Gosched()`, and the
  scheduler assumes CPU-bound goroutines are fair game to deschedule.
- `runtime.LockOSThread` pins a G to an M but **not** an M to a CPU —
  true affinity requires cgo + `pthread_setaffinity_np`, which costs
  cross-platform portability and the netpoller's flexibility.
- A pure spin barrier also starves `GOMAXPROCS`-sized P pools of any
  background work (GC, finalizers, goroutines from `net/http` handlers
  in the same process).

The pragmatic middle ground is a long-lived worker pool fed by a
buffered channel or a single `atomic.Int32` counter, with **bounded
spinning** (`for i := 0; i < 100; i++ { runtime.Gosched() }`) before
falling back to a channel receive — this captures most of the
wake-latency win without antagonizing the scheduler.

No mainstream Go library targets this niche: `errgroup` and `ants`
optimize for many small independent jobs, not 40-way fan-out on a
tight kernel. We'd be writing it ourselves, closely modeled on ggml's
barrier but using `sync/atomic` + `runtime.Gosched` instead of PAUSE.

## Sources

- [LLaMA Now Goes Faster on CPUs — Justine Tunney](https://justine.lol/matmul/)
- [ggml-org/llama.cpp PR #12488 — Thread pool pinning](https://github.com/ggml-org/llama.cpp/pull/12488)
- [golang/go #34231 — futex contention in scheduler](https://github.com/golang/go/issues/34231)
- [golang/go #38860 — CPU-bound goroutines cause timer latency](https://github.com/golang/go/issues/38860)
- [Using Goroutines is Slower?? — dev.to](https://dev.to/jpoly1219/using-goroutines-is-slower-3b53)
- [Go Wiki: Debugging performance issues](https://go.dev/wiki/Performance)
