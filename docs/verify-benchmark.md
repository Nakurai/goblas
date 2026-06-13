# Verifying and benchmarking goblas

Two separate questions, two separate workflows:

- **Correctness** — does it compute the right answers? (Run the tests.)
- **Performance** — how fast is it, and versus what? (Run the benchmarks.)

## Quick smoke test

If you just want to confirm a checkout is healthy, these three commands are enough:

```sh
go test ./...                              # all correctness tests, every package
go test -race ./internal/kernel/ ./blasadapt/   # data-race check on the parallel code
go test -run '^$' -bench 'Dgemm|Ddot' -benchtime=500ms ./...   # a couple of speed numbers
```

If those pass, the library is correct on your machine and the accelerated path is running.

---

## Correctness

### The idea: the pure-Go reference *is* the specification

Every accelerated kernel is tested by comparing its output, on random fixed-seed inputs, to
the **pure-Go generic kernel** computing the same thing. The generic kernel is simple enough
to read and trust, so it serves as the ground truth. (Because tiling reorders floating-point
sums, the comparison is "within a small tolerance," not bit-exact.) Separately, the Gonum
adapter is tested against *stock Gonum* — if our row→column relabeling were wrong, those tests
would fail immediately.

You do not need to know the individual test names to run everything (`go test ./...` does),
but here is the map of what is being checked and where:

| Area | What it proves | Where |
|------|----------------|-------|
| **Public API** | `Ddot`, `Daxpy`, `Dgemv`, `Dgemm`, … give correct results and reject bad arguments | `blas_test.go` |
| **Accelerated `dgemm` vs generic** | the NEON / AVX2 matrix-multiply matches the reference across all shapes, transposes, edge tiles, and alpha/beta | `internal/kernel/dgemm_arm64_test.go`, `dgemm_amd64_test.go` |
| **Level-3 triangular/symmetric** | `Dsyrk`/`Dtrsm`/`Dsymm`/`Dtrmm` match a dense reference | `internal/kernel/l3_tri_test.go` |
| **Level-1 accelerated** | `Ddot`/`Daxpy`/`Dscal` NEON paths match generic | `internal/kernel/*_arm64_test.go` |
| **Portable blocked `dgemm`** | the pure-Go tiled path (used on un-accelerated CPUs) matches the naive triple loop | `internal/kernel/dgemm_driver_test.go` |
| **Gonum adapter** | every overridden routine matches stock Gonum; and a full `mat` run (`Solve`, `LU`, `Cholesky`, `QR`, `SVD`) on goblas matches stock | `blasadapt/adapter_test.go`, `blasadapt/mat_test.go` |
| **CPU detection** | the chip is classified and cache size read | `internal/cpu/*_test.go` |

### Commands

```sh
go test ./...                 # everything
go test ./internal/kernel/    # just the kernels
go test -v -run TestDgemm ./internal/kernel/   # one routine, verbose
```

### The race check (important for Level 3)

`Dgemm` runs in parallel across goroutines, each writing a disjoint band of the output. To
prove there is no accidental shared-memory write, run the suite under Go's race detector:

```sh
go test -race ./internal/kernel/ ./blasadapt/
```

A clean run here is what justifies the parallelism being safe.

### Checking platforms you cannot run

The build tags mean only your host's kernels compile by default. To confirm goblas still
*builds and type-checks* everywhere — and that assembly stack frames are sound — cross-compile:

```sh
GOARCH=amd64           go vet ./...     # x86-64: runs the assembler + asmdecl checker
GOOS=linux GOARCH=arm64 go build ./...  # Linux ARM64
GOARCH=amd64           go test -c -o /dev/null ./internal/kernel/   # tests link
```

For an architecture you cannot execute (e.g. x86 AVX2 from an ARM Mac), `go vet` plus a
disassembly review is the strongest available check — see
[adding-new-cpu.md](adding-new-cpu.md#verifying-assembly-you-cannot-run). The runtime
correctness tests for such a kernel `t.Skip()` themselves when the CPU feature is absent, so
they will not falsely pass.

---

## Benchmarking

### What is measured

For matrix multiply, performance is reported in **GFLOPS** — billions of floating-point
operations per second. A multiply of two N×N matrices does `2·N³` operations, so
`GFLOPS = 2·N³ / seconds / 1e9`. Higher is better.

Expect a **rise-then-fall curve** as N grows: small matrices fit in fast cache and scale up
nicely, then at some size the data spills out of cache and the rate dips — this is normal and
visible in every BLAS.

### The everyday benchmarks (no extra dependencies)

```sh
# A spread of routines and sizes, ~1s each
go test -run '^$' -bench . -benchtime=1s ./...

# Generic (pure-Go) vs accelerated, head to head
go test -run '^$' -bench Dgemm ./internal/kernel/
```

The `-run '^$'` part disables the *tests* so only benchmarks run (it matches no test names).

### Comparison benchmarks (opt-in via build tags)

goblas keeps heavy/foreign dependencies out of the normal build by gating comparison
benchmarks behind build tags. Each needs a prerequisite:

| Compare against | Build tag | Prerequisite | Command |
|-----------------|-----------|--------------|---------|
| **Gonum** (pure-Go BLAS — the primary baseline) | `gonumbench` | none (Gonum already a dep) | `go test -tags gonumbench -run '^$' -bench . ./...` |
| **Apple Accelerate** (the AMX hardware ceiling) | `accelbench` | macOS | `go test -tags accelbench -run '^$' -bench Accelerate .` |
| **OpenBLAS** (the like-for-like SIMD reference) | `openblasbench` | `brew install openblas` | `go test -tags openblasbench -run '^$' -bench Openblas .` |

These are what produce the percentage figures in the README ("96% of OpenBLAS", "84% of
Accelerate").

### Tuning benchmarks

Two benchmarks exist to *make decisions*, not just report numbers:

```sh
# Sweep the micro-kernel shape (8x4 vs 8x6) and cache blocking (kc, mc).
go test -run '^$' -bench DgemmBlockSweep -benchtime=500ms ./internal/kernel/

# Sweep how many worker goroutines dgemm uses, at several sizes.
go test -run '^$' -bench DgemmWorkerSweep -benchtime=1s ./internal/kernel/
```

`BenchmarkDgemmBlockSweep` is the tool you use when [tuning for a new
CPU](adding-new-cpu.md#tier-1-no-assembly-detection-tuning): it tells you the best `kc`/`mc`
for your cache.

### A real caveat: thermal drift

Laptop CPUs (the M5 Pro included) **slow down as they heat up**. A long benchmark run drifts
downward over time, which can make whichever kernel ran *second* look worse than it is. This
bit us during tuning. To compare two kernels fairly:

- use a **short `-benchtime`** (e.g. `500ms`) and **interleave** the two — run A, B, A, B, …
  rather than all of A then all of B, so both see the same thermal state;
- use **`-count=3`** (or more) and look at the spread, not a single number;
- let the machine cool and stay otherwise idle between heavy runs.

Absolute GFLOPS will vary with temperature and what else your machine is doing; **relative**
comparisons under interleaving are the trustworthy signal.

### Controlling parallelism

```sh
GOMAXPROCS=1 go test -run '^$' -bench Dgemm ./internal/kernel/   # single-core number
```

Single-core GFLOPS isolates the micro-kernel's quality from the threading; the default
(all cores) shows the end-to-end result.

---

## See also

- [overview.md](overview.md) — what the numbers mean and why the ceilings are where they are.
- [adding-new-cpu.md](adding-new-cpu.md) — using these tools to tune or validate a new kernel.
- [../README.md](../README.md) — the published benchmark tables.
