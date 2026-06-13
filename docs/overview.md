# goblas — overview, rationale, and limitations

## What goblas is

goblas is a **BLAS** (Basic Linear Algebra Subprograms) library for Go, written in
**pure Go by default** with hand-tuned **assembly kernels** for the operations that matter
most on processors we have tuned for (today: Apple Silicon via ARM64 NEON, plus an x86-64
AVX2 `dgemm` kernel).

BLAS is the layer almost every numerical program stands on. It defines a small, stable set
of routines for the three levels of dense linear algebra:

- **Level 1** — vector–vector: dot products (`Ddot`), scaled adds (`Daxpy`), norms, etc.
- **Level 2** — matrix–vector: `Dgemv` and friends.
- **Level 3** — matrix–matrix: `Dgemm` (general matrix multiply) and the triangular /
  symmetric variants (`Dsyrk`, `Dtrsm`, `Dsymm`, `Dtrmm`).

If you make matrix multiply fast, you make almost everything above it fast — linear solves,
least squares, Cholesky/LU/QR/SVD factorizations, and most of classical machine learning and
neural-network math. That is the whole leverage of a good BLAS.

## Why it exists

Go has no production-quality, fast, pure-Go BLAS. The options before goblas were:

1. **[Gonum](https://www.gonum.org/)'s pure-Go BLAS** — correct, portable, and the de-facto
   standard, but the Go compiler does not auto-vectorize, so for large matrices it leaves
   roughly **4–10× of performance** on the table compared to hand-tuned SIMD code.
2. **CGo bindings to a C BLAS** (OpenBLAS, Apple Accelerate, MKL) — fast, but CGo brings real
   costs: you lose trivial cross-compilation, you need a C toolchain and system libraries at
   build time, deployment gets heavier, and every call pays the CGo boundary overhead.

goblas takes a third path: **be pure Go and fast.** The pure-Go implementation is always
present and always correct — it is both the portable fallback and the reference the assembly
is tested against. On top of it, per-CPU assembly kernels are layered in for the hot
routines. The result keeps Go's "compile anywhere, `go build` and ship a static binary"
experience while reaching, on Apple Silicon, **~96% of OpenBLAS** and **~84% of Apple
Accelerate** on large `dgemm` (see the [README](../README.md) for the full benchmark tables).

## How it is built (design principles)

You do not need this section to *use* goblas, but it explains the limitations below.

- **Column-major layout.** Matrices are stored Fortran-style: element A(i, j) lives at
  `a[i + j*lda]`, where `lda` (the "leading dimension") is the column stride. This matches
  classic BLAS, LAPACK, OpenBLAS, and Accelerate, and it matches the access pattern of the
  `dgemm` inner loop.
- **One interface, a generic implementation, and accelerated overrides.** There is a single
  `Kernel` interface listing every routine. A `genericKernel` implements all of them in plain
  Go. Accelerated kernels (e.g. the NEON one) **embed** the generic kernel and **override
  only** the routines they have assembly for — so anything not yet written in assembly
  automatically falls back to correct Go. A bug in assembly can only ever make something
  *slower by being skipped*, never silently wrong, because the generic path is the spec.
- **Runtime dispatch.** At process start, goblas detects the CPU once and picks the best
  available kernel. No build flags, no configuration.
- **Extensibility as a first-class goal.** Adding support for a new processor is designed to
  be *"one kernel file + one registration,"* with no changes to the public API or to any
  calling code. See [adding-new-cpu.md](adding-new-cpu.md).
- **It plugs in under Gonum.** A single `blasadapt.Use()` call registers goblas as the BLAS
  for the whole Gonum stack, so `gonum/mat` and Gonum's pure-Go LAPACK (`Solve`, `LU`,
  `Cholesky`, `QR`, `SVD`, …) run on goblas kernels with zero code changes. The bridge between
  Gonum's row-major world and goblas's column-major one is a relabeling (swap operands, flip
  flags) — no data is copied.

## Limitations

It matters to separate two very different kinds of limitation: things that are simply **not
done yet** (and could be, within the same design), versus things that are **inherent to the
idea** of a pure-Go, no-CGo, CPU BLAS (and will not change without abandoning a core
premise).

### Current limitations (not done yet — fixable within the design)

- **float64 only.** No `float32` (single precision) or complex routines. The whole library is
  double precision. float32 would roughly double SIMD throughput and is what most neural-net
  training wants — it is a natural future addition, just not built.
- **Assembly coverage is partial.** On ARM64, `Dgemm`/`Dsyrk`/`Dtrsm` run on NEON; `Dsymm`
  and `Dtrmm` still use portable reference loops, and several Level-1/Level-2 routines fall
  back to pure Go. More kernels can be added one at a time.
- **Only Apple Silicon is tuned.** Other ARM64 chips (Graviton, Snapdragon, …) get the NEON
  kernels with conservative, cache-size-derived blocking rather than a hand-tuned sweep.
- **The AVX2 (x86-64) kernel is unverified on real hardware.** It was written and checked by
  cross-compilation, `go vet`, and disassembly review on an ARM64 machine, but has not yet
  been *run* on an Intel/AMD CPU. It is marked experimental until that happens. (x86 also has
  no Level-1/Level-2 assembly yet — only `dgemm`.)
- **We rely on Gonum's LAPACK, not our own.** The high-level factorizations come from Gonum's
  pure-Go LAPACK calling our BLAS. That is the right call (LAPACK is ~1,700 routines), but it
  means the factorization *orchestration* is not itself goblas-tuned — only the BLAS calls it
  makes are.

None of these change the architecture; each is "write another kernel" or "add another type."

### Inherent limitations (the cost of the idea itself)

- **It cannot beat Apple Accelerate.** Accelerate uses the **AMX** matrix coprocessor — a
  separate, undocumented unit that does float64 matrix math at a throughput ordinary SIMD
  instructions cannot match. goblas reaches ~84% of Accelerate on large `dgemm` and is capped
  there *by physics we cannot access*: NEON is the fastest documented path, and AMX's encoding
  is reverse-engineered and unstable across chip generations. Closing that last gap would mean
  emitting undocumented instructions — a different, riskier project.
- **No GPU.** The "no CGo" promise rules it out: talking to the GPU on a Mac means the Metal
  framework, which is not reachable from pure Go without CGo (or a runtime-linking trick like
  `purego`). And even if it were, **Apple GPUs have no native float64** — Metal's shading
  language has no `double` type — so a double-precision GPU `dgemm` is not just hard, it is
  impossible on this hardware. GPU acceleration is fundamentally out of scope for a pure-Go
  *float64* CPU library.
- **Single node, CPU only.** No multi-machine / distributed computation. Parallelism is
  goroutines across the cores of one chip.
- **BLAS only helps work that is actually matrix/vector-bound.** This is the big one for the
  use-case tutorials. A fast `dgemm` accelerates algorithms whose cost is *dense linear
  algebra* — regressions, kernel methods' Gram matrices, neural-net layers. It does **nothing**
  for algorithms that are branchy, comparison-driven, or sequential (decision trees, most
  sorting/search, graph traversal). Putting goblas under such an algorithm changes nothing,
  because there is no matrix multiply for it to speed up. (This is exactly why the use-case
  docs *omit* decision trees.)
- **Level 1 and Level 2 are near the hardware limit already.** Dot products, `axpy`, and
  matrix-vector products move a lot of memory per arithmetic operation, so they are
  **memory-bandwidth-bound**: once the data does not fit in cache, the bottleneck is RAM
  speed, not compute. goblas already beats Gonum 6×+ on these, but there is little headroom
  left — you cannot out-engineer the memory bus.

## Where to go next

- **Use it:** [README — Getting started](../README.md#getting-started), then the
  [use-case tutorials](use-cases/README.md).
- **Extend it to your CPU:** [adding-new-cpu.md](adding-new-cpu.md).
- **Test and benchmark it:** [verify-benchmark.md](verify-benchmark.md).
