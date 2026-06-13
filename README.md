# goblas

A pure-Go BLAS (Basic Linear Algebra Subprograms) library for float64 **and float32**, with hand-tuned ARM64 NEON assembly kernels on Apple Silicon.

**No CGo. No external dependencies at runtime.** The library is fully portable — it compiles and runs correctly on any platform — and selects accelerated assembly automatically on supported processors at startup.

**AI Disclaimer**: Since it looks like this is a polarizing topic, let's make it clear that this entire repo has been generated. Use with caution, at your own risks.

## Status

| Level | Routine | Pure-Go | NEON (ARM64) |
|-------|---------|---------|--------------|
| **L1** | `Ddot` | ✅ | ✅ ~17.6 GFLOPS |
| **L1** | `Daxpy` | ✅ | ✅ ~15 GFLOPS |
| **L1** | `Dscal` | ✅ | ✅ |
| **L1** | `Dnrm2`, `Dasum` | ✅ | ✅ ~8× / ~5× (NEON reduction) |
| **L1** | `Idamax`, `Dcopy`, `Dswap` | ✅ | (fallback to pure-Go) |
| **L2** | `Dgemv` | ✅ | ✅ ~17 GFLOPS |
| **L2** | `Dger`, `Dtrsv` | ✅ | ✅ ~2× (reuse NEON `daxpy`/`ddot`) |
| **L3** | `Dgemm` | ✅ | ✅ ~390 GFLOPS (tiled, multithreaded, tuned) |
| **L3** | `Dsyrk`, `Dtrsm` | ✅ | ✅ recursive blocking — bulk runs on the NEON `Dgemm` |
| **L3** | `Dsymm`, `Dtrmm` | ✅ | ✅ recursive blocking — bulk runs on the NEON `Dgemm` |

All routines are correct and tested on every platform. The pure-Go fallback is always available; the NEON kernel is selected at runtime when running on Apple Silicon.

### Single precision (float32)

Every routine above has a single-precision `S`-prefixed twin (`Sdot`, `Saxpy`, `Sscal`, `Snrm2`, `Sasum`, `Isamax`, `Scopy`, `Sswap`, `Sgemv`, `Sger`, `Strsv`, `Sgemm`, `Ssyrk`, `Strsm`, `Ssymm`, `Strmm`). The float32 and float64 paths are the *same* blocked driver and triangular/symmetric recursion instantiated at the two element types via Go generics — only the leaf micro-kernels differ. A 128-bit NEON register holds 4 float32 (`.S4`) vs 2 float64 (`.D2`), so single precision runs faster:

| Level | Routine | NEON (ARM64) |
|-------|---------|--------------|
| **L1** | `Sdot`, `Sasum`, `Snrm2` | ✅ ~17.5 Gelem/s (bandwidth-bound; ~13× / ~49× / ~28× vs pure-Go) |
| **L1** | `Saxpy`, `Sscal` | ✅ `.S4` kernels |
| **L1** | `Isamax`, `Scopy`, `Sswap` | (fallback to pure-Go; `Isamax` can't be vectorized — see below) |
| **L2** | `Sgemv` | ✅ `.S4` column-axpy |
| **L2** | `Sger`, `Strsv` | ✅ reuse NEON `saxpy`/`sdot` |
| **L3** | `Sgemm` | ✅ **~562 GFLOPS** (8×8 micro-kernel; ~1.57× the float64 `Dgemm`) |
| **L3** | `Ssyrk`, `Strsm`, `Ssymm`, `Strmm` | ✅ recursive blocking onto the NEON `Sgemm` |

## Benchmarks (M5 Pro, unit-stride float64)

### vs Gonum's pure-Go BLAS (primary baseline)

| Routine | Gonum | goblas NEON | Speedup |
|---------|-------|-------------|---------|
| `Ddot` (n=16384) | ~2.6 GFLOPS | **17.8 GFLOPS** | **6.9x** ✅ |
| `Dgemv` (n=1024) | ~2.8 GFLOPS | **17.5 GFLOPS** | **6.2x** ✅ |
| `Dgemm` (n=64) | ~8.3 GFLOPS | **33.3 GFLOPS** | **4.0x** ✅ |
| `Dgemm` (n=256) | ~59 GFLOPS | **126 GFLOPS** | **2.1x** ✅ |
| `Dgemm` (n=512) | ~79 GFLOPS | **282 GFLOPS** | **3.6x** ✅ |
| `Dgemm` (n=1024) | ~81 GFLOPS | **390 GFLOPS** | **4.8x** ✅ |

### vs Apple Accelerate (the hardware ceiling)

| n | Accelerate | goblas | % of ceiling |
|---|-----------|--------|--------------|
| 256 | 388 GFLOPS | 127 GFLOPS | 33% |
| 512 | 451 GFLOPS | 282 GFLOPS | 63% |
| 1024 | 473 GFLOPS | **397 GFLOPS** | **84%** |

Accelerate uses Apple's undocumented AMX matrix coprocessor, so it is the absolute hardware ceiling — not a like-for-like SIMD comparison. Reaching 84% of it with pure Go + NEON assembly exceeds the investigation's expectations ("matching Accelerate is probably impossible").

### vs OpenBLAS (the like-for-like SIMD comparison)

| n | OpenBLAS | goblas | % of OpenBLAS |
|---|----------|--------|---------------|
| 64 | 25 GFLOPS | **33 GFLOPS** | **130%** (goblas faster) |
| 256 | 158 GFLOPS | 116 GFLOPS | 73% |
| 512 | 365 GFLOPS | 247 GFLOPS | 68% |
| 1024 | 390 GFLOPS | **376 GFLOPS** | **96%** |

OpenBLAS is the de-facto open-source optimized BLAS, with hand-tuned NEON kernels for ARM64 — the investigation's stated target was "70–85% of OpenBLAS". goblas reaches 96% at n=1024 and is *faster* at small sizes (goroutines beat OpenBLAS's thread pool on small problems, exactly as the investigation predicted). Run it yourself: `brew install openblas`, then `go test -tags openblasbench -run '^$' -bench Openblas .`

### Single precision (float32 `Sgemm`, M5 Pro)

| n | Gonum blas32 | goblas | Accelerate (AMX) | goblas vs Gonum |
|---|--------------|--------|------------------|-----------------|
| 256 | ~58 GFLOPS | **174 GFLOPS** | 1555 | **3.0×** |
| 512 | ~83 GFLOPS | **339 GFLOPS** | 1552 | **4.1×** |
| 1024 | ~79 GFLOPS | **566 GFLOPS** | 1762 | **7.1×** |

goblas `Sgemm` is ~7× Gonum's pure-Go single-precision GEMM and ~1.57× goblas's own `Dgemm`. Against Accelerate it reaches a smaller fraction than in float64 (~33% at n=1024): Apple's AMX coprocessor is dramatically faster in single precision than NEON can match — an honest hardware-ceiling note, not a regression.

**L1 & L2:** Beat the 4–6x investigation target on `Ddot` and `Dgemv`. Both are memory-bandwidth-bound, so further improvement is limited by hardware.

**L3 (`Dgemm`):** The tiled NEON implementation packs A and B into cache-resident micro-panels, runs a register micro-kernel in assembly, and parallelizes row blocks across cores with goroutines. Phase 6 tuning (kc=512, mc=24 — sized so a packed A block sits inside the 128 KB P-core L1d) brought n=1024 from 236 to ~390 GFLOPS: **4.8x faster than Gonum**, hitting the investigation's 4–6x target even against Gonum's tiled, multithreaded `dgemm`. Phase 12 replaced the 8×4 micro-kernel with an **8×6** one (24 accumulator registers; each A load feeds 6 columns instead of 4), a further ~6% in interleaved A/B runs at n=1024.

## Accelerating Gonum (high-level linear algebra)

goblas plugs in underneath the entire Gonum numerical stack. One call registers
it as the BLAS used by `gonum/mat` and Gonum's pure-Go LAPACK — after that,
`Solve`, `Inverse`, `mat.LU`, `mat.Cholesky`, `mat.QR`, `mat.SVD`, `mat.Eigen`
and friends all run on goblas kernels:

```go
import (
	"github.com/nakurai/goblas/blasadapt"
	"gonum.org/v1/gonum/mat"
)

func init() { blasadapt.Use() }

func main() {
	a := mat.NewDense(1000, 1000, data)
	var x mat.Dense
	x.Solve(a, b)        // LU factorization + solve — on goblas kernels
	var chol mat.Cholesky
	chol.Factorize(spd)  // blocked Cholesky — dsyrk/dtrsm/dgemm on goblas
}
```

The bridge is zero-copy: Gonum's row-major calls are relabeled onto goblas's
column-major kernels via transpose identities (swap operands and dimensions,
flip flags) — no data movement, no extra FLOPs. Routines goblas doesn't
accelerate fall back to Gonum's own BLAS automatically.

**Float32 BLAS:** `blasadapt.Use32()` registers goblas as the implementation for
Gonum's `blas32` package, so `blas32.Gemm`/`Gemv`/`Trsm`/`Syrk`/… on
`blas32.General` run on goblas single-precision kernels. It is independent of
`Use()` — call both and float32 and float64 work each dispatch to their own
kernels in the same program.

**Float32 matrices — the `mat32` package.** Gonum has no float32 LAPACK and
`gonum/mat` is float64-only, so goblas ships its own native float32 matrix type
in [`mat32`](mat32). It provides `Dense32` with `Mul`/`MulVec`/`Add`/`Scale`/…
and the **`Cholesky32` and `LU32` solves end-to-end in float32 — no float64
casting** (the trailing FLOPs run on the goblas `Sgemm`/`Strsm`/`Ssyrk` kernels),
so float32 inputs stay float32. The advanced factorizations (`QR32`, `SVD32`,
`EigenSym32`, `Eigen32`) are provided via a float64 bridge to gonum and *do*
cast internally. See [mat32 below](#float32-matrices-mat32).

Measured on `gonum/mat` operations at n=1024 (M5 Pro, stock Gonum vs goblas registered):

| Operation | Stock Gonum | With goblas | Speedup |
|-----------|------------|-------------|---------|
| `Dense.Mul` | 27.2 ms | **5.9 ms** | **4.6x** |
| `Cholesky.Factorize` | 34.5 ms | **17.5 ms** | **2.0x** |
| `Dense.Solve` (LU) | 38.2 ms | **26.4 ms** | **1.4x** |

## Float32 matrices (`mat32`)

Because gonum offers no high-level float32 linear algebra, the
[`mat32`](mat32) package provides a native one built on the goblas `S`-kernels —
so float32 data (sensor streams, ML activations) gets matrices and solves
**without casting to float64**:

```go
import "github.com/nakurai/goblas/mat32"

// Ridge regression weights end-to-end in float32: w = (XᵀX + λI)⁻¹ Xᵀy.
var xtx mat32.Dense32
xtx.Mul(X.T(), X)                  // goblas Sgemm
// ... add λ to the diagonal ...
a := mat32.SymDense32FromDense(&xtx, true)
var xty mat32.VecDense32
xty.MulVec(X.T(), y)               // goblas Sgemv

var chol mat32.Cholesky32
chol.Factorize(a)                  // native float32 Cholesky (Ssyrk/Strsm/Sgemm)
var w mat32.Dense32
chol.SolveTo(&w, &xty)             // native float32 triangular solves
```

**Precision boundary:** `Dense32` arithmetic and the **`Cholesky32` / `LU32`
solves are end-to-end float32** — no float64 round-trips, the trailing FLOPs run
on goblas `Sgemm`/`Strsm`/`Ssyrk`/`Sgemv`. The advanced factorizations
(`QR32`, `SVD32`, `EigenSym32`, `Eigen32`) use a float64 bridge to gonum and
*do* cast internally (gonum has no float32 LAPACK). float32 carries ~7 digits,
so `Det` overflows for moderate sizes — prefer the solves.

Measured (M5 Pro) vs stock gonum/mat float64:

| Operation | gonum float64 | goblas `mat32` float32 | Speedup |
|-----------|---------------|------------------------|---------|
| `Mul` (n=1024) | 27.1 ms | **3.3 ms** | **8.2×** |
| `Cholesky` solve (n=512) | 7.1 ms | **2.5 ms** | **2.9×** |

## Getting started

goblas is an ordinary Go module — there's nothing to compile or install separately.
You add it as a dependency and the Go toolchain builds it from source into your
program (picking the NEON assembly or the portable fallback automatically for
whatever you're building for).

**1. Have a Go module.** In your project directory:

```sh
go mod init github.com/you/yourproject   # skip if you already have a go.mod
```

**2. Add goblas.** Either run `go get`, or just import it and run `go mod tidy`:

```sh
go get github.com/nakurai/goblas@latest
```

**3. Use it.** Two ways, depending on what you want:

### A. Call the BLAS routines directly

```go
package main

import (
	"fmt"

	"github.com/nakurai/goblas"
)

func main() {
	// C = A * B for two 2x2 column-major matrices.
	a := []float64{1, 2, 3, 4} // [[1,3],[2,4]]
	b := []float64{5, 6, 7, 8} // [[5,7],[6,8]]
	c := make([]float64, 4)
	goblas.Dgemm(goblas.NoTrans, goblas.NoTrans, 2, 2, 2, 1, a, 2, b, 2, 0, c, 2)
	fmt.Println(c) // [23 34 31 46]
}
```

### B. Accelerate the whole Gonum stack (recommended for most users)

If you want matrices, vectors, `Solve`, `Cholesky`, `QR`, `SVD`, etc., you almost
certainly want [`gonum/mat`](https://pkg.go.dev/gonum.org/v1/gonum/mat) — the
standard Go linear-algebra library — with goblas plugged in underneath it. Call
`blasadapt.Use()` once at startup and every Gonum operation runs on goblas
kernels (see [Accelerating Gonum](#accelerating-gonum-high-level-linear-algebra)
above):

```go
import (
	"github.com/nakurai/goblas/blasadapt"
	"gonum.org/v1/gonum/mat"
)

// Register goblas as the BLAS for the whole process. Put this in an init()
// or at the top of main(), before any gonum/mat work.
func init() { blasadapt.Use() }
```

`Use()` is a single, explicit call rather than an import side-effect, so *you*
decide when (and whether) goblas takes over the process-wide BLAS.

That's the entire setup. **No CGo, no C compiler, no system libraries, no build
flags** — it works anywhere Go does, and goes fast on Apple Silicon automatically.

### Where the documentation lives

The full, always-up-to-date API reference is generated automatically from the
source and published at **[pkg.go.dev/github.com/nakurai/goblas](https://pkg.go.dev/github.com/nakurai/goblas)**
— every exported function, its parameters, and the package conventions. You can
also read it offline:

```sh
go doc github.com/nakurai/goblas            # package overview + function list
go doc github.com/nakurai/goblas.Dgemm      # one routine's signature + docs
```

## Usage

All matrices are **column-major** (Fortran order): element A(i,j) lives at `a[i + j*lda]`, where `lda` is the leading dimension (column stride, must be ≥ number of rows). Vectors take an increment (`incX`); a value of 1 means unit-stride (contiguous), which is also the fast path for the NEON kernels.

```go
import "github.com/nakurai/goblas"

// --- Level 1: vector operations ---

// Dot product: returns x·y
result := goblas.Ddot(n, x, 1, y, 1)

// y = alpha*x + y
goblas.Daxpy(n, alpha, x, 1, y, 1)

// x = alpha*x
goblas.Dscal(n, alpha, x, 1)

// Euclidean norm
norm := goblas.Dnrm2(n, x, 1)

// Sum of absolute values
asum := goblas.Dasum(n, x, 1)

// Index of element with largest absolute value
idx := goblas.Idamax(n, x, 1)

// Copy x into y
goblas.Dcopy(n, x, 1, y, 1)

// Swap x and y
goblas.Dswap(n, x, 1, y, 1)

// --- Level 2: matrix-vector ---

// y = alpha*A*x + beta*y  (A is m×n, column-major with leading dimension lda)
goblas.Dgemv(goblas.NoTrans, m, n, alpha, a, lda, x, 1, beta, y, 1)

// y = alpha*Aᵀ*x + beta*y
goblas.Dgemv(goblas.Trans, m, n, alpha, a, lda, x, 1, beta, y, 1)

// --- Level 3: matrix-matrix ---

// C = alpha*A*B + beta*C  (all column-major)
goblas.Dgemm(goblas.NoTrans, goblas.NoTrans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)

// C = alpha*Aᵀ*B + beta*C
goblas.Dgemm(goblas.Trans, goblas.NoTrans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
```

### Allocating column-major matrices

```go
// Allocate an m×n matrix A (lda = m for a compact layout)
a := make([]float64, m*n)

// Set A(i, j) = value
a[i + j*m] = value

// Read A(i, j)
value := a[i + j*m]
```

## Running benchmarks

```sh
# Quick comparison: generic (pure-Go) vs NEON vs Gonum, all routines
go test -tags gonumbench -run '^$' -bench '.' -benchtime=1s ./...

# Stable GFLOPS numbers (recommended before tuning decisions)
go test -tags gonumbench -run '^$' -bench '.' -benchtime=10s ./...

# Just a specific routine (faster)
go test -tags gonumbench -run '^$' -bench 'Dgemm|Dgemv|Ddot' -benchtime=2s ./...

# Internal generic-vs-NEON comparison (no Gonum dependency)
go test -run '^$' -bench '.' -benchtime=1s ./internal/kernel/
```

The `-tags gonumbench` flag enables Gonum comparison benchmarks (gated to keep Gonum out of the core dependency graph).

## Documentation

The [`docs/`](docs/) folder holds conceptual and instructional guides beyond this README:

- **[docs/overview.md](docs/overview.md)** — what goblas is, why it exists, and an honest
  limitations section separating what is *not done yet* from what is *inherent to the idea*.
- **[docs/adding-new-cpu.md](docs/adding-new-cpu.md)** — extend goblas to a new processor,
  written for someone with no CPU-architecture background (two tiers: tuning-only, or a full
  SIMD kernel).
- **[docs/verify-benchmark.md](docs/verify-benchmark.md)** — everything testable and every
  benchmark, with copy-followable commands.
- **[docs/use-cases/](docs/use-cases/README.md)** — teaching tutorials that build real ML
  algorithms on goblas + gonum/mat and explain *where* the acceleration comes from: linear &
  logistic regression, KNN, kernel SVM, MLP / CNN / LSTM neural networks, and reservoir
  computing.
- **[docs/plan.md](docs/plan.md)** — the full development history and roadmap.

The API reference is auto-generated at
[pkg.go.dev/github.com/nakurai/goblas](https://pkg.go.dev/github.com/nakurai/goblas).

## Architecture

The library is designed so that adding support for a new CPU is exactly: **one kernel file + one registration**. No public API changes, no callers updated.

```
goblas/
  blas.go          // public API (validates args, delegates to active kernel)
  dispatch.go      // selects active Kernel at init via internal/cpu.Detect()
  doc.go           // package overview + column-major conventions
  blasadapt/
    adapter.go               // row-major blas.Float64 over goblas; blasadapt.Use() for gonum
  internal/
    cpu/
      cpu.go                   // CPU struct: Microarch, HasNEON, HasAVX2FMA, L1DBytes
      cpu_darwin_arm64.go      // sysctl detection for Apple Silicon (chip + L1d size)
      cpu_arm64.go             // generic ARM64 (non-Apple): NEON assumed
      cpu_amd64.go             // x86-64: AVX2+FMA detection via golang.org/x/sys/cpu
      cpu_other.go             // all other platforms (!arm64 && !amd64)
    kernel/
      kernel.go                // Kernel interface (all routines)
      generic.go               // genericKernel: pure-Go, always correct, universal fallback
      generic_l1.go/_l2.go/_l3.go  // pure-Go implementations by level
      l3_tri.go                // recursive Dsyrk/Dtrsm/Dsymm/Dtrmm (bulk on Dgemm)
      dgemm_driver.go          // shared tiled/parallel blocking driver (all kernels)
      arm64.go                 // neonKernel embeds genericKernel + Select() for ARM64
      ddot_arm64.s/.go         // NEON ddot + go:noescape stub
      daxpy_arm64.s/.go        // NEON daxpy + stub
      dscal_arm64.s/.go        // NEON dscal + stub
      dasum_arm64.s/.go        // NEON dasum (abs via sign-mask, FMLA-ones sum)
      dnrm2_arm64.s/.go        // NEON sum-of-squares + guarded fallback
      dgemv_arm64.s/.go        // NEON dgemv + stub
      dger_arm64.go            // NEON dger (reuses the daxpy kernel per column)
      dtrsv_arm64.go           // NEON dtrsv (reuses the daxpy/ddot kernels)
      dgemm_arm64.s/.go        // NEON 8x4 micro-kernel + kernel selection vars
      dgemm8x6_arm64.s         // NEON 8x6 micro-kernel (default on ARM64)
      avx2_amd64.go            // avx2Kernel + Select() for x86-64 (experimental)
      dgemm_amd64.s            // AVX2 8x4 micro-kernel (VFMADD231PD)
      select_other.go          // !arm64 && !amd64: always returns genericKernel
    accel/    openblas/        // CGo benchmark wrappers (build-tagged, not in normal builds)
```

**Key design:** `neonKernel`/`avx2Kernel` embed `genericKernel`. Any routine without an assembly override automatically falls back to the pure-Go reference — no `if asm { } else { }` scattered through the code. Adding a new routine in assembly means adding one `.s` file, one stub `.go`, and one method on the accelerated kernel. See [docs/adding-new-cpu.md](docs/adding-new-cpu.md) for the full walkthrough.

### CPU detection (Apple Silicon)

On darwin/arm64 the library reads `sysctl` at startup to identify the M-series chip and read the **performance-core L1 data cache size** (used to size `dgemm` tiles). On the M5 Pro this is 128 KB. On all other platforms it falls back to the pure-Go path with no detection overhead.

### Assembly notes (for contributors)

Go's ARM64 assembler has narrower NEON float support than raw ARM64 assembly. Key findings from implementation:

- ✅ Available: `VLD1.P`, `VST1.P`, `VFMLA` (fused multiply-add), `VDUP` (lane broadcast, including `VDUP Rn, Vd.D2` from a general register), `VAND`/`VEOR` (bitwise — e.g. sign-bit clear for `abs`), scalar `FADDD`/`FMULD`/`FMOVD`/`FABSD`
- ❌ **`VADD .D2` is integer add**, not floating-point — silently produces garbage on float data
- ❌ `VFADD`, `VFMUL`, `FADDP` are unrecognized by the assembler
- ❌ `VFMAX` and `VCMHI` (vector FP max / vector compare) are also unrecognized — there is no clean way to vectorize an argmax (`Idamax`), so it stays pure-Go
- Idioms: accumulate a sum with `VFMLA` against a `[1.0, 1.0]` ones vector (no vector FP add); horizontal-reduce with `VDUP V0.D[1], V1.D2` then `FADDD F1, F0, F0` (not stack spill); compute `abs` with `VAND` against a `0x7FFF…` mask

## Next steps (roadmap)

### Phase 11 — x86-64 AVX2 dgemm kernel (✅ written, awaiting x86 hardware validation)

- **8×4 AVX2+FMA micro-kernel** ([dgemm_amd64.s](internal/kernel/dgemm_amd64.s)): each tile
  column is a YMM pair (4 float64 per register), 8 accumulators, `VBROADCASTSD` for B,
  `VFMADD231PD` for the FMAs, real `VADDPD` for the writeback — 64 FLOPs per k-iteration,
  same packed-panel format and shared blocked driver as the NEON kernels.
- **CPU detection** ([cpu_amd64.go](internal/cpu/cpu_amd64.go)): AVX2+FMA via CPUID
  (`golang.org/x/sys/cpu`); hosts without it (pre-2013) keep the pure-Go path. Conservative
  kc=256/mc=16 blocking sized for the typical 32 KB x86 L1d.
- **Verification status:** this was developed on an ARM64 machine, so the assembly is
  verified statically — `go vet`'s asmdecl check, cross-compiled test binaries, and an
  instruction-level disassembly review (LLVM objdump confirms every encoding and operand
  order). The runtime correctness test (`TestDgemmAVX2MatchesGeneric`, which fuzzes the AVX2
  kernel against the pure-Go reference across shapes/transposes/edges) compiles but needs an
  actual x86-64 machine: `GOARCH=amd64 go test ./...` on any Intel/AMD box, or x86 CI.
  Treat the AVX2 path as **experimental until that has run**.

### Future work

- **New CPU targets:** Adding new CPUs. This has to be done on machines with different chips to tune the simd instructions properly.

## Platform support

| Platform | Status |
|----------|--------|
| Apple Silicon (M-series, darwin/arm64) | ✅ NEON, tuned blocking (M5 Pro sweep) |
| Any ARM64 (linux/arm64, Graviton, etc.) | ✅ NEON, conservative blocking sized to detected L1d |
| x86-64 with AVX2+FMA (2013+) | ⚠️ AVX2 dgemm kernel (experimental — statically verified, needs a test run on real x86 hardware) |
| x86-64 without AVX2 | ✅ Tiled + parallel pure-Go (~100 GFLOPS-class dgemm) |
| Any other Go platform | ✅ Tiled + parallel pure-Go fallback |
