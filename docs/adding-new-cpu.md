# Adding support for a new CPU

This guide is for someone who wants goblas to run *faster* on a processor it has not been
tuned for — and who does **not** necessarily know anything about CPU architecture or assembly.
It starts from zero and tells you exactly which files to touch.

## The 60-second mental model

A **kernel** in goblas is just "a complete set of the BLAS routines." There are several:

- a **generic kernel** written in plain Go that is always correct and runs everywhere, and
- **accelerated kernels** that replace a few hot routines with faster, CPU-specific code.

The faster code is **assembly** that uses **SIMD** instructions. SIMD ("Single Instruction,
Multiple Data") means one instruction that does the same arithmetic on several numbers at
once — e.g. multiply four pairs of float64s in one step instead of four. That parallelism is
where the speed comes from, and the Go compiler does not do it automatically, which is why we
hand-write it.

The single most important thing to understand is the **safety net**:

> Accelerated kernels *embed* the generic kernel and *override* only the routines they have
> a faster version of. Anything they do not override automatically uses the correct Go code.

So you can add as much or as little as you want. If you only speed up one routine, every
other routine still works. And if your fast code were ever wrong, the fix is to not register
it — you can never make goblas produce wrong answers by *failing* to add a kernel.

This leads to **two tiers** of contribution. Tier 1 needs no assembly at all.

---

## Tier 1 — no assembly: detection + tuning (start here)

If your CPU shares an instruction set with one goblas already has a kernel for — most
commonly **another ARM64 chip** (a Graviton server, a Snapdragon laptop, a Raspberry Pi) —
then the NEON assembly *already runs on it*. What it lacks is **tuning**: the cache-blocking
sizes that decide how the matrix is cut into pieces are picked for the Apple M5 Pro. Getting
those right for your chip's cache can be a large speedup, and it is pure Go config.

### What you change

Everything lives in two places.

**1. CPU detection — `internal/cpu/`.** Each platform has a `Detect()` function behind a build
tag. It returns a `CPU` struct (defined in `cpu.go`):

```go
type CPU struct {
    Microarch  Microarch // which family of chip
    HasNEON    bool      // ARM64 SIMD available?
    HasAVX2FMA bool      // x86-64 SIMD available?
    L1DBytes   int       // per-core L1 data cache size, 0 if unknown
}
```

The Apple detector (`cpu_darwin_arm64.go`) reads the chip name and L1 cache size from the OS
via `sysctl`. The generic ARM64 detector (`cpu_arm64.go`, build tag `arm64 && !darwin`) just
reports `HasNEON: true` and leaves the cache size unknown. To recognize *your* chip, you would
add a branch (or a new `Microarch` constant in `cpu.go`) and fill in `L1DBytes` if you can
read it.

**2. Kernel selection + blocking — `internal/kernel/arm64.go`.** The `Select()` function turns
a detected `CPU` into a kernel and sets the two blocking parameters:

```go
var (
    dgemmKC = 512 // how deep a slice of the shared dimension each pass handles
    dgemmMC = 24  // how many rows of A are packed into cache at once
)
```

The rule of thumb the code already uses: a packed block of A is `dgemmMC × dgemmKC × 8` bytes
and should fit inside the L1 data cache. `Select()` already derives a conservative `dgemmMC`
from `L1DBytes` when it is known. To tune for your chip, you measure (see
[verify-benchmark.md](verify-benchmark.md) — `BenchmarkDgemmBlockSweep` sweeps these values
for you) and set the winners for your `Microarch`.

### How to know it worked

Run `BenchmarkDgemmBlockSweep` on your machine, pick the fastest `kc`/`mc`, wire them in, and
re-run the normal benchmarks. Correctness is already guaranteed — you did not touch any math.

That is a complete, useful contribution. You can stop here.

---

## Tier 2 — a new SIMD kernel (the hard path)

You need this only when your CPU has a **different instruction set** with no existing kernel —
for example, you want RISC-V vector kernels, or you are the person who will finally *run and
fix* the x86-64 AVX2 kernel on real Intel hardware. The freshly-written AVX2 kernel is the
worked example to copy; here is the shape of the job.

### The micro-kernel contract

All the cache-blocking, packing, threading, and edge handling is already written once, shared
by every kernel, in `internal/kernel/dgemm_driver.go`. You do **not** rewrite that. You write
only the innermost loop — the **micro-kernel** — and it has a fixed, simple contract:

> Compute `C[MR × NR] += A_panel × B_panel`, where the driver hands you data already laid out
> for you: `A_panel` is `k` consecutive groups of `MR` float64 (a vertical strip of A,
> column by column), `B_panel` is `k` consecutive groups of `NR` float64 (a horizontal strip
> of B, row by row), and `C` is column-major with a leading dimension `ldc` you are told.

`MR` and `NR` are the tile dimensions (8×4/8×6 for the float64 kernels, 8×8 for the float32 one).
Because the panels are pre-packed into contiguous memory, your inner loop is just: load `MR`
values of A, broadcast each of `NR` values of B, multiply-accumulate into `MR×NR` running totals,
repeat `k` times, then add the totals into C. That is the entire algorithm. The packing that
makes this possible is `packAPanels` / `packBPanels` in the driver.

The driver, packing, and triangular/symmetric recursion are **generic over the element type**
(`gemmBlocked[T float32|float64]`, `microKernel[T]`), so a new architecture can add a float64
kernel, a float32 kernel, or both — each is just a micro-kernel satisfying the same contract at
its element type, registered in `Select` (float64) / `Select32` (float32). `Isamax`/`Idamax`
stay on the pure-Go path everywhere (no vectorizable argmax in Go's assembler).

### The files you add (mirroring the amd64 example)

For an architecture `arch`, the existing pattern is five files:

| File | Purpose |
|------|---------|
| `internal/cpu/cpu_arch.go` | `Detect()` for the new arch: report the SIMD feature flag + cache size. (For amd64: `cpu_amd64.go`, detecting AVX2+FMA via `golang.org/x/sys/cpu`.) |
| `internal/kernel/dgemm_arch.s` | the SIMD micro-kernel in Go assembly. (Example: `dgemm_amd64.s`.) |
| `internal/kernel/avx2_amd64.go` (or `arch.go`) | the Go side: a kernel struct embedding `genericKernel`, a `//go:noescape` stub declaring the assembly function, the `Dgemm`/`Dsyrk`/`Dtrsm` overrides, and the `Select()` that registers it when the feature flag is present. |
| `internal/kernel/platform_arch_test.go` | adds your kernel to `platformKernels()` so the shared tests exercise it. |
| `internal/kernel/dgemm_arch_test.go` | `TestDgemm<ARCH>MatchesGeneric`: fuzz your kernel against the generic reference across shapes, transposes, edges, and alpha/beta values. |

The build tags (`//go:build amd64`, etc.) are what make the right files compile for the right
target; `select_other.go` carries the `!arm64 && !amd64` fallback and you widen its tag to
exclude your new arch too.

### Verifying assembly you cannot run

You do not need to own the hardware to get real confidence. The AVX2 kernel was validated
entirely from an ARM64 Mac:

1. **`GOARCH=arch go vet ./...`** — Go's assembler runs, and the `asmdecl` checker confirms
   your assembly's stack frame and argument offsets match the Go function declaration. This
   catches a huge class of mistakes.
2. **Cross-compile the tests** — `GOARCH=arch go test -c` proves everything links.
3. **Disassemble and eyeball it** — dump the built binary and read the instructions back to
   confirm the encodings and, critically, the **operand order** (Go's assembler reverses
   Intel's source/destination convention, a classic trap). LLVM's `objdump` even annotates
   what each FMA computes.

The runtime correctness test (`TestDgemm<ARCH>MatchesGeneric`) still needs the real chip — it
should `t.Skip()` when the feature flag is absent — but the above gets you most of the way and
is exactly how the experimental AVX2 path stands today.

---

## Recap

- **Most contributors want Tier 1:** recognize your chip in `internal/cpu/`, set good
  `dgemmKC`/`dgemmMC` in `Select()`, measure with the block-sweep benchmark. No assembly, no
  risk of wrong answers.
- **Tier 2 is one micro-kernel:** implement the packed-panel contract from `dgemm_driver.go`
  in SIMD assembly, register it in `Select()`, mirror the test files, and lean on
  vet + disassembly when you cannot run the target.
- The generic kernel guarantees correctness throughout, so you can ship a partial kernel and
  it will simply fall back for anything you have not written.

Next: [verify-benchmark.md](verify-benchmark.md) to test and measure whatever you add.
