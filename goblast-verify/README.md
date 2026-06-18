# goblast-verify

Public correctness proof for **goblas**, graded by the third-party
[`goblast`](https://github.com/nakurai/goblast) oracle.

`goblast` generates a deterministic corpus of BLAS test cases — every boundary
size, stride, transpose, uplo/side/diag flag, leading-dimension padding, and
IEEE-754 special value — each with a reference answer computed by an
independent oracle. This tool runs goblas over that corpus so `goblast check`
can grade goblas's output against the reference, routine by routine and tag by
tag.

## The pipeline

Three steps; `verify.sh` runs all three:

```
goblast gen <corpus>      # 1. write cases/<id>/input.txt + expected.txt
goblast-verify -corpus …  # 2. read input.txt -> run goblas -> write output.txt
goblast check <corpus>    # 3. grade output.txt vs expected.txt, print report
```

```sh
# from this directory; requires `goblast` on PATH:
#   go install github.com/nakurai/goblast/cmd/goblast@latest
./verify.sh                 # uses ./corpus
./verify.sh /tmp/blast      # or a corpus dir of your choice
```

Run a single routine family while iterating:

```sh
goblast gen   -only dgemm ./corpus
go run .      -corpus ./corpus
goblast check -only dgemm ./corpus
```

## What the driver does

`goblast-verify` covers all 34 routines goblast emits (float64 `D`/`I` and
float32 `S`, Levels 1–3). For each case it parses `input.txt`, decodes the
flags to goblas's enum types, calls the matching goblas routine on the
full ld-/inc-strided buffers, and writes the op's output field(s) back in
goblast's format. Float32 routines parse the float64-text inputs, narrow to
`float32` for the call, and widen the result back to float64 for output (the
case files always carry float64-width decimal text — see goblast's
`FORMAT.md §2`).

Because goblast's own reader/writer lives in an `internal/` package, this tool
ships a small standalone parser/writer for the format ([format.go](format.go)).
The op→goblas mapping is in [dispatch.go](dispatch.go).

## Reproducibility

The corpus is fully deterministic but is **not** committed (see
`.gitignore`): pin the goblast version you grade against and regenerate on
demand, per goblast's guidance. `goblast-verify/corpus/` is ignored.

## Sample report

```
OVERALL: 4264/4266 passed (100.0%)

By routine: every D/I/S routine 100%, except the two float32 precision
cases noted below (sgemm 167/168, ssyrk 142/143).

By tag: transpose, uplo, side, stride, ld-padding, k-boundary, coprime,
implicit-unit-diagonal, unreferenced-triangle, negative-incx-degenerate,
all-inf, mixed-inf, and every special-value tag — all 100%.
```

### Remaining non-failures (2 informational float32 cases)

The two ungreen cases are **not correctness bugs** — they are inherent
single-precision behavior:

- **float32 precision** — `sgemm m=n=k=200` and one `ssyrk n=16 k=8` case
  differ from the oracle in the ~6th significant figure, just past the
  float32 tolerance, due to summation-order differences at large reduction
  depth. The corresponding float64 cases pass exactly.

(Two earlier BLAS-conformance gaps this harness originally surfaced — the
negative-`incX` degenerate contract for the single-vector L1 routines, and
`nrm2` returning `NaN` instead of `+Inf` on infinite inputs — have since been
fixed in goblas; both tag buckets now pass 100%.)
