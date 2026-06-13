# mat32 fundamentals — float32 matrices and solves

The other tutorials use `gonum/mat`, which is **float64**. When your data is already
**float32** — sensor streams, image pixels, neural-network activations — casting it up to
float64 wastes memory and bandwidth (and float64 isn't available on GPUs anyway). For those
cases goblas ships its own native single-precision matrix package,
[`mat32`](../../mat32), so float32 stays float32 end to end. This is goblas's "LAPACK layer":
a matrix type plus the linear-algebra solves built on it.

This page is the float32 counterpart of [gonum-fundamentals.md](gonum-fundamentals.md). Read
that one first if `gonum/mat` is new to you — `mat32` deliberately mirrors its API, so most of
what you learn there transfers directly.

## Why a separate package (and not just `gonum/mat`)

gonum has **no float32 linear algebra**: there is no `lapack32`, and `gonum/mat` is float64-only
(no `mat32`, no `.Solve32()`). So a native float32 matrix type has to live in goblas. `mat32`
provides it, with the same shapes and method names you already know.

## Setup — no `blasadapt.Use()` needed

Unlike the gonum tutorials, you do **not** call `blasadapt.Use()` for the native float32 path:
`mat32` calls the goblas single-precision kernels (`Sgemm`, `Strsm`, `Ssyrk`, `Sgemv`, …)
directly. Just import it:

```go
import "github.com/nakurai/goblas/mat32"
```

(The one exception is the float64 *bridge* used by `QR32`/`SVD32`/`Eigen32`; see the last
section. There, registering `blasadapt.Use()` makes the bridge's float64 BLAS goblas-accelerated
too.)

## The types

| Type | What it is | gonum/mat analogue |
|------|-----------|--------------------|
| `mat32.Dense32` | a dense row-major float32 matrix | `mat.Dense` |
| `mat32.VecDense32` | a float32 column vector | `mat.VecDense` |
| `mat32.SymDense32` | a symmetric float32 matrix (Cholesky input) | `mat.SymDense` |

Construct them like their gonum cousins (row-major data, or `nil` for a fresh zero matrix):

```go
// A 2×3 matrix, row-major: [[1 2 3] [4 5 6]].
A := mat32.NewDense32(2, 3, []float32{1, 2, 3, 4, 5, 6})
x := mat32.NewVecDense32(3, []float32{1, 0, -1})

r, c := A.Dims()        // 2, 3
v := A.At(0, 2)         // 3
A.Set(1, 1, 9)          // mutate in place
```

## Arithmetic — all end-to-end float32

The matrix operations mirror `gonum/mat` and run their heavy steps on the goblas S-kernels — no
float64 casting anywhere on this path:

```go
var C mat32.Dense32
C.Mul(A, B)             // C = A·B            → goblas Sgemm
C.Add(A, B)             // C = A + B          (elementwise)
C.Sub(A, B)             // C = A − B
C.Scale(2.0, A)         // C = 2·A
C.MulElem(A, B)         // Hadamard product
C.Apply(func(i, j int, v float32) float32 { // elementwise function
    if v > 0 { return v }                    // e.g. ReLU
    return 0
}, A)

var y mat32.VecDense32
y.MulVec(A, x)          // y = A·x            → goblas Sgemv

n := C.Norm()           // Frobenius norm     → goblas Snrm2
```

`A.T()` is a lazy transpose view, so `C.Mul(A.T(), B)` computes `Aᵀ·B` with no copy — exactly
like `gonum/mat`.

## Solving linear systems — native float32 Cholesky and LU

This is the "LAPACK layer." Two factorizations are implemented **natively in float32** (their
trailing FLOPs run on goblas `Ssyrk`/`Strsm`/`Sgemm`), so a solve on float32 data never touches
float64:

**General system `A·x = b`** — LU with partial pivoting, via `Dense32.Solve`:

```go
var x mat32.Dense32
if err := x.Solve(A, b); err != nil { // A square; b is n×nrhs
    log.Fatal(err)
}
```

**Symmetric positive-definite system** — Cholesky (faster, and the right tool for normal
equations / covariance matrices):

```go
sym := mat32.SymDense32FromDense(A, true) // use the upper triangle of A
var chol mat32.Cholesky32
if !chol.Factorize(sym) {
    log.Fatal("not positive definite")
}
var x mat32.Dense32
chol.SolveTo(&x, b)                       // x = A⁻¹·b
```

A worked end-to-end example (ridge regression in float32) lives in the package's
[`Example_ridgeRegression`](../../mat32/example_test.go).

> **Precision caveat.** float32 carries only ~7 significant digits. The solves are fine for
> well-conditioned problems (and that's most real data), but ill-conditioned systems lose
> accuracy faster than in float64. In particular `Det` overflows to ±Inf for even moderately
> sized matrices — prefer the solves over forming a determinant.

## The advanced factorizations — QR, SVD, Eigen (these *do* cast)

`QR32`, `SVD32`, `EigenSym32` and `Eigen32` exist, but they are **not** end-to-end float32: under
the hood they cast to float64 and call gonum's LAPACK (which has no float32 version), then cast
the result back. They are there for completeness and convenience, not for the no-cast guarantee.

```go
import "github.com/nakurai/goblas/blasadapt"

func init() { blasadapt.Use() } // makes the bridge's float64 BLAS run on goblas

var svd mat32.SVD32
svd.Factorize(A, mat.SVDThin)
vals := svd.Values(nil)         // []float32 singular values
```

If you need single precision *through* a QR/SVD/Eigen, that requires a native float32 LAPACK,
which goblas does not have yet — use the float64 bridge here, or stay in `gonum/mat`.

## When to use which

- **float32 data, matrix arithmetic or a Solve/Cholesky** → `mat32` (no casting, faster, less
  memory). This is the sweet spot, and it's what the [MLP](neural-net-mlp.md),
  [CNN](neural-net-cnn.md), [KNN](knn.md), and [YOLO](yolo-object-detection.md) tutorials use.
- **float64 data, or you need QR/SVD/Eigen in full precision, or rich factorization APIs** →
  `gonum/mat` with `blasadapt.Use()` (the other tutorials).

Both can coexist in one program; pick per matrix based on where the data comes from.
