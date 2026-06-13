# Support vector machines (the kernel matrix)

A support vector machine (SVM) is a powerful classifier, and **kernel** SVMs are where the
"BLAS-heavy" part lives. This tutorial is honest about which half of an SVM goblas accelerates
(the kernel/Gram matrix — the part that dominates as your dataset grows) and which half it does
not (the solver). Read [knn.md](knn.md) first; the kernel matrix is built much like the distance
matrix there.

## The intuition: the widest street

A linear SVM separates two classes with a boundary, but not just any boundary — the one with the
**widest margin**, the line that leaves as much empty space ("street") as possible on either
side. The points that touch the edges of the street are the **support vectors**; they alone
define the boundary.

```
   o   o                 o = class +1
     o   \  margin       x = class −1
       o  \  /
   ─────────\/────────   ← the maximum-margin boundary
          /  \
        x  \   x
       x    \  x
```

## The kernel trick: curved boundaries for free

A straight line cannot separate classes that are tangled (imagine one class forming a ring
around the other). The **kernel trick** handles this: instead of comparing two points by their
plain dot product `xᵢ · xⱼ`, we compare them with a **kernel function** `K(xᵢ, xⱼ)` that
measures similarity in a richer, implicitly higher-dimensional space — without ever building
that space. A popular choice is the **RBF (Gaussian) kernel**:

```
K(xᵢ, xⱼ) = exp(−γ · ‖xᵢ − xⱼ‖²)
```

— "two points are similar (≈1) when close, dissimilar (≈0) when far." With a kernel, the SVM can
draw curved boundaries while its math stays the same.

## Where the cost is: the Gram matrix

Training a kernel SVM needs the kernel value between **every pair** of training points. That is
the **Gram matrix** (or kernel matrix) `K`, an `n × n` matrix with `K[i][j] = K(xᵢ, xⱼ)`. For
`n` training points this is `n²` kernel evaluations, each touching `d` features — `n²·d` work.
**This is the dominant cost for large `n`,** and it is exactly the kind of all-pairs computation
that becomes a matrix multiply.

For the RBF kernel, the same trick as KNN applies. The exponent needs all pairwise squared
distances, and `‖xᵢ−xⱼ‖² = ‖xᵢ‖² + ‖xⱼ‖² − 2·xᵢ·xⱼ`. The cross term `xᵢ·xⱼ` over all pairs is:

```
X · Xᵀ        →  an n × n matrix of all dot products  →  one Dgemm (or Dsyrk, since it is symmetric)
```

Because `X·Xᵀ` is symmetric, goblas can compute it with **`Dsyrk`** (symmetric rank-k update),
which does roughly half the work of a full `Dgemm` by exploiting the symmetry. That is the
goblas-accelerated heart of kernel-SVM training.

## Building the kernel matrix with gonum/mat

Setup:

```go
import (
    "math"
    "github.com/nakurai/goblas/blasadapt"
    "gonum.org/v1/gonum/mat"
    "gonum.org/v1/gonum/floats"
)

func init() { blasadapt.Use() }
```

Training points, one per row, and their squared norms:

```go
X := mat.NewDense(n, d, trainData)

sq := make([]float64, n)
for i := 0; i < n; i++ {
    row := X.RawRowView(i)
    sq[i] = floats.Dot(row, row) // ‖xᵢ‖²
}
```

The all-pairs dot products via goblas. `SymDense.SymRankK` computes `α·X·Xᵀ` directly into a
symmetric matrix — this is the `Dsyrk` call:

```go
gram := mat.NewSymDense(n, nil)
gram.SymRankK(gram, 1, X) // gram = X · Xᵀ  (symmetric, via Dsyrk)
```

Now turn dot products into RBF kernel values, using the squared-distance identity:

```go
gamma := 0.5
K := mat.NewDense(n, n, nil)
K.Apply(func(i, j int, _ float64) float64 {
    d2 := sq[i] + sq[j] - 2*gram.At(i, j) // ‖xᵢ−xⱼ‖²
    return math.Exp(-gamma * d2)          // RBF kernel
}, K)
```

`K` is the Gram matrix. For a **linear** SVM you would skip the exponential and use `gram`
directly as the kernel.

## The honest part: the solver is not BLAS work

Building `K` was dense linear algebra and goblas-fast. But *training* the SVM from `K` means
solving a constrained **quadratic optimization** problem to find which points are support
vectors and with what weights. The classic algorithm, **SMO** (Sequential Minimal
Optimization), works by repeatedly picking a pair of points and adjusting them — it is iterative
and largely scalar, **not** a matrix multiply. goblas does not accelerate that inner solver.

Why is the SVM still a fair goblas use case? Because for non-trivial dataset sizes, **forming
and repeatedly accessing the `n×n` kernel matrix dominates the runtime** — the solver's
per-step work is small next to the `n²·d` kernel construction. Speed up the Gram matrix and you
speed up the practical bottleneck. (For *prediction*, scoring a batch of new points against the
support vectors is again a matrix multiply — `Dgemm` — so goblas helps there too.)

This is the nuance the [use-cases index](README.md) promised: we keep SVM because the part that
*scales* is genuinely BLAS-bound, while being upfront that the optimizer itself is not.

## Where goblas earned its keep

| Step | BLAS routine | goblas role |
|------|--------------|-------------|
| All-pairs dot products `X·Xᵀ` | `Dsyrk` | accelerated — the dominant cost |
| Per-point squared norms | `Ddot` | accelerated (cheap) |
| Dot products → kernel values | elementwise `Apply` | plain Go (cheap) |
| SMO / QP solver | — | **not BLAS work**, not accelerated |
| Batched prediction | `Dgemm` | accelerated |

## Recap

- A kernel SVM draws curved maximum-margin boundaries via a kernel function.
- Its scaling cost is the `n×n` Gram matrix; for RBF that is the KNN distance trick again, and
  `X·Xᵀ` is a symmetric **`Dsyrk`** on goblas.
- The SMO solver that follows is iterative and scalar — goblas does not speed it up, but it is
  the smaller cost at scale.

Next: the neural-network tutorials, where *every* layer is a matrix multiply — start with
[neural-net-mlp.md](neural-net-mlp.md).
