# Use-case tutorials

These tutorials teach you to build real machine-learning algorithms **on top of goblas** — and,
just as importantly, to *understand* what you are building. They assume **no prior machine-
learning knowledge and no Gonum experience**. Each one starts from the intuition, derives the
math you need (no more), and then shows how to express it with [`gonum/mat`](https://pkg.go.dev/gonum.org/v1/gonum/mat),
the standard Go matrix library, running on goblas kernels.

## How goblas fits in (read this once)

goblas does not provide a machine-learning API. It provides a very fast **BLAS** — the dense
matrix and vector arithmetic that ML algorithms are *made of*. The standard way to use it is:

```go
import (
    "github.com/nakurai/goblas/blasadapt"
    "gonum.org/v1/gonum/mat"
)

func init() { blasadapt.Use() } // makes every gonum/mat operation run on goblas
```

That single `init` registers goblas as the BLAS for the whole Gonum stack. From then on, when
you call `m.Mul(a, b)` or factorize a matrix, the heavy lifting runs on goblas's tuned kernels.
**You write ordinary `gonum/mat` code; goblas makes it fast underneath.** Every tutorial below
assumes you have done this once in your program.

### Important: you never call goblas directly in these tutorials

There are two packages that share the "goblas" name, and it is worth being clear about which
one you use:

| Package | What it is | When you use it |
|---------|-----------|-----------------|
| `github.com/nakurai/goblas/blasadapt` | the **adapter** — `blasadapt.Use()` plugs goblas in under Gonum | the **float64** tutorials — then write pure `gonum/mat` code |
| `github.com/nakurai/goblas/mat32` | the native **float32** matrix type (`Dense32`, `Cholesky32`, …) | the **float32** tutorials — when data is already single precision |
| `github.com/nakurai/goblas` | the **raw BLAS** — low-level functions like `goblas.Dgemm`/`Sgemm` on flat slices | only if you have raw slices and want a single BLAS call without a matrix type |

Most float64 tutorials import **only `blasadapt`** (for the one-time `Use()` call) and otherwise
write `gonum/mat` code — `m.Mul(...)`, `m.Solve(...)`, `chol.Factorize(...)`. The **float32**
tutorials instead import `mat32` and write `Dense32` code (`m.Mul(...)`, `chol.Factorize(...)`);
`mat32` calls the goblas single-precision kernels directly, so it needs no `Use()` registration.
See [goblas-mat32-fundamentals.md](goblas-mat32-fundamentals.md).

So when a tutorial says *"this is a `Dgemm` on goblas"* or lists *`Dgemm`* in a "where goblas
helps" table, it is naming the **BLAS operation Gonum runs underneath** your `m.Mul` call — not
a function you are meant to call. `Dgemm`, `Dgemv`, `Dsyrk` are the *names of the operations*;
`m.Mul`, `m.MulVec`, `SymRankK` are the *Gonum methods you write* that trigger them. Think of
the BLAS-routine names as labels for "what gets accelerated," not as your API.

(The raw `goblas` package is documented in the [main README](../../README.md#a-call-the-blas-routines-directly)
for the rarer case where you are working with flat slices outside of Gonum.)

## What BLAS accelerates — and what it does not

This is the honest framing that decides which algorithms are here:

- **BLAS speeds up dense linear algebra** — anything whose real cost is matrix multiplication,
  matrix–vector products, or the factorizations built from them. That covers a surprising
  amount of ML: regressions, kernel methods' Gram matrices, every neural-network layer.
- **BLAS does nothing for branchy, sequential, or comparison-driven algorithms.** A decision
  tree spends its time comparing feature values and splitting — there is no matrix multiply to
  speed up. Putting goblas underneath it changes nothing. That is why **decision trees are not
  in this list**: it would be dishonest to imply a benefit that does not exist.

So these tutorials cover only algorithms where goblas genuinely earns its keep.

## Two caveats that apply to all of them

1. **float64 or float32 — pick per workload.** goblas supports both. The classical-ML tutorials
   (regressions, SVM) stay in **float64** via `gonum/mat`, where numerical conditioning matters
   and gonum's full LAPACK is available. The neural-network and vision tutorials (MLP, CNN, KNN,
   YOLO) use **float32** via [`mat32`](goblas-mat32-fundamentals.md) — the precision real
   frameworks use, where single precision halves memory traffic and runs faster. Each tutorial
   says which it uses and why.
2. **No automatic differentiation, no GPU.** You compute gradients by hand (the tutorials show
   how) and everything runs on the CPU. Again: excellent for understanding, CPU-scale for size.

## The tutorials

Best read roughly in this order — later ones reuse ideas from earlier ones. All tutorials have corresponding datasets provided in the `data/` folder so you can follow along!

| Tutorial | What you build | Precision | Where goblas helps |
|----------|----------------|-----------|--------------------|
| [gonum-fundamentals.md](gonum-fundamentals.md) | Loading data, normalizing, creating matrices, and plotting | float64 | — |
| [goblas-mat32-fundamentals.md](goblas-mat32-fundamentals.md) | The float32 matrix type (`mat32`) and its solves | float32 | `Sgemm`/`Strsm`/`Ssyrk` |
| [linear-regression.md](linear-regression.md) | Fit a line/plane to data via least squares | float64 | `Dsyrk`/`Dgemm` + Cholesky solve |
| [logistic-regression.md](logistic-regression.md) | A binary classifier trained by gradient descent | float64 | `Dgemv`/`Dgemm` each step |
| [knn.md](knn.md) | k-nearest-neighbors with a fast batched distance matrix | **float32** | one big `Sgemm` |
| [svm.md](svm.md) | A kernel SVM's Gram matrix (the part that scales) | float64 | `Dsyrk`/`Dgemm` |
| [neural-net-mlp.md](neural-net-mlp.md) | A multilayer perceptron, forward and backward | **float32** | every layer is `Sgemm` |
| [neural-net-cnn.md](neural-net-cnn.md) | A convolutional layer via the im2col trick | **float32** | convolution becomes one `Sgemm` |
| [neural-net-lstm.md](neural-net-lstm.md) | An LSTM recurrent cell | float64 | the gates are a `Dgemm` per step |
| [yolo-object-detection.md](yolo-object-detection.md) | A YOLO object detector | **float32** | conv backbone = `Sgemm`; NMS is *not* BLAS |
| [reservoir-computing.md](reservoir-computing.md) | An Echo State Network | float64 | `Dgemv` per step + a ridge-regression readout |

If you are brand new, start with [gonum-fundamentals.md](gonum-fundamentals.md) to understand basic operations like loading and plotting data. Then move to [linear-regression.md](linear-regression.md): it introduces
`gonum/mat`, the column-major convention, and the solve-a-linear-system pattern that several
later tutorials build on.
