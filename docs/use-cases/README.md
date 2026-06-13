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
| `github.com/nakurai/goblas/blasadapt` | the **adapter** — `blasadapt.Use()` plugs goblas in under Gonum | **always, in these tutorials** — then write pure `gonum/mat` code |
| `github.com/nakurai/goblas` | the **raw BLAS** — low-level functions like `goblas.Dgemm` on flat `[]float64` | only if you have raw slices and want a single BLAS call without Gonum types |

In every tutorial here you import **only `blasadapt`** (for the one-time `Use()` call) and
otherwise write `gonum/mat` code — `m.Mul(...)`, `m.Solve(...)`, `chol.Factorize(...)`. You do
**not** `import "github.com/nakurai/goblas"` or call `goblas.Dgemm` yourself.

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

1. **goblas is float64 only.** Great for classical ML and for learning. Production neural-net
   *training* usually uses float32 (or GPUs) for speed — so treat the neural-network tutorials
   as a way to *understand* the math by building it, not as a competitive training stack.
2. **No automatic differentiation, no GPU.** You compute gradients by hand (the tutorials show
   how) and everything runs on the CPU. Again: excellent for understanding, CPU-scale for size.

## The tutorials

Best read roughly in this order — later ones reuse ideas from earlier ones. All tutorials have corresponding datasets provided in the `data/` folder so you can follow along!

| Tutorial | What you build | Where goblas helps |
|----------|----------------|--------------------|
| [gonum-fundamentals.md](gonum-fundamentals.md) | Loading data, normalizing, creating matrices, and plotting | — |
| [linear-regression.md](linear-regression.md) | Fit a line/plane to data via least squares | `Dsyrk`/`Dgemm` + Cholesky solve |
| [logistic-regression.md](logistic-regression.md) | A binary classifier trained by gradient descent | `Dgemv`/`Dgemm` each step |
| [knn.md](knn.md) | k-nearest-neighbors with a fast batched distance matrix | one big `Dgemm` |
| [svm.md](svm.md) | A kernel SVM's Gram matrix (the part that scales) | `Dsyrk`/`Dgemm` |
| [neural-net-mlp.md](neural-net-mlp.md) | A multilayer perceptron, forward and backward | every layer is `Dgemm` |
| [neural-net-cnn.md](neural-net-cnn.md) | A convolutional layer via the im2col trick | convolution becomes one `Dgemm` |
| [neural-net-lstm.md](neural-net-lstm.md) | An LSTM recurrent cell | the gates are a `Dgemm` per step |
| [yolo-object-detection.md](yolo-object-detection.md) | A YOLO object detector | conv backbone = `Dgemm`; NMS is *not* BLAS |
| [reservoir-computing.md](reservoir-computing.md) | An Echo State Network | `Dgemv` per step + a ridge-regression readout |

If you are brand new, start with [gonum-fundamentals.md](gonum-fundamentals.md) to understand basic operations like loading and plotting data. Then move to [linear-regression.md](linear-regression.md): it introduces
`gonum/mat`, the column-major convention, and the solve-a-linear-system pattern that several
later tutorials build on.
