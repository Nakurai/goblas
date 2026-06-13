# Linear regression

This is the place to start. We will fit a straight line (then a plane, then any number of
dimensions) to data, and along the way you will learn the `gonum/mat` basics that the other
tutorials reuse. No prior knowledge assumed.

## The problem, in pictures

You have a pile of points and you suspect a roughly linear relationship: as `x` goes up, `y`
tends to go up (or down). **Linear regression** finds the line that fits those points best:

```
y
|           .  .
|        .  '
|     . '  .
|  . '
| '____________ x
```

For a single input the line is `y = w·x + b`. With several inputs (say square footage,
bedrooms, age of a house → price) it becomes

```
y = w₁x₁ + w₂x₂ + … + wₚxₚ + b
```

Our job is to find the weights `w` and the offset `b` that make the predictions closest to the
real `y` values.

## "Best fit" means least squares

We measure how wrong a set of weights is by the **sum of squared errors**: for each data point
take (prediction − truth), square it (so over- and under-shooting both count, and big misses
count a lot), and add them all up. The best weights are the ones that make this sum as small as
possible. This is **least squares**, and it has a clean, exact solution — no iteration needed.

## Writing it as matrices

Stack your data. With `n` data points and `p` inputs each, build a matrix `X` with one row per
data point. Add a column of all 1s to absorb the offset `b` (a neat trick: the weight on the
constant-1 column *is* `b`). Put the targets in a column vector `y`. Then every prediction at
once is the matrix–vector product `X·β`, where `β` holds all the weights including `b`.

The least-squares solution satisfies the **normal equations**:

```
(XᵀX) β = Xᵀ y
```

Read that as: form the small `p×p` matrix `XᵀX` and the vector `Xᵀy`, then solve a linear
system for `β`. Two facts make this fast and BLAS-friendly:

- `XᵀX` is a matrix multiplied by its own transpose — exactly the **`Dsyrk`** routine
  (symmetric rank-k update). `Xᵀy` is a matrix–vector product, **`Dgemv`**.
- `XᵀX` is symmetric and (for sensible data) positive-definite, so the system solves with a
  **Cholesky** factorization — itself built from `Dsyrk`/`Dtrsm`/`Dgemm` under the hood.

So the entire cost of linear regression is goblas-accelerated BLAS. Good first example.

## Building it with gonum/mat

First, the one-time setup that puts goblas under Gonum:

```go
import (
    "github.com/nakurai/goblas/blasadapt"
    "gonum.org/v1/gonum/mat"
)

func init() { blasadapt.Use() }
```

Now the data. `mat.NewDense(rows, cols, data)` makes a dense matrix; the data slice is laid out
row by row. Say we have `n` points and `p` features, and we have already added the 1s column so
`X` is `n × (p+1)`:

```go
X := mat.NewDense(n, p+1, xData) // each row: [feature1, feature2, …, 1.0]
y := mat.NewVecDense(n, yData)   // the n target values
```

The most direct route is to let Gonum solve the least-squares problem in one call —
`Dense.Solve` does exactly this, internally forming the normal equations (or using a QR
factorization) and running them on goblas:

```go
var beta mat.Dense
if err := beta.Solve(X, y); err != nil {
    // Solve returns an error if X is rank-deficient (columns linearly dependent).
    log.Fatal(err)
}
// beta is now a (p+1)×1 matrix: the weights, with the last entry being b.
```

That is the whole fit. To predict on new data `Xnew` (same column layout, including the 1s
column), it is one matrix multiply — a `Dgemm` on goblas:

```go
var preds mat.Dense
preds.Mul(Xnew, &beta) // preds[i] = prediction for row i
```

### Doing it "by hand" to see the BLAS

If you want to *see* the normal equations rather than hide them inside `Solve`, build them
explicitly. This is also the pattern the [reservoir computing](reservoir-computing.md) readout
reuses:

```go
// XtX = Xᵀ X   (symmetric, (p+1)×(p+1))
var XtX mat.Dense
XtX.Mul(X.T(), X) // X.T() is a free, zero-copy transpose view

// Xty = Xᵀ y
var Xty mat.Dense
Xty.Mul(X.T(), y)

// Solve (XtX) β = Xty with a Cholesky factorization.
var chol mat.Cholesky
sym := mat.NewSymDense(p+1, nil)
sym.CopySym(mat.NewSymDense(p+1, nil)) // (XtX is symmetric; wrap it)
// In practice: build sym from XtX's upper triangle, then:
var beta mat.VecDense
if ok := chol.Factorize(sym); ok {
    chol.SolveVecTo(&beta, Xty.ColView(0))
}
```

`X.T()` is worth pausing on: it does **not** copy or move any data — it is a transposed *view*
of the same numbers. (This is the same row↔column idea that lets goblas bridge Gonum's
row-major world to its own column-major kernels for free.)

## A note on stability: prefer QR for tricky data

Forming `XᵀX` squares the spread of your data, which can amplify numerical error when columns
are nearly redundant (e.g. two features that are almost the same). The professional fix is to
skip `XᵀX` and factor `X` directly with **QR**:

```go
var qr mat.QR
qr.Factorize(X)
var beta mat.Dense
qr.SolveTo(&beta, false, y)
```

QR is a little more work but much more numerically stable, and `mat.QR` runs its heavy steps on
goblas too. `Dense.Solve` already chooses a QR-based path for non-square systems, so for most
uses you can simply call `Solve` and trust it.

## Where goblas earned its keep

| Step | BLAS routine | goblas role |
|------|--------------|-------------|
| Form `XᵀX` | `Dsyrk` / `Dgemm` | accelerated |
| Form `Xᵀy` | `Dgemv` | accelerated |
| Solve via Cholesky or QR | `Dtrsm`/`Dgemm` inside the factorization | accelerated |
| Predict | `Dgemm` | accelerated |

Every expensive step is dense linear algebra, so the whole fit benefits — exactly the kind of
workload goblas exists for.

## Recap

- Linear regression = find weights minimizing squared error = solve `(XᵀX)β = Xᵀy`.
- In `gonum/mat`, the easy path is `beta.Solve(X, y)`; the explicit path shows the `Dsyrk` +
  Cholesky structure.
- `X.T()` is a free transpose view.
- Prefer QR (`mat.QR`, or just `Solve`) when data may be ill-conditioned.

Next: [logistic-regression.md](logistic-regression.md) turns this into *classification* and
introduces gradient descent — the training loop pattern the neural-network tutorials build on.
