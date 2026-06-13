# k-nearest neighbors (KNN)

KNN is the most intuitive classifier there is, and it hides a lovely trick: computing *all* the
distances you need turns out to be a single matrix multiply. That is what makes it a great fit
for goblas.

## The idea

To classify a new point, look at the `k` training points closest to it and take a vote. "Tell
me who your neighbors are and I'll tell you who you are." There is no training phase at all —
you just store the data and, at prediction time, find nearest neighbors.

```
   class A         ?  ← new point: its 3 nearest
   o   o          /     neighbors are 2×B, 1×A
     o      x   x       → predict B
            x  ·
        x   x
       class B
```

The entire computational cost is **measuring distances** from each query point to each training
point. If you have `m` query points and `n` training points, that is `m × n` distances — and
naively, a double loop. The trick is to do them all at once with linear algebra.

## The distance-matrix trick

The squared Euclidean distance between two vectors `x` and `t` expands like this:

```
‖x − t‖²  =  ‖x‖²  +  ‖t‖²  −  2 · (x · t)
```

The first two terms are just the squared lengths of each point (cheap — one pass over the data).
The interesting term is `x · t`, the dot product between a query and a training point. Stack all
queries into a matrix `Q` (`m × d`, one point per row) and all training points into `T` (`n ×
d`). Then **every** query-training dot product at once is the matrix multiply:

```
Q · Tᵀ        →  an m × n matrix of all dot products  →  one Dgemm
```

So the whole distance matrix is: that one `Dgemm`, plus adding the per-row and per-column
squared norms. The expensive part — the `m·n·d` arithmetic — is the `Dgemm`, fully on goblas.
This is exactly how production libraries (scikit-learn, FAISS) compute batched distances.

## Building it with gonum/mat

Setup:

```go
import (
    "github.com/nakurai/goblas/blasadapt"
    "gonum.org/v1/gonum/mat"
)

func init() { blasadapt.Use() }
```

Queries and training data, one point per row:

```go
Q := mat.NewDense(m, d, queryData)    // m points to classify
T := mat.NewDense(n, d, trainData)    // n labeled training points
```

Step 1 — the cross term, one big goblas `Dgemm`:

```go
var cross mat.Dense
cross.Mul(Q, T.T()) // cross[i][j] = Qᵢ · Tⱼ   (m × n)
```

Step 2 — the squared norms of every query and every training point:

```go
qNorm := make([]float64, m)
for i := 0; i < m; i++ {
    row := Q.RawRowView(i)
    qNorm[i] = floats.Dot(row, row) // ‖Qᵢ‖²
}
tNorm := make([]float64, n)
for j := 0; j < n; j++ {
    row := T.RawRowView(j)
    tNorm[j] = floats.Dot(row, row) // ‖Tⱼ‖²
}
```

(`floats.Dot` is from `gonum.org/v1/gonum/floats`; each of those dot products also routes
through goblas's `Ddot`.)

Step 3 — assemble the squared distances. We do not even need to take square roots: nearest by
squared distance is the same ordering as nearest by distance.

```go
dist2 := mat.NewDense(m, n, nil)
dist2.Apply(func(i, j int, v float64) float64 {
    return qNorm[i] + tNorm[j] - 2*v // ‖Qᵢ−Tⱼ‖²
}, &cross)
```

Step 4 — for each query row, find the `k` smallest entries and vote among their labels. This
last step is *not* matrix algebra — it is a partial sort per row — and that is fine: it is cheap
compared to the `Dgemm`, and it is honest to point out that goblas does not accelerate it (there
is no matrix multiply in a sort).

```go
for i := 0; i < m; i++ {
    row := dist2.RawRowView(i)          // n squared distances for query i
    idx := kSmallestIndices(row, k)     // your partial-sort helper
    label := majorityVote(idx, trainLabels)
    // …record label for query i…
}
```

## Why this is the right shape for goblas

The naive KNN is a triple loop (`m × n × d`) written in Go — slow, and exactly the kind of
scalar code the Go compiler cannot vectorize. By re-expressing the `m·n·d` work as `Q · Tᵀ`, you
hand it to goblas's `Dgemm`, which *is* vectorized and multithreaded. For a large training set
this is the difference between seconds and milliseconds. The per-row norm computations and the
final top-k selection are linear-ish and cheap; the dominant cost is the one matrix multiply.

## Where goblas earned its keep

| Step | BLAS routine | goblas role |
|------|--------------|-------------|
| All query·training dot products | `Dgemm` (`Q · Tᵀ`) | accelerated — the whole cost |
| Per-point squared norms | `Ddot` | accelerated (cheap) |
| Combine into distances | elementwise `Apply` | plain Go (cheap) |
| Top-k + vote | partial sort | plain Go — *not* BLAS work |

## Recap

- KNN classifies by majority vote of the `k` closest training points; the cost is all the
  pairwise distances.
- The identity `‖x−t‖² = ‖x‖² + ‖t‖² − 2x·t` turns the whole distance matrix into **one
  `Dgemm`** (`Q·Tᵀ`) plus cheap norms — that is the goblas win.
- Skip the square roots (ordering is preserved) and the final top-k is a plain sort, not BLAS.

Next: [svm.md](svm.md) reuses this "fill a matrix with all pairwise comparisons" idea, but the
comparison is a *kernel* and the matrix is the heart of a support vector machine.
