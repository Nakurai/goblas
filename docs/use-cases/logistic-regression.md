# Logistic regression

Linear regression predicted a *number*. Logistic regression predicts a *probability* — "is this
email spam, yes or no?", "will this customer churn?" It is the simplest real classifier, and it
introduces **gradient descent**, the iterative training loop that every neural network in these
tutorials also uses. Read [linear-regression.md](linear-regression.md) first.

**Real-world examples**: Classifying emails as spam or not spam, or predicting whether a patient has a specific disease based on their medical history.

## From a number to a probability

We still compute a linear score `z = X·β` exactly as before. But a raw score can be any number
from −∞ to +∞, and we want a probability between 0 and 1. We squash it through the **sigmoid**
function:

```
σ(z) = 1 / (1 + e^(−z))
```

```
σ(z)
1 |        _____
  |      /
0.5|    /
  |  /
0 |_/______________ z
        0
```

Big positive score → probability near 1 ("yes"). Big negative → near 0 ("no"). Score of 0 →
0.5 (maximally unsure). So the model is: `p = σ(X·β)`, and we predict "yes" when `p > 0.5`.

## How we train it: gradient descent

There is no neat closed-form solution like the normal equations here, so we *search* for good
weights. The idea of **gradient descent**:

1. Start with some weights (say all zero).
2. Compute the model's predictions and how wrong they are.
3. Nudge each weight a little in the direction that reduces the error.
4. Repeat until it stops improving.

The "direction that reduces error" is the **gradient** (you can think of this like a multi-dimensional slope or a compass pointing downhill toward the lowest possible error). For logistic regression the math works
out beautifully simple — the gradient of the standard loss is:

```
gradient = Xᵀ (p − y)
```

where `p` is the vector of predicted probabilities and `y` is the vector of true 0/1 labels.
Read it as: take the prediction errors `(p − y)`, and `Xᵀ` spreads each error back onto the
features that caused it. Then we step the weights against the gradient:

```
β ← β − learningRate · gradient
```

The two matrix operations here — `X·β` (predictions) and `Xᵀ·(p−y)` (gradient) — are a
**`Dgemv`** each (or a **`Dgemm`** if you process a batch of problems at once). Both run on
goblas. The training loop is just those two products plus a cheap elementwise sigmoid, repeated.

## Building it with gonum/mat

Setup as always:

```go
import (
    "math"
    "github.com/nakurai/goblas/blasadapt"
    "gonum.org/v1/gonum/mat"
)

func init() { blasadapt.Use() }
```

Data: `X` is `n × (p+1)` (features plus a 1s column for the offset, just like before). You can use the synthetic `classification.csv` dataset in the `data/` folder to follow along. `y` is
the `n` labels (each 0 or 1), and `beta` starts at zeros:

```go
X := mat.NewDense(n, p+1, xData)
y := mat.NewVecDense(n, yData)
beta := mat.NewVecDense(p+1, nil) // all zeros to start
```

The sigmoid, applied elementwise with `Apply` (its function receives each entry's row, column,
and value):

```go
sigmoid := func(_, _ int, v float64) float64 { return 1 / (1 + math.Exp(-v)) }
```

Now the training loop. Each iteration is: predict, measure error, form the gradient, step.

```go
lr := 0.1
pred := mat.NewVecDense(n, nil) // reused prediction buffer
grad := mat.NewVecDense(p+1, nil)

for iter := 0; iter < 500; iter++ {
    // z = X · beta              → a Dgemv on goblas
    var z mat.VecDense
    z.MulVec(X, beta)

    // pred = sigmoid(z)         → cheap elementwise (in place)
    for i := 0; i < n; i++ {
        pred.SetVec(i, 1/(1+math.Exp(-z.AtVec(i))))
    }

    // err = pred − y            → elementwise subtract
    var errv mat.VecDense
    errv.SubVec(pred, y)

    // grad = Xᵀ · err           → another Dgemv on goblas
    grad.MulVec(X.T(), &errv)

    // beta ← beta − lr · grad
    grad.ScaleVec(lr, grad)
    beta.SubVec(beta, grad)
}
```

The *substance* is the two `MulVec` calls — your two goblas `Dgemv`s — with a cheap elementwise
sigmoid and subtraction between them. (We wrote the sigmoid as an explicit loop here for
clarity; `Dense.Apply` with the `sigmoid` function above does the same thing in one call when
you are working with a matrix rather than a vector.)

After the loop, `beta` holds the trained weights. To classify a new point `x` (with its 1s
entry): compute `σ(x·β)` and threshold at 0.5.

```go
score := mat.Dot(beta, xNew)  // x·β, a dot product
prob := 1 / (1 + math.Exp(-score))
label := 0
if prob > 0.5 {
    label = 1
}
```

## Batching: where Dgemm replaces Dgemv

Above, each step does matrix–vector products (`Dgemv`). If you train **many** logistic models
at once, or process the whole dataset as a batch where predictions form a matrix rather than a
vector, those become matrix–**matrix** products (`Dgemm`) — and `Dgemm` is where goblas is
fastest (the 8×6 NEON kernel, ~360 GFLOPS). The bigger and more batched the work, the more the
acceleration shows.

## Where goblas earned its keep

| Step (per iteration) | BLAS routine | goblas role |
|----------------------|--------------|-------------|
| Predictions `X·β` | `Dgemv` (or `Dgemm` batched) | accelerated |
| Gradient `Xᵀ·(p−y)` | `Dgemv` (or `Dgemm` batched) | accelerated |
| Sigmoid, subtract, scale | elementwise | plain Go (cheap) |

The cost is dominated by the two products, both on goblas. The elementwise parts are
negligible and not BLAS work.

## Recap

- Logistic regression = linear score → sigmoid → probability → threshold.
- Trained by gradient descent: repeatedly predict, form `Xᵀ(p−y)`, step the weights.
- The two products per step are `Dgemv`/`Dgemm` on goblas; batching turns them into the fast
  `Dgemm` path.
- This predict → error → gradient → step loop is the same skeleton the neural-network
  tutorials use, just with more layers.

Next: [knn.md](knn.md) — a different flavor, where a single clever `Dgemm` computes all pairwise
distances at once.
