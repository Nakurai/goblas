# Neural networks I — the multilayer perceptron (MLP)

This is where matrix multiply stops being a trick and becomes the *whole point*. A neural
network is, computationally, a stack of matrix multiplies with simple nonlinear functions
between them. Every layer is a `Sgemm`. If you understood
[logistic-regression.md](logistic-regression.md), you already understand one neuron — a network
is just many of them, in layers.

**Real-world examples**: Fraud detection in finance, or complex pattern recognition where rules are not easily defined by humans.

> **Precision note.** This tutorial uses **float32** via goblas's
> [`mat32`](goblas-mat32-fundamentals.md) package — the single precision real training stacks
> use, where every matrix multiply moves half the bytes of float64 and runs faster (goblas's
> `Sgemm` hits ~560 GFLOPS vs ~360 for `Dgemm` on the M5 Pro). What stays educational is the
> rest: goblas is CPU-only with no automatic differentiation, so you compute gradients by hand.
> The math is exactly what a GPU framework does. (Prefer double precision? The identical code
> works with `gonum/mat` + `blasadapt.Use()`.)

## From one neuron to a layer

A single neuron computes a weighted sum of its inputs, adds a bias, and passes the result
through a nonlinear **activation** function:

```
output = activation(w · x + b)
```

A **layer** is many neurons looking at the same inputs. If the layer has `h` neurons and each
input has `d` features, stack the neurons' weight vectors as the columns of a weight matrix `W`
(`d × h`). Then for a whole **batch** of `n` input rows `X` (`n × d`), the entire layer's
pre-activations are one matrix multiply:

```
Z = X · W + b        →  Sgemm, an (n × h) matrix
A = activation(Z)    →  elementwise
```

That `X · W` is the beating heart of neural networks, and it is goblas's fastest routine. A deep
network just repeats this: the activations `A` of one layer are the inputs `X` of the next.

```
input → [W₁] → activation → [W₂] → activation → … → output
        Sgemm               Sgemm
```

## The forward pass

Let us build a tiny network: input → hidden layer (with ReLU activation) → output layer. ReLU is
the most common activation, and could not be simpler: `ReLU(z) = max(0, z)` — it keeps positives,
zeros out negatives.

Setup:

```go
import "github.com/nakurai/goblas/mat32"

// ReLU keeps positives, zeros negatives. float32 in, float32 out.
relu := func(_, _ int, v float32) float32 {
    if v > 0 {
        return v
    }
    return 0
}
```

Weights (initialized small and random in practice), and the forward pass — two `Sgemm`s with a
ReLU between:

```go
// Shapes: X is n×d, W1 is d×h, W2 is h×k (k outputs).
// You can use the non-linear classification dataset `moons.csv` in the `data/` folder for this!
W1 := mat32.NewDense32(d, h, w1Data)
W2 := mat32.NewDense32(h, k, w2Data)

// Hidden layer: H = ReLU(X · W1)
var Z1, H mat32.Dense32
Z1.Mul(X, W1)        // Sgemm on goblas  → n×h
H.Apply(relu, &Z1)   // elementwise ReLU

// Output layer: Y = H · W2
var Y mat32.Dense32
Y.Mul(&H, W2)        // Sgemm on goblas  → n×k
```

(We have omitted the bias terms for brevity; in practice you add a bias row, or append a 1s
column to `X` and a bias row to `W`, exactly the offset trick from
[linear regression](linear-regression.md).)

That is a complete forward pass — predictions for a whole batch, with the two expensive steps on
goblas. For classification you would pass `Y` through a **softmax** (a function that turns a list of raw scores into a list of probabilities that sum to 100%); for regression you would use `Y`
directly.

## The backward pass (training)

Training adjusts the weights to reduce error, by the same **gradient descent** idea as logistic
regression — now applied layer by layer, working *backwards* from the output. This is
**backpropagation**. The beautiful part for us: the gradients are *also* matrix multiplies.

Conceptually, if `dY` is how wrong the output is (prediction − target), then:

```
gradient of W2  =  Hᵀ · dY              → Sgemm
error reaching H =  dY · W2ᵀ            → Sgemm
gradient of W1  =  Xᵀ · (dH ⊙ ReLU′)    → Sgemm  (⊙ = elementwise)
```

Each gradient is a `Sgemm`; each "push the error to the previous layer" is a `Sgemm`. In code,
one training step looks like:

```go
// Forward (as above) gives Z1, H, Y. Suppose dY = Y − target (n×k).
var dY mat32.Dense32
dY.Sub(&Y, target)

// Gradient for W2:  Hᵀ · dY            (h×k)
var gW2 mat32.Dense32
gW2.Mul(H.T(), &dY)                      // Sgemm

// Error backpropagated to the hidden layer: dY · W2ᵀ   (n×h)
var dH mat32.Dense32
dH.Mul(&dY, W2.T())                       // Sgemm

// Apply ReLU's derivative: zero out where Z1 was negative.
dH.Apply(func(i, j int, v float32) float32 {
    if Z1.At(i, j) > 0 {
        return v
    }
    return 0
}, &dH)

// Gradient for W1:  Xᵀ · dH            (d×h)
var gW1 mat32.Dense32
gW1.Mul(X.T(), &dH)                       // Sgemm

// Gradient-descent step on both weight matrices (gradient scaled by the rate).
const lr float32 = 0.01
var s2, s1 mat32.Dense32
s2.Scale(lr, &gW2)
W2.Sub(W2, &s2)                           // W2 ← W2 − lr·gW2
s1.Scale(lr, &gW1)
W1.Sub(W1, &s1)                           // W1 ← W1 − lr·gW1
```

Wrap that in a loop over many batches and epochs and you are training a neural network. Every
`Mul` is a goblas `Sgemm`; the activations and their derivatives are cheap elementwise passes.

## Why neural nets are the ideal BLAS workload

Look at the tables: forward pass = 2 `Sgemm`s, backward pass = 4 `Sgemm`s, for a 2-layer net.
Scale to deeper nets and bigger batches and it is `Sgemm` almost all the way down — which is
precisely the routine goblas tuned hardest (the float32 8×8 NEON kernel, ~560 GFLOPS). The
nonlinearities are elementwise and negligible. This is why GPU vendors and BLAS authors obsess
over matrix multiply: **make `Sgemm` fast and you make deep learning fast.**

## Where goblas earned its keep

| Step | BLAS routine | goblas role |
|------|--------------|-------------|
| Each layer forward `X·W` | `Sgemm` | accelerated |
| Weight gradients `Hᵀ·dY`, `Xᵀ·dH` | `Sgemm` | accelerated |
| Backprop error `dY·Wᵀ` | `Sgemm` | accelerated |
| Activations & their derivatives | elementwise | plain Go (cheap) |

## Recap

- A neural network layer is `A = activation(X·W)` — a `Sgemm` plus an elementwise function.
- Forward and backward passes are *all* `Sgemm`s; training is gradient descent layer by layer.
- This is the workload BLAS exists to accelerate; goblas runs every matrix multiply.
- Precision: float32 via `mat32` (what real training uses); still CPU/no-autodiff, built for
  understanding rather than large-scale training.

Next: [neural-net-cnn.md](neural-net-cnn.md) shows how even *convolution* — which looks nothing
like a matrix multiply — is turned into one big `Sgemm`.
