# Neural networks I — the multilayer perceptron (MLP)

This is where matrix multiply stops being a trick and becomes the *whole point*. A neural
network is, computationally, a stack of matrix multiplies with simple nonlinear functions
between them. Every layer is a `Dgemm`. If you understood
[logistic-regression.md](logistic-regression.md), you already understand one neuron — a network
is just many of them, in layers.

**Real-world examples**: Fraud detection in finance, or complex pattern recognition where rules are not easily defined by humans.

> **Caveat up front.** goblas is float64 and CPU-only with no automatic differentiation. This
> tutorial teaches you the mechanics by building them yourself, in double precision, on the CPU.
> That is ideal for *understanding*; it is not a competitive training stack for large models
> (those use float32 on GPUs). The math, though, is exactly the same.

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
Z = X · W + b        →  Dgemm, an (n × h) matrix
A = activation(Z)    →  elementwise
```

That `X · W` is the beating heart of neural networks, and it is goblas's fastest routine. A deep
network just repeats this: the activations `A` of one layer are the inputs `X` of the next.

```
input → [W₁] → activation → [W₂] → activation → … → output
        Dgemm               Dgemm
```

## The forward pass

Let us build a tiny network: input → hidden layer (with ReLU activation) → output layer. ReLU is
the most common activation, and could not be simpler: `ReLU(z) = max(0, z)` — it keeps positives,
zeros out negatives.

Setup:

```go
import (
    "math"
    "github.com/nakurai/goblas/blasadapt"
    "gonum.org/v1/gonum/mat"
)

func init() { blasadapt.Use() }

relu := func(_, _ int, v float64) float64 { return math.Max(0, v) }
```

Weights (initialized small and random in practice), and the forward pass — two `Dgemm`s with a
ReLU between:

```go
// Shapes: X is n×d, W1 is d×h, W2 is h×k (k outputs).
// You can use the non-linear classification dataset `moons.csv` in the `data/` folder for this!
W1 := mat.NewDense(d, h, w1Data)
W2 := mat.NewDense(h, k, w2Data)

// Hidden layer: H = ReLU(X · W1)
var Z1, H mat.Dense
Z1.Mul(X, W1)        // Dgemm on goblas  → n×h
H.Apply(relu, &Z1)   // elementwise ReLU

// Output layer: Y = H · W2
var Y mat.Dense
Y.Mul(&H, W2)        // Dgemm on goblas  → n×k
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
gradient of W2  =  Hᵀ · dY              → Dgemm
error reaching H =  dY · W2ᵀ            → Dgemm
gradient of W1  =  Xᵀ · (dH ⊙ ReLU′)    → Dgemm  (⊙ = elementwise)
```

Each gradient is a `Dgemm`; each "push the error to the previous layer" is a `Dgemm`. In code,
one training step looks like:

```go
// Forward (as above) gives Z1, H, Y. Suppose dY = Y − target (n×k).
var dY mat.Dense
dY.Sub(&Y, target)

// Gradient for W2:  Hᵀ · dY            (h×k)
var gW2 mat.Dense
gW2.Mul(H.T(), &dY)                      // Dgemm

// Error backpropagated to the hidden layer: dY · W2ᵀ   (n×h)
var dH mat.Dense
dH.Mul(&dY, W2.T())                       // Dgemm

// Apply ReLU's derivative: zero out where Z1 was negative.
dH.Apply(func(i, j int, v float64) float64 {
    if Z1.At(i, j) > 0 {
        return v
    }
    return 0
}, &dH)

// Gradient for W1:  Xᵀ · dH            (d×h)
var gW1 mat.Dense
gW1.Mul(X.T(), &dH)                       // Dgemm

// Gradient-descent step on both weight matrices.
lr := 0.01
W2.Sub(W2, scale(lr, &gW2))               // W2 ← W2 − lr·gW2
W1.Sub(W1, scale(lr, &gW1))               // W1 ← W1 − lr·gW1
```

Wrap that in a loop over many batches and epochs and you are training a neural network. Every
`Mul` is a goblas `Dgemm`; the activations and their derivatives are cheap elementwise passes.

## Why neural nets are the ideal BLAS workload

Look at the tables: forward pass = 2 `Dgemm`s, backward pass = 4 `Dgemm`s, for a 2-layer net.
Scale to deeper nets and bigger batches and it is `Dgemm` almost all the way down — which is
precisely the routine goblas tuned hardest (the 8×6 NEON kernel, ~360 GFLOPS). The
nonlinearities are elementwise and negligible. This is why GPU vendors and BLAS authors obsess
over matrix multiply: **make `Dgemm` fast and you make deep learning fast.**

## Where goblas earned its keep

| Step | BLAS routine | goblas role |
|------|--------------|-------------|
| Each layer forward `X·W` | `Dgemm` | accelerated |
| Weight gradients `Hᵀ·dY`, `Xᵀ·dH` | `Dgemm` | accelerated |
| Backprop error `dY·Wᵀ` | `Dgemm` | accelerated |
| Activations & their derivatives | elementwise | plain Go (cheap) |

## Recap

- A neural network layer is `A = activation(X·W)` — a `Dgemm` plus an elementwise function.
- Forward and backward passes are *all* `Dgemm`s; training is gradient descent layer by layer.
- This is the workload BLAS exists to accelerate; goblas runs every matrix multiply.
- Caveat: float64/CPU/no-autodiff — built for understanding, not large-scale training.

Next: [neural-net-cnn.md](neural-net-cnn.md) shows how even *convolution* — which looks nothing
like a matrix multiply — is turned into one big `Dgemm`.
