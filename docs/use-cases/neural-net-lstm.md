# Neural networks III — recurrent networks and the LSTM

Some data is a *sequence*: words in a sentence, samples in an audio clip, days of a stock price.
**Recurrent** neural networks process sequences one step at a time, carrying a memory forward.
The **LSTM** (Long Short-Term Memory) is the classic, robust recurrent cell. Its internals look
intimidating, but computationally each time step is — once again — a single matrix multiply.
Read [neural-net-mlp.md](neural-net-mlp.md) first.

**Real-world examples**: Language translation, speech recognition, or stock market trend prediction.

> **Precision note.** Real LSTMs run in **float32**, and the per-step gate `Dgemm` has a float32
> twin (`Sgemm` via [`mat32`](goblas-mat32-fundamentals.md)). This tutorial stays in `gonum/mat`
> float64 only because it leans on convenience methods (`Slice`, `Copy` of sub-blocks) that
> `mat32` keeps lean and does not expose; the matrix-multiply lesson is identical in either
> precision.

## The idea: a network with memory

A plain feed-forward network treats every input independently. A recurrent network keeps a
**hidden state** `h` that it updates at each step and feeds back into itself:

```
x₁    x₂    x₃        (sequence of inputs)
 │     │     │
 ▼     ▼     ▼
[cell]→[cell]→[cell]→ …   (same cell, reused each step)
 │     │     │
 ▼     ▼     ▼
 h₁    h₂    h₃        (hidden state, carried forward)
```

The same cell, with the same weights, runs at every step; what changes is the input `xₜ` and the
incoming state `hₜ₋₁`. That is how the network "remembers" earlier parts of the sequence.

## Why LSTM: controlling the memory

A naive recurrent cell forgets quickly and is hard to train over long sequences. The LSTM adds a
separate **cell state** `c` (a memory conveyor belt) and three **gates** — small neural networks
that decide, at each step, what to do with the memory:

- **forget gate** `f` — how much of the old memory to keep,
- **input gate** `i` — how much new information to write,
- **output gate** `o` — how much of the memory to expose as the hidden state.

Each gate looks at the same two things — the current input `xₜ` and the previous hidden state
`hₜ₋₁` — and produces a number between 0 and 1 (via a **sigmoid** function, which squashes any value into a probability between 0 and 1) per memory slot. There is also a
**candidate** `g` (via **tanh**, which squashes any value to a range between -1 and 1) proposing new memory content.

## The key realization: four gates, one matrix multiply

Each gate computes `activation(Wₓ·xₜ + Wₕ·hₜ₋₁ + b)` — that is just a neuron layer, like the
MLP. There are four such computations (forget, input, output, candidate). The trick that makes
LSTMs efficient: **stack all four weight matrices together and concatenate the inputs**, so all
four gates are produced by **one** matrix multiply per step.

Concretely, stack `xₜ` and `hₜ₋₁` into one vector `[xₜ; hₜ₋₁]`, and stack the four gates' weight
matrices into one big `W`. Then:

```
gates = W · [xₜ ; hₜ₋₁] + b      →  one Dgemm (per step, or per batch of sequences)
```

and you slice the result into the four gate pre-activations. For a **batch** of sequences
processed together (the normal case), `[X ; H]` is a matrix and this is a full `Dgemm` — goblas's
fast path.

Then the cheap elementwise part combines them into the new memory and hidden state:

```
f = σ(gates_f)      i = σ(gates_i)      o = σ(gates_o)      g = tanh(gates_g)
cₜ = f ⊙ cₜ₋₁ + i ⊙ g          (update the memory: forget some, add some)
hₜ = o ⊙ tanh(cₜ)              (expose part of the memory as output)
```

(`σ` = sigmoid, `⊙` = elementwise multiply.)

## Building one LSTM step with gonum/mat

Setup:

```go
import (
    "math"
    "github.com/nakurai/goblas/blasadapt"
    "gonum.org/v1/gonum/mat"
)

func init() { blasadapt.Use() }

sigmoid := func(_, _ int, v float64) float64 { return 1 / (1 + math.Exp(-v)) }
tanh := func(_, _ int, v float64) float64 { return math.Tanh(v) }
```

Process a batch of `n` sequences with input size `d` and hidden size `H` (you can try this with the `time_series.csv` dataset in the `data/` folder). Stack input and
previous hidden state side by side into a `n × (d+H)` matrix, and let `W` be `(d+H) × 4H` (the
four gates stacked along the columns):

```go
// xt: n×d (this step's inputs)   hPrev: n×H   cPrev: n×H
// Build concat = [xt | hPrev], an n×(d+H) matrix.
concat := mat.NewDense(n, d+H, nil)
concat.Slice(0, n, 0, d).(*mat.Dense).Copy(xt)
concat.Slice(0, n, d, d+H).(*mat.Dense).Copy(hPrev)

// One Dgemm produces all four gates at once: n×(4H)
var gates mat.Dense
gates.Mul(concat, W)       // ← the goblas Dgemm, the cost of the step
// (add bias here in practice)

// Slice out the four gate blocks (each n×H) and activate them.
fGate := apply(sigmoid, gates.Slice(0, n, 0, H))
iGate := apply(sigmoid, gates.Slice(0, n, H, 2*H))
oGate := apply(sigmoid, gates.Slice(0, n, 2*H, 3*H))
gCand := apply(tanh, gates.Slice(0, n, 3*H, 4*H))

// Update memory and hidden state — all cheap elementwise:
// cNext = fGate ⊙ cPrev + iGate ⊙ gCand
cNext := mat.NewDense(n, H, nil)
cNext.MulElem(fGate, cPrev)
tmp := mat.NewDense(n, H, nil)
tmp.MulElem(iGate, gCand)
cNext.Add(cNext, tmp)

// hNext = oGate ⊙ tanh(cNext)
hNext := mat.NewDense(n, H, nil)
hNext.Apply(tanh, cNext)
hNext.MulElem(oGate, hNext)
```

(`apply` here is a tiny helper that runs `Apply` on a slice and returns the result — Gonum
bookkeeping; the substance is the single `gates.Mul(concat, W)`.) You then loop this over the
time steps of your sequence, feeding `hNext`/`cNext` back in as `hPrev`/`cPrev`.

## The honest part: sequential over time

There is one structural cost worth naming. Within a step, the work is a fat `Dgemm` — great. But
the steps are **inherently sequential**: step `t` needs `hₜ₋₁` from step `t−1`, so you cannot run
the time steps in parallel. goblas parallelizes *within* each step's matrix multiply (across the
batch and the matrix dimensions), but it cannot collapse the time loop. That is a property of
recurrence, not of goblas — the same constraint exists on a GPU, and it is why Transformers
(which process a whole sequence in parallel) have largely displaced LSTMs for very long
sequences. For learning the mechanics, the per-step `Dgemm` is exactly the pattern to internalize.

## Where goblas earned its keep

| Step (per time step) | BLAS routine | goblas role |
|----------------------|--------------|-------------|
| All four gates `[X;H]·W` | `Dgemm` | accelerated — the cost of the step |
| Sigmoids, tanh, gate combine | elementwise | plain Go (cheap) |
| The loop over time | — | sequential by nature, not parallelizable |

## Recap

- Recurrent networks carry a hidden state across a sequence; the LSTM adds gated memory.
- The four gates are computed by **one `Dgemm`** per step on `[xₜ; hₜ₋₁]·W`, then combined with
  cheap elementwise sigmoid/tanh.
- goblas accelerates that per-step `Dgemm`; the time loop itself is inherently sequential.

Next: [reservoir-computing.md](reservoir-computing.md) — a recurrent network you barely train,
where the per-step update is a `Dgemv` and the *training* collapses to the linear regression you
already know.
