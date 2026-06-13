# Reservoir computing (Echo State Networks)

This is a beautiful finale: a recurrent neural network for sequences that you **barely train**.
Reservoir computing keeps the expensive, fiddly recurrent part *fixed and random*, and trains
only a simple linear readout on top. The payoff for us: the per-step update is a goblas
matrix–vector product, and the *entire training step* is the linear regression you already
learned. Read [neural-net-lstm.md](neural-net-lstm.md) and
[linear-regression.md](linear-regression.md) first — this tutorial ties them together.

**Real-world examples**: Real-time signal processing in telecommunications, or forecasting chaotic systems like weather patterns.

## The surprising idea

Training a recurrent network (like an LSTM) is slow and delicate because you must adjust all the
recurrent weights through time. Reservoir computing asks: *what if we don't?*

- Build a big recurrent layer — the **reservoir** — with **random, fixed** weights. Never train
  them.
- Feed your input sequence through it. The reservoir's job is to be a rich, nonlinear *echo
  chamber*: its state at each moment is a complicated mixture of recent inputs. It transforms
  your sequence into a high-dimensional set of features "for free."
- Train **only** a linear readout that maps the reservoir's state to your desired output.

```
input  →  [ random fixed reservoir ]  →  states  →  [ trained linear readout ]  →  output
sequence    (never trained, just echoes)             (the only thing you train)
```

It works astonishingly well for time-series tasks, and because the only trained part is linear,
training is fast, convex, and has a closed-form solution — no gradient descent, no
backpropagation through time.

## Part 1 — running the reservoir (a Dgemv per step)

The reservoir holds a state vector `x` of size `N` (the number of reservoir neurons, often
hundreds). At each time step it updates using the new input `u` and its own previous state:

```
x(t) = tanh( W_in · u(t)  +  W · x(t−1) )
```

- `W_in` (`N × d`) maps the input into the reservoir — fixed random.
- `W` (`N × N`) is the recurrent reservoir matrix — fixed random (scaled carefully so the echoes
  neither die out nor blow up — the "echo state property").
- `tanh` squashes elementwise.

Each step is two matrix–vector products (`W_in·u` and `W·x`) — **`Dgemv`** on goblas — added and
passed through tanh. The big one is `W·x`: with `N` in the hundreds, that `N×N` times `N` product
is the per-step cost, and goblas runs it.

```go
import (
    "math"
    "github.com/nakurai/goblas/blasadapt"
    "gonum.org/v1/gonum/mat"
)

func init() { blasadapt.Use() }

// Win: N×d (fixed random)   W: N×N (fixed random, spectral-radius scaled)
// (You can use the `time_series.csv` dataset in `data/` for the `inputAt` step)
x := mat.NewVecDense(N, nil) // reservoir state, starts at zero
states := mat.NewDense(T, N, nil) // we will collect one state row per time step

for t := 0; t < T; t++ {
    u := inputAt(t) // a d-vector for this time step

    var a, b mat.VecDense
    a.MulVec(Win, u)  // W_in · u   → Dgemv on goblas
    b.MulVec(W, x)    // W · x      → Dgemv on goblas

    a.AddVec(&a, &b)
    for i := 0; i < N; i++ {
        x.SetVec(i, math.Tanh(a.AtVec(i))) // elementwise tanh
    }
    states.SetRow(t, x.RawVector().Data) // remember this step's state
}
```

After this loop, `states` (`T × N`) holds the reservoir's response to the whole input
sequence — one rich feature vector per time step. We never adjusted `W` or `W_in`.

## Part 2 — training the readout (this *is* linear regression)

Now we want output weights `W_out` so that, at each step, `W_out · x(t)` predicts the target
`y(t)`. Stack the collected states as the matrix `States` (`T × N`) and the targets as `Y`
(`T × outDim`). Finding `W_out` to minimize squared error is **exactly the least-squares problem
from [linear regression](linear-regression.md)** — with a small regularization term `λ` added
for stability (this is **ridge regression**, which means adding a term to prevent weights from exploding by penalizing excessively large values, ensuring a stable model):

```
W_out = (Statesᵀ·States + λI)⁻¹ · Statesᵀ·Y
```

Recognize every piece:

- `Statesᵀ·States` is a symmetric `N×N` matrix — a **`Dsyrk`** (or `Dgemm`).
- adding `λI` nudges the diagonal (cheap),
- `Statesᵀ·Y` is a **`Dgemm`**,
- solving the system uses a **Cholesky** factorization (`Dtrsm`/`Dgemm` inside).

So the entire training step is goblas-accelerated dense linear algebra, and it is *the same code
pattern* as the explicit normal-equations path in the linear-regression tutorial:

```go
// StatesᵀStates + λI   (N×N, symmetric)
var StS mat.Dense
StS.Mul(states.T(), states)         // Dgemm/Dsyrk on goblas
for i := 0; i < N; i++ {
    StS.Set(i, i, StS.At(i, i)+lambda) // ridge term
}

// StatesᵀY              (N×outDim)
var StY mat.Dense
StY.Mul(states.T(), Y)              // Dgemm on goblas

// Solve (StS) · Wout = StY  for Wout.
var Wout mat.Dense
if err := Wout.Solve(&StS, &StY); err != nil {  // Cholesky/LU on goblas
    log.Fatal(err)
}
```

That single `Solve` *is* the whole training. No epochs, no learning rate, no backprop — a closed
form, computed once, on goblas.

## Predicting

Run a new input sequence through the (same, fixed) reservoir to get its states, then apply the
trained readout — one matrix multiply:

```go
var preds mat.Dense
preds.Mul(newStates, &Wout) // newStates (T'×N) · Wout (N×outDim)  → Dgemm
```

## Why this is such a clean goblas fit

Both halves are BLAS-bound, and the *training* half has no non-BLAS bottleneck at all (unlike
the SMO solver in [SVM](svm.md) or the elementwise-heavy steps elsewhere): it is a `Dsyrk`, a
`Dgemm`, and a Cholesky solve — goblas's strengths end to end. The only inherently sequential
part is running the reservoir forward in time (the same recurrence constraint as the LSTM), and
even there each step is a goblas `Dgemv`.

## Where goblas earned its keep

| Step | BLAS routine | goblas role |
|------|--------------|-------------|
| Reservoir update `W·x`, `W_in·u` (per step) | `Dgemv` | accelerated |
| Readout `Statesᵀ·States` | `Dsyrk`/`Dgemm` | accelerated |
| Readout `Statesᵀ·Y` | `Dgemm` | accelerated |
| Solve for `W_out` | Cholesky (`Dtrsm`/`Dgemm`) | accelerated |
| Predict `States·W_out` | `Dgemm` | accelerated |
| tanh per step | elementwise | plain Go (cheap) |
| The time loop | — | sequential by nature |

## Recap

- Reservoir computing leaves the recurrent layer random and fixed, training only a linear
  readout — so a hard recurrent-training problem becomes easy.
- Running the reservoir is a `Dgemv` per step; **training the readout is ridge regression** —
  `Dsyrk` + `Dgemm` + Cholesky solve, exactly the [linear-regression](linear-regression.md)
  machinery.
- It is one of the cleanest end-to-end goblas workloads: both running and training are dense
  linear algebra.

That completes the use-case tutorials. Back to the [index](README.md), or out to
[overview.md](../overview.md) for the project's rationale and limits.
