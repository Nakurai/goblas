# goblas documentation

Guides and tutorials for goblas — a pure-Go BLAS with hand-tuned assembly kernels. For
installation and the benchmark tables, see the [project README](../README.md); for the
auto-generated API reference, see
[pkg.go.dev/github.com/nakurai/goblas](https://pkg.go.dev/github.com/nakurai/goblas).

## Start here

- **[overview.md](overview.md)** — what goblas is, why it exists, and where its limits are
  (carefully separating *current/fixable* limitations from ones *inherent to the idea*). Read
  this first if you want to understand the project.

## Guides

- **[adding-new-cpu.md](adding-new-cpu.md)** — make goblas faster on a processor it has not
  been tuned for. Written for someone who knows nothing about CPU architecture; a two-tier
  path (tuning only, no assembly → a full SIMD kernel).
- **[verify-benchmark.md](verify-benchmark.md)** — how to test correctness and measure
  performance: every test suite and benchmark, with exact commands, plus the gotchas
  (race detection, cross-compilation, thermal drift).

## Use-case tutorials

Teaching tutorials that build real machine-learning algorithms on goblas + `gonum/mat`,
assuming no ML or Gonum background, and showing exactly where the BLAS acceleration comes
from. Start at the **[use-cases index](use-cases/README.md)**, or jump in:

- [linear-regression.md](use-cases/linear-regression.md) — least squares; the gonum/mat basics
- [logistic-regression.md](use-cases/logistic-regression.md) — classification + gradient descent
- [knn.md](use-cases/knn.md) — the all-distances-in-one-`Dgemm` trick
- [svm.md](use-cases/svm.md) — the kernel/Gram matrix (and an honest note on the solver)
- [neural-net-mlp.md](use-cases/neural-net-mlp.md) — every layer is a `Dgemm`
- [neural-net-cnn.md](use-cases/neural-net-cnn.md) — convolution via im2col → `Dgemm`
- [neural-net-lstm.md](use-cases/neural-net-lstm.md) — the gates are one `Dgemm` per step
- [yolo-object-detection.md](use-cases/yolo-object-detection.md) — a real model: conv backbone is `Dgemm`, NMS is not BLAS
- [reservoir-computing.md](use-cases/reservoir-computing.md) — `Dgemv` per step + ridge-regression readout

(Decision trees are intentionally omitted — they are branchy, not matrix work, so goblas would
not accelerate them. See the [use-cases index](use-cases/README.md) for that reasoning.)

## Project history

- **[plan.md](plan.md)** — the full phase-by-phase development plan and roadmap.
- **[investigation.md](investigation.md)** — the original research conversation that started
  the project.
