# Object detection with YOLO

This tutorial puts the [CNN tutorial](neural-net-cnn.md) to work on a real, recognizable task:
**object detection** — not just "is there a cat?" but "there is a cat *here*, a dog *there*,"
drawing boxes around each. **YOLO** ("You Only Look Once") is the famous family of real-time
detectors. It is an excellent lens for one of goblas's most important lessons: a real-world
model is **part dense linear algebra (which goblas accelerates) and part not (which it does
not).** Read [neural-net-cnn.md](neural-net-cnn.md) first.

**Real-world examples**: Real-time pedestrian tracking for self-driving cars, or quality control identifying defective parts on a fast-moving assembly line.

> **Honesty up front.** YOLO is a large convolutional network. Running one in float64 on the
> CPU through gonum is an *educational* exercise — you will understand exactly how detection
> works and see where the compute goes, but it will be far slower than a real deployment
> (float32, GPU, or an optimized engine like ONNX Runtime / NCNN). Treat this as "how YOLO
> works and which parts goblas speeds up," not "deploy YOLO on goblas."

## The problem: from classification to detection

A classifier answers one question about a whole image. A **detector** must answer, for an image
that may contain several objects: *what* is each object, and *where* (a bounding box)? That is
harder, because the number of objects is not fixed and their positions matter.

```
   ┌─────────────────────┐
   │   ┌─────┐           │
   │   │ dog │   ┌────┐  │   detection output:
   │   └─────┘   │ cat│  │   • dog,  box (x,y,w,h), confidence 0.94
   │             └────┘  │   • cat,  box (x,y,w,h), confidence 0.88
   └─────────────────────┘
```

## YOLO's key idea: detection in a single forward pass

Older detectors ran a classifier thousands of times over candidate regions — slow. YOLO's
insight (the "only look once") is to treat detection as **one** forward pass of a single
convolutional network that outputs *all* the boxes at once.

It works by dividing the image into a grid. The convolutional network processes the whole image
and, for each grid cell, predicts: a few candidate boxes (positions and sizes), a confidence
that each box contains an object, and class probabilities. One image in, a grid of predictions
out — in a single pass.

```
image → [ convolutional backbone ] → [ neck ] → [ detection head ] → grid of box predictions
            (most of the compute)                                      → postprocess → final boxes
```

The three network parts:

- **Backbone** — a deep stack of convolutional layers that extract features (edges → textures →
  shapes → object parts). This is the bulk of the computation.
- **Neck** — combines features at multiple scales so the model can detect both big and small
  objects.
- **Head** — a few final convolutions that turn features into the actual box/confidence/class
  numbers per grid cell.

## Where goblas accelerates — and where it cannot

Here is the lesson this tutorial exists to teach. Walk through what YOLO actually computes:

**The backbone, neck, and head are convolutions.** And from the [CNN tutorial](neural-net-cnn.md)
you already know the punchline: a convolution becomes, via **im2col**, a single `Dgemm`. The
overwhelming majority of YOLO's compute — easily 90%+ — is these convolutions, i.e. matrix
multiplies. **goblas accelerates all of it**, exactly as in the CNN tutorial; there is nothing
new in the *mechanism*, just many more layers of it.

```go
// Each conv layer in the backbone is the same pattern as the CNN tutorial:
//   patches = im2col(featureMap)      // plain Go gather
//   out.Mul(filters, patches)         // ← Dgemm on goblas, repeated for every layer
```

**But the detection postprocessing is not BLAS work.** After the network produces its grid of
raw predictions, two steps finish the job, and neither is a matrix multiply:

1. **Decoding** — convert each grid cell's raw numbers into actual box coordinates and apply a
   confidence threshold. This is cheap elementwise arithmetic.
2. **Non-Maximum Suppression (NMS)** — the network typically predicts the *same* object several
   times with overlapping boxes. NMS keeps the highest-confidence box and suppresses the others
   that overlap it too much. This is a **sorting-and-comparing** algorithm — sort boxes by
   confidence, then repeatedly compare overlaps and discard. It is branchy and sequential,
   exactly the kind of work the [overview](../overview.md#inherent-limitations-the-cost-of-the-idea-itself)
   warns BLAS does nothing for. goblas does **not** accelerate NMS — and nor would a GPU's
   matrix units; NMS is a different kind of computation.

This is the whole point of including YOLO. A real model is a **pipeline**: a big BLAS-bound core
(the convolutional network — goblas's home turf) bookended by non-BLAS glue (NMS, decoding).
Knowing which is which tells you exactly what a fast BLAS does and does not buy you.

## Sketching it with gonum/mat

You would not realistically hand-build YOLO, but the *shape* is worth seeing. The backbone is a
loop over conv layers, each an im2col + `Dgemm` (from the CNN tutorial), with pooling/activation
between:

```go
import (
    "github.com/nakurai/goblas/blasadapt"
    "gonum.org/v1/gonum/mat"
)

func init() { blasadapt.Use() }

// Backbone: feature maps flow through many conv layers. Each layer:
// (You could use the `images_tiny.csv` dataset in `data/` to represent a tiny image input)
func convLayer(input *mat.Dense, filters *mat.Dense, kH, kW int) *mat.Dense {
    patches := im2col(input, kH, kW) // plain Go (see the CNN tutorial)
    var out mat.Dense
    out.Mul(filters, patches)         // ← Dgemm on goblas
    relu(&out)                        // elementwise
    return &out
}
```

The head produces, per grid cell, raw box/confidence/class numbers (more `Dgemm`s). Then the
non-BLAS tail:

```go
boxes := decodePredictions(headOutput) // elementwise: grid cell → box coords + scores
final := nonMaxSuppression(boxes, 0.5)  // sort + overlap test — NOT BLAS, plain Go
```

`nonMaxSuppression` is a loop that sorts candidate boxes by confidence and walks them, dropping
any that overlap an already-kept box by more than the threshold. No matrix multiply appears — and
that is correct and expected.

## Where goblas earned its keep

| Stage | Operation | goblas role |
|-------|-----------|-------------|
| Backbone conv layers | im2col + `Dgemm` (per layer) | accelerated — the bulk of the cost |
| Neck (multi-scale fusion) | convolutions = `Dgemm` | accelerated |
| Detection head | convolutions = `Dgemm` | accelerated |
| im2col, pooling, activations | gather / elementwise | plain Go (cheap) |
| Decode predictions | elementwise | plain Go (cheap) |
| **Non-Maximum Suppression** | sort + overlap comparisons | **not BLAS — not accelerated** |

## Recap

- YOLO detects all objects in one forward pass of a convolutional network (backbone → neck →
  head), then postprocesses the grid of predictions into final boxes.
- The network is convolutions, and convolutions are `Dgemm` (via im2col) — so goblas accelerates
  ~90%+ of YOLO's compute, with no new mechanism beyond the [CNN tutorial](neural-net-cnn.md).
- The postprocessing — decoding and especially **Non-Maximum Suppression** — is branchy,
  sorting-style work that goblas (and matrix hardware in general) does not speed up.
- Caveat: float64/CPU makes this educational, not a deployable detector. The value is seeing,
  concretely, that a real model is a BLAS-bound core wrapped in non-BLAS glue.

Back to the [use-cases index](README.md).
