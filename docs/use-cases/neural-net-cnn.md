# Neural networks II — convolutional networks (CNNs) via im2col

Convolutional neural networks power most image recognition. A convolution looks nothing like a
matrix multiply — it slides a little filter over an image. Yet the standard way every framework
(PyTorch, TensorFlow, cuDNN) actually *computes* convolution is to reshape it into a single
`Dgemm`. That reshape is called **im2col**, and it is the reason a fast BLAS makes CNNs fast.
Read [neural-net-mlp.md](neural-net-mlp.md) first.

**Real-world examples**: Facial recognition, self-driving cars interpreting camera feeds, or medical image analysis (like finding tumors in x-rays).

## What a convolution does

Instead of connecting every input to every neuron (as an MLP does), a convolution uses a small
**filter** (say 3×3) and slides it across the image. At each position it multiplies the filter
against the patch of pixels underneath and sums — producing one output pixel. The same filter is
reused at every position, so it learns to detect a local pattern (an edge, a texture) wherever
it appears.

```
image (5×5)        filter (3×3)        slide it everywhere →
┌───────────┐      ┌─────┐             output feature map
│ . . . . . │      │ a b c│
│ . [patch] │  ⊛   │ d e f│   =        each output pixel =
│ . . . . . │      │ g h i│            sum(filter ⊙ patch)
└───────────┘      └─────┘
```

A real layer has many filters (each finds a different pattern) and the input has multiple
channels (e.g. red/green/blue). Computed naively, that is a deep nest of loops — over output
positions, filters, channels, and filter rows/columns. Slow, and exactly the scalar code Go
cannot vectorize.

## The im2col trick: convolution → one Dgemm

Here is the insight. At each filter position, the operation is a **dot product** between the
flattened filter and the flattened patch under it. A dot product of every filter against every
patch is... a matrix multiply. So:

1. **im2col** ("image to columns"): walk over every patch position in the image and copy that
   patch, flattened, into a column of a big matrix `P`. If there are `L` output positions and
   each patch has `c = channels × filterH × filterW` values, `P` is `c × L`.
2. **Stack the filters**: flatten each of the `f` filters into a row of a matrix `W` (`f × c`).
3. **One matrix multiply** does the entire convolution:

```
Out = W · P        →  Dgemm, an (f × L) matrix
```

Row `i`, column `j` of `Out` is exactly "filter `i` applied at position `j`" — the whole
convolution, every filter at every location, as one goblas `Dgemm`. Reshape `Out` back to the
output image dimensions and you are done.

This is not a goblas peculiarity — it is how the heavy hitters do it, because it converts an
awkward, cache-unfriendly loop nest into the one operation hardware is best at.

## Building it with gonum/mat

Setup:

```go
import (
    "github.com/nakurai/goblas/blasadapt"
    "gonum.org/v1/gonum/mat"
)

func init() { blasadapt.Use() }
```

**Step 1 — im2col.** This part is plain Go data movement (no math, no BLAS): for each output
position, gather the underlying patch into a column. For a single-channel `H×W` image, a `kH×kW`
filter, and stride 1 (you can use the synthetic `images_tiny.csv` dataset in the `data/` folder):

```go
outH, outW := H-kH+1, W-kW+1
L := outH * outW          // number of patch positions
c := kH * kW              // values per patch (×channels if multi-channel)

patches := mat.NewDense(c, L, nil)
col := 0
for oy := 0; oy < outH; oy++ {
    for ox := 0; ox < outW; ox++ {
        row := 0
        for fy := 0; fy < kH; fy++ {
            for fx := 0; fx < kW; fx++ {
                patches.Set(row, col, image.At(oy+fy, ox+fx))
                row++
            }
        }
        col++
    }
}
```

**Step 2 — stack the filters** as rows of `W` (`f` filters, each flattened to length `c`):

```go
W := mat.NewDense(f, c, filterData) // each row is one flattened filter
```

**Step 3 — the convolution is one goblas `Dgemm`:**

```go
var out mat.Dense
out.Mul(W, patches)   // (f × c)·(c × L) = f × L   ← the whole conv layer
```

Each of the `f` rows of `out` is one filter's output feature map, laid out as `L = outH·outW`
values; reshape it back to `outH × outW`. Add a bias, apply an activation (ReLU, as in the MLP),
and that is a complete convolutional layer.

## Pooling and the rest of the network

CNNs interleave convolution with **pooling** (downsampling — e.g. take the max of each 2×2
block), then usually finish with one or two fully-connected (MLP) layers for the final
classification. Pooling is a cheap elementwise/reduction pass — not BLAS work — while the
fully-connected layers are `Dgemm`s exactly as in the [MLP tutorial](neural-net-mlp.md). So a
whole CNN's compute is: im2col + `Dgemm` per conv layer (the bulk), cheap pooling, and `Dgemm`
for the dense layers. **Backpropagation** through a conv layer is again matrix multiplies
(`col2im`, the reverse of im2col, plus `Dgemm`s for the gradients).

## The honest cost note

im2col itself (Step 1) is memory copying, not arithmetic, and it duplicates overlapping pixels —
so it trades extra memory for the ability to use a fast `Dgemm`. For typical filter sizes that
trade is overwhelmingly worth it, which is why it is the industry-standard approach. The copying
is plain Go and not accelerated by goblas; the `Dgemm` that follows — the actual `f·c·L`
arithmetic — is.

## Where goblas earned its keep

| Step | BLAS routine | goblas role |
|------|--------------|-------------|
| im2col (gather patches) | — | plain Go memory movement |
| Convolution `W·P` | `Dgemm` | accelerated — the bulk of the cost |
| Fully-connected layers | `Dgemm` | accelerated |
| Pooling, activations | elementwise/reduction | plain Go (cheap) |

## Recap

- A convolution slides filters over an image; computed directly it is a slow loop nest.
- **im2col** flattens every patch into the columns of a matrix; then the entire layer is one
  `Dgemm` (`W·P`) — the same approach real frameworks use.
- goblas accelerates that `Dgemm` (and the dense layers); im2col and pooling are cheap plain Go.

Next: [neural-net-lstm.md](neural-net-lstm.md) — recurrent networks for sequences, where the
matrix multiply happens once per time step.
