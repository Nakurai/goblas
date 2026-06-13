package kernel

// Strsv overrides the generic triangular solve op(A)*x = b (single RHS, in
// place). The inner work per pivot row is a contiguous span of column j:
//   - NoTrans: x[span] -= x[j] * A[span, j]   — a saxpy down the column,
//   - Trans:   x[j]   -= dot(A[span, j], x[span]) — an sdot down the column.
//
// Both reuse the existing NEON L1 kernels on the unit-stride fast path. The
// scalar diagonal division stays in Go. Strided x falls back to the reference.
func (k neonKernel) Strsv(upper, transA, unit bool, n int, a []float32, lda int, x []float32, incX int) {
	if n == 0 {
		return
	}
	if incX != 1 {
		k.genericKernel.Strsv(upper, transA, unit, n, a, lda, x, incX)
		return
	}

	switch {
	case !transA && upper:
		// Back substitution: subtract the solved x[j] times the column above it.
		for j := n - 1; j >= 0; j-- {
			if !unit {
				x[j] /= a[j+j*lda]
			}
			if xj := x[j]; xj != 0 && j > 0 {
				saxpyUnitNEON(j, -xj, &a[j*lda], &x[0])
			}
		}
	case !transA && !upper:
		// Forward substitution: subtract x[j] times the column below it.
		for j := 0; j < n; j++ {
			if !unit {
				x[j] /= a[j+j*lda]
			}
			if xj := x[j]; xj != 0 && j+1 < n {
				col := a[j*lda:]
				saxpyUnitNEON(n-j-1, -xj, &col[j+1], &x[j+1])
			}
		}
	case transA && upper:
		// A^T lower-triangular: forward, each x[j] is a dot against the column.
		for j := 0; j < n; j++ {
			s := x[j]
			if j > 0 {
				s -= sdotUnitNEON(j, &a[j*lda], &x[0])
			}
			if !unit {
				s /= a[j+j*lda]
			}
			x[j] = s
		}
	default:
		// transA && !upper: A^T upper-triangular: back substitution.
		for j := n - 1; j >= 0; j-- {
			s := x[j]
			col := a[j*lda:]
			if j+1 < n {
				s -= sdotUnitNEON(n-j-1, &col[j+1], &x[j+1])
			}
			if !unit {
				s /= a[j+j*lda]
			}
			x[j] = s
		}
	}
}
