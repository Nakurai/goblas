package mat32

import (
	"errors"
	"math"

	"github.com/nakurai/goblas"
)

// Cholesky32 is the Cholesky factorization A = L·Lᵀ of a symmetric
// positive-definite float32 matrix. It is computed entirely in float32 (no
// float64 casting) using a blocked right-looking algorithm whose trailing
// updates run on the goblas Ssyrk/Strsm kernels.
type Cholesky32 struct {
	n int
	l []float32 // column-major lower factor L, leading dimension n
}

// cholBlock is the panel width for the blocked factorization; the trailing
// Ssyrk/Strsm updates (the bulk of the FLOPs) run on goblas at this granularity.
const cholBlock = 64

// Factorize computes the Cholesky factorization of the SPD matrix a. It returns
// false (and leaves the receiver unusable) if a is not positive definite.
func (c *Cholesky32) Factorize(a *SymDense32) (ok bool) {
	buf, n, ld := colMajor(a)
	if !choleskyColMajor(buf, n, ld) {
		return false
	}
	// Keep only L: zero the strict upper triangle.
	for j := 0; j < n; j++ {
		for i := 0; i < j; i++ {
			buf[i+j*ld] = 0
		}
	}
	c.n, c.l = n, buf
	return true
}

// choleskyColMajor factors a (column-major, ld) in place into its lower
// Cholesky factor, returning false if a is not positive definite.
func choleskyColMajor(a []float32, n, ld int) bool {
	for k := 0; k < n; k += cholBlock {
		kb := min(cholBlock, n-k)
		a11 := a[k+k*ld:]
		if !cholUnblocked(a11, kb, ld) {
			return false
		}
		if k+kb < n {
			rows := n - k - kb
			a21 := a[(k+kb)+k*ld:]
			// Solve X·L11ᵀ = A21 for the sub-diagonal panel.
			goblas.Strsm(goblas.Right, goblas.Lower, goblas.Trans, goblas.NonUnit,
				rows, kb, 1, a11, ld, a21, ld)
			// Trailing symmetric update A22 -= A21·A21ᵀ (lower triangle).
			a22 := a[(k+kb)+(k+kb)*ld:]
			goblas.Ssyrk(goblas.Lower, goblas.NoTrans, rows, kb, -1, a21, ld, 1, a22, ld)
		}
	}
	return true
}

// cholUnblocked factors the small kb×kb diagonal block (column-major) with a
// scalar left-looking Cholesky.
func cholUnblocked(a []float32, n, ld int) bool {
	for j := 0; j < n; j++ {
		d := a[j+j*ld]
		for p := 0; p < j; p++ {
			d -= a[j+p*ld] * a[j+p*ld]
		}
		if d <= 0 {
			return false
		}
		d = float32(math.Sqrt(float64(d)))
		a[j+j*ld] = d
		for i := j + 1; i < n; i++ {
			s := a[i+j*ld]
			for p := 0; p < j; p++ {
				s -= a[i+p*ld] * a[j+p*ld]
			}
			a[i+j*ld] = s / d
		}
	}
	return true
}

// SolveTo sets dst = A⁻¹·b using the factorization, where b is n×nrhs. It runs
// the two triangular solves on goblas Strsm.
func (c *Cholesky32) SolveTo(dst *Dense32, b Matrix32) error {
	if c.l == nil {
		return errors.New("mat32: Cholesky32 not factorized")
	}
	br, nrhs := b.Dims()
	if br != c.n {
		return errors.New("mat32: dimension mismatch")
	}
	rhs, n, _, ld := colMajorRHS(b)
	// L·Y = B, then Lᵀ·X = Y.
	goblas.Strsm(goblas.Left, goblas.Lower, goblas.NoTrans, goblas.NonUnit, n, nrhs, 1, c.l, c.n, rhs, ld)
	goblas.Strsm(goblas.Left, goblas.Lower, goblas.Trans, goblas.NonUnit, n, nrhs, 1, c.l, c.n, rhs, ld)
	res := denseFromColMajor(rhs, n, nrhs, ld)
	dst.reuseAsNonZeroed(n, nrhs)
	dst.Copy(res)
	return nil
}

// Det returns the determinant of the factorized matrix (product of squared
// diagonal entries of L). Note that in float32 the determinant overflows to
// +Inf for even moderately sized matrices (the product grows as ~λⁿ); use it
// only for small or well-scaled problems, or work with the diagonal of L
// directly for a log-determinant.
func (c *Cholesky32) Det() float32 {
	var d float32 = 1
	for i := 0; i < c.n; i++ {
		v := c.l[i+i*c.n]
		d *= v * v
	}
	return d
}
