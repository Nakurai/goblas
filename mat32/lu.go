package mat32

import (
	"errors"

	"github.com/nakurai/goblas"
)

// LU32 is the LU factorization with partial pivoting (P·A = L·U) of a square
// float32 matrix. It is computed entirely in float32 (no float64 casting) with
// a blocked right-looking algorithm: the panel factorization uses goblas
// Isamax/Sswap/Sscal/Sger, and the trailing update runs on Strsm/Sgemm.
type LU32 struct {
	n    int
	lu   []float32 // column-major; unit-lower L (no diagonal) over upper U
	piv  []int     // row interchanges: step i swapped rows i and piv[i]
	sign float32   // determinant sign from the permutation
}

const luBlock = 64

// Factorize computes the LU factorization of the square matrix a.
func (lu *LU32) Factorize(a Matrix32) {
	buf, n, ld := colMajor(a)
	piv := make([]int, n)
	sign := luColMajor(buf, n, ld, piv)
	lu.n, lu.lu, lu.piv, lu.sign = n, buf, piv, sign
}

// luColMajor factors a (column-major, ld) in place and fills piv; it returns the
// sign of the row permutation.
func luColMajor(a []float32, n, ld int, piv []int) float32 {
	sign := float32(1)
	for k := 0; k < n; k += luBlock {
		kb := min(luBlock, n-k)
		if luPanel(a, n, k, kb, ld, piv) {
			sign = -sign
		}
		if k+kb < n {
			a11 := a[k+k*ld:]
			a12 := a[k+(k+kb)*ld:]
			// U12 = L11⁻¹·A12 (unit lower triangular solve).
			goblas.Strsm(goblas.Left, goblas.Lower, goblas.NoTrans, goblas.Unit,
				kb, n-k-kb, 1, a11, ld, a12, ld)
			// A22 -= A21·U12.
			a21 := a[(k+kb)+k*ld:]
			a22 := a[(k+kb)+(k+kb)*ld:]
			goblas.Sgemm(goblas.NoTrans, goblas.NoTrans, n-k-kb, n-k-kb, kb,
				-1, a21, ld, a12, ld, 1, a22, ld)
		}
	}
	return sign
}

// luPanel factors the panel a[k:n, k:k+kb] (column-major) with partial pivoting,
// swapping full rows across the whole matrix. It returns true if an odd number
// of row interchanges occurred.
func luPanel(a []float32, n, k, kb, ld int, piv []int) bool {
	flip := false
	for jj := 0; jj < kb; jj++ {
		col := k + jj
		// Pivot: index of max |.| in column col, rows [col, n).
		p := col + goblas.Isamax(n-col, a[col+col*ld:], 1)
		piv[col] = p
		if p != col {
			// Swap full rows col and p (stride ld across all n columns).
			goblas.Sswap(n, a[col:], ld, a[p:], ld)
			flip = !flip
		}
		if pivot := a[col+col*ld]; pivot != 0 && col+1 < n {
			goblas.Sscal(n-col-1, 1/pivot, a[(col+1)+col*ld:], 1)
		}
		// Rank-1 update of the rest of the panel.
		if jj+1 < kb && col+1 < n {
			goblas.Sger(n-col-1, kb-jj-1, -1,
				a[(col+1)+col*ld:], 1, a[col+(col+1)*ld:], ld, a[(col+1)+(col+1)*ld:], ld)
		}
	}
	return flip
}

// SolveTo sets dst = A⁻¹·b using the factorization, where b is n×nrhs.
func (lu *LU32) SolveTo(dst *Dense32, b Matrix32) error {
	if lu.lu == nil {
		return errors.New("mat32: LU32 not factorized")
	}
	br, nrhs := b.Dims()
	if br != lu.n {
		return errors.New("mat32: dimension mismatch")
	}
	rhs, n, _, ld := colMajorRHS(b)
	// Apply row interchanges P·b.
	for i := 0; i < n; i++ {
		if lu.piv[i] != i {
			goblas.Sswap(nrhs, rhs[i:], ld, rhs[lu.piv[i]:], ld)
		}
	}
	// Forward solve L·Y = P·b (unit lower), then back solve U·X = Y.
	goblas.Strsm(goblas.Left, goblas.Lower, goblas.NoTrans, goblas.Unit, n, nrhs, 1, lu.lu, lu.n, rhs, ld)
	goblas.Strsm(goblas.Left, goblas.Upper, goblas.NoTrans, goblas.NonUnit, n, nrhs, 1, lu.lu, lu.n, rhs, ld)
	res := denseFromColMajor(rhs, n, nrhs, ld)
	dst.reuseAsNonZeroed(n, nrhs)
	dst.Copy(res)
	return nil
}

// Det returns the determinant of the factorized matrix. As with Cholesky32.Det,
// the float32 result overflows to ±Inf for even moderately sized matrices; it
// is meaningful only for small or well-scaled problems.
func (lu *LU32) Det() float32 {
	d := lu.sign
	for i := 0; i < lu.n; i++ {
		d *= lu.lu[i+i*lu.n]
	}
	return d
}

// Solve sets dst = A⁻¹·b for a square matrix a, via an LU factorization.
func (dst *Dense32) Solve(a, b Matrix32) error {
	var lu LU32
	lu.Factorize(a)
	return lu.SolveTo(dst, b)
}
