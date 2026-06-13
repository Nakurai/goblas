// Package blasadapt makes goblas a drop-in BLAS for the Gonum ecosystem.
//
// Gonum's high-level packages (gonum.org/v1/gonum/mat and its pure-Go LAPACK)
// issue every BLAS call through the implementation registered with
// blas64.Use. Calling Use() from this package installs goblas there, so
// mat.Dense.Mul, mat.LU, mat.Cholesky, mat.QR, mat.SVD, (*Dense).Solve and
// friends all run on goblas's accelerated kernels:
//
//	import "github.com/nakurai/goblas/blasadapt"
//
//	func init() { blasadapt.Use() }
//
// Gonum's BLAS interface is row-major while goblas is column-major. The
// bridge is free: a buffer read row-major is the transpose of the same buffer
// read column-major, so each overridden routine only relabels its arguments
// (swap operands and dimensions, flip Trans/Uplo/Side) — no data is copied
// and no extra FLOPs are spent.
//
// Implementation embeds Gonum's own pure-Go BLAS, so every routine goblas
// does not (yet) accelerate falls back to a complete, well-tested
// implementation automatically.
package blasadapt

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	gonumblas "gonum.org/v1/gonum/blas/gonum"

	"github.com/nakurai/goblas"
)

// Implementation is a row-major blas.Float64 backed by goblas kernels for the
// hot routines and Gonum's pure-Go BLAS for everything else.
type Implementation struct {
	gonumblas.Implementation
}

// Use registers goblas as the BLAS used by gonum/mat and gonum's LAPACK.
func Use() { blas64.Use(Implementation{}) }

// trans converts a gonum transpose flag (treating ConjTrans as Trans, exact
// for real matrices).
func trans(t blas.Transpose) goblas.Transpose {
	if t == blas.NoTrans {
		return goblas.NoTrans
	}
	return goblas.Trans
}

func flipUplo(ul blas.Uplo) goblas.Uplo {
	if ul == blas.Upper {
		return goblas.Lower
	}
	return goblas.Upper
}

func flipSide(s blas.Side) goblas.Side {
	if s == blas.Left {
		return goblas.Right
	}
	return goblas.Left
}

func diag(d blas.Diag) goblas.Diag {
	if d == blas.Unit {
		return goblas.Unit
	}
	return goblas.NonUnit
}

// --- Level 1: vectors have no layout; delegate directly ---

func (Implementation) Ddot(n int, x []float64, incX int, y []float64, incY int) float64 {
	return goblas.Ddot(n, x, incX, y, incY)
}

func (Implementation) Daxpy(n int, alpha float64, x []float64, incX int, y []float64, incY int) {
	goblas.Daxpy(n, alpha, x, incX, y, incY)
}

func (Implementation) Dscal(n int, alpha float64, x []float64, incX int) {
	goblas.Dscal(n, alpha, x, incX)
}

// --- Level 2/3: row-major call mapped onto the column-major kernel by
// computing the transposed equation on the same buffers ---

// Dgemv: row-major op(A) ≡ column-major op(Aᵀ), so flip the transpose flag
// and swap the dimensions.
func (Implementation) Dgemv(tA blas.Transpose, m, n int, alpha float64, a []float64, lda int, x []float64, incX int, beta float64, y []float64, incY int) {
	goblas.Dgemv(!trans(tA), n, m, alpha, a, lda, x, incX, beta, y, incY)
}

// Dger: row-major A += α x yᵀ ≡ column-major Aᵀ += α y xᵀ, so swap the
// dimensions and the two vectors (the buffer read column-major is Aᵀ).
func (Implementation) Dger(m, n int, alpha float64, x []float64, incX int, y []float64, incY int, a []float64, lda int) {
	goblas.Dger(n, m, alpha, y, incY, x, incX, a, lda)
}

// Dtrsv: the column-major buffer is Aᵀ, so the triangle flips and op flips
// (NoTrans on A becomes Trans on Aᵀ and vice versa); the single RHS vector x
// has no layout.
func (Implementation) Dtrsv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n int, a []float64, lda int, x []float64, incX int) {
	goblas.Dtrsv(flipUplo(ul), !trans(tA), diag(d), n, a, lda, x, incX)
}

// Dgemm: row-major C = op(A)·op(B) ≡ column-major Cᵀ = op(B)ᵀ·op(A)ᵀ, which
// on the raw buffers means swapping the operands and m/n.
func (Implementation) Dgemm(tA, tB blas.Transpose, m, n, k int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int) {
	goblas.Dgemm(trans(tB), trans(tA), n, m, k, alpha, b, ldb, a, lda, beta, c, ldc)
}

// Dsyrk: transposing C flips the stored triangle and which product
// (A·Aᵀ vs Aᵀ·A) appears.
func (Implementation) Dsyrk(ul blas.Uplo, t blas.Transpose, n, k int, alpha float64, a []float64, lda int, beta float64, c []float64, ldc int) {
	goblas.Dsyrk(flipUplo(ul), !trans(t), n, k, alpha, a, lda, beta, c, ldc)
}

// Dtrsm: transposing both sides of op(A)·X = αB turns it into
// Xᵀ·op(A)ᵀ = αBᵀ — the side and triangle flip, the transpose flag does not.
func (Implementation) Dtrsm(s blas.Side, ul blas.Uplo, tA blas.Transpose, d blas.Diag, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int) {
	goblas.Dtrsm(flipSide(s), flipUplo(ul), trans(tA), diag(d), n, m, alpha, a, lda, b, ldb)
}

// Dsymm: symmetric A is its own transpose; the side and triangle flip.
func (Implementation) Dsymm(s blas.Side, ul blas.Uplo, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int) {
	goblas.Dsymm(flipSide(s), flipUplo(ul), n, m, alpha, a, lda, b, ldb, beta, c, ldc)
}

// Dtrmm: same relabeling as Dtrsm.
func (Implementation) Dtrmm(s blas.Side, ul blas.Uplo, tA blas.Transpose, d blas.Diag, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int) {
	goblas.Dtrmm(flipSide(s), flipUplo(ul), trans(tA), diag(d), n, m, alpha, a, lda, b, ldb)
}
