package blasadapt

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas32"

	"github.com/nakurai/goblas"
)

// This file adds the float32 (single-precision) side of the adapter. The
// embedded gonum implementation already satisfies blas.Float32, so Implementation
// is a complete float32 BLAS out of the box; the methods below override the hot
// routines with goblas kernels using the same zero-copy row↔column relabeling as
// the float64 side (see adapter.go).
//
// Note the gonum asymmetry: there is no float32 LAPACK and gonum/mat is
// float64-only, so Use32 accelerates BLAS-level float32 work (blas32.General /
// blas32.Vector and the blas32.* functions) but cannot accelerate high-level
// float32 factorizations/solvers — those do not exist upstream. See plan.md.

// Use32 registers goblas as the float32 BLAS used by gonum's blas32 package.
// It is independent of Use (float64): a program may call both, and float32 and
// float64 work then each dispatch to their own goblas-backed kernels.
func Use32() { blas32.Use(Implementation{}) }

// --- Level 1: vectors have no layout; delegate directly ---

func (Implementation) Sdot(n int, x []float32, incX int, y []float32, incY int) float32 {
	return goblas.Sdot(n, x, incX, y, incY)
}

func (Implementation) Saxpy(n int, alpha float32, x []float32, incX int, y []float32, incY int) {
	goblas.Saxpy(n, alpha, x, incX, y, incY)
}

func (Implementation) Sscal(n int, alpha float32, x []float32, incX int) {
	goblas.Sscal(n, alpha, x, incX)
}

// --- Level 2/3: row-major call mapped onto the column-major kernel by
// computing the transposed equation on the same buffers (see adapter.go) ---

func (Implementation) Sgemv(tA blas.Transpose, m, n int, alpha float32, a []float32, lda int, x []float32, incX int, beta float32, y []float32, incY int) {
	goblas.Sgemv(!trans(tA), n, m, alpha, a, lda, x, incX, beta, y, incY)
}

func (Implementation) Sger(m, n int, alpha float32, x []float32, incX int, y []float32, incY int, a []float32, lda int) {
	goblas.Sger(n, m, alpha, y, incY, x, incX, a, lda)
}

func (Implementation) Strsv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n int, a []float32, lda int, x []float32, incX int) {
	goblas.Strsv(flipUplo(ul), !trans(tA), diag(d), n, a, lda, x, incX)
}

func (Implementation) Sgemm(tA, tB blas.Transpose, m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	goblas.Sgemm(trans(tB), trans(tA), n, m, k, alpha, b, ldb, a, lda, beta, c, ldc)
}

func (Implementation) Ssyrk(ul blas.Uplo, t blas.Transpose, n, k int, alpha float32, a []float32, lda int, beta float32, c []float32, ldc int) {
	goblas.Ssyrk(flipUplo(ul), !trans(t), n, k, alpha, a, lda, beta, c, ldc)
}

func (Implementation) Strsm(s blas.Side, ul blas.Uplo, tA blas.Transpose, d blas.Diag, m, n int, alpha float32, a []float32, lda int, b []float32, ldb int) {
	goblas.Strsm(flipSide(s), flipUplo(ul), trans(tA), diag(d), n, m, alpha, a, lda, b, ldb)
}

func (Implementation) Ssymm(s blas.Side, ul blas.Uplo, m, n int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	goblas.Ssymm(flipSide(s), flipUplo(ul), n, m, alpha, a, lda, b, ldb, beta, c, ldc)
}

func (Implementation) Strmm(s blas.Side, ul blas.Uplo, tA blas.Transpose, d blas.Diag, m, n int, alpha float32, a []float32, lda int, b []float32, ldb int) {
	goblas.Strmm(flipSide(s), flipUplo(ul), trans(tA), diag(d), n, m, alpha, a, lda, b, ldb)
}
