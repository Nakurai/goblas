package kernel

//go:noescape
func dgemvNoTransNEON(m, n int, alpha float64, a *float64, lda int, x, y *float64)

// Dgemv overrides the generic matrix-vector multiply with a NEON kernel for
// the NoTrans, unit-stride fast path. Trans and strided inputs fall back to the
// portable reference automatically via the embedded genericKernel.
func (k neonKernel) Dgemv(trans bool, m, n int, alpha float64, a []float64, lda int, x []float64, incX int, beta float64, y []float64, incY int) {
	if !trans && incX == 1 && incY == 1 {
		// NoTrans unit-stride fast path: scale y ourselves then axpy.
		scaleStrided(m, beta, y, incY)
		if alpha != 0 && m > 0 && n > 0 {
			dgemvNoTransNEON(m, n, alpha, &a[0], lda, &x[0], &y[0])
		}
		return
	}
	// Trans or strided: the generic implementation is correct for all cases.
	k.genericKernel.Dgemv(trans, m, n, alpha, a, lda, x, incX, beta, y, incY)
}
