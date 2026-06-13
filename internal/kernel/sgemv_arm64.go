package kernel

//go:noescape
func sgemvNoTransNEON(m, n int, alpha float32, a *float32, lda int, x, y *float32)

// Sgemv overrides the generic matrix-vector multiply with a NEON kernel for
// the NoTrans, unit-stride fast path. Trans and strided inputs fall back to the
// portable reference automatically via the embedded genericKernel.
func (k neonKernel) Sgemv(trans bool, m, n int, alpha float32, a []float32, lda int, x []float32, incX int, beta float32, y []float32, incY int) {
	if !trans && incX == 1 && incY == 1 {
		// NoTrans unit-stride fast path: scale y ourselves then axpy.
		scaleStrided(m, beta, y, incY)
		if alpha != 0 && m > 0 && n > 0 {
			sgemvNoTransNEON(m, n, alpha, &a[0], lda, &x[0], &y[0])
		}
		return
	}
	k.genericKernel.Sgemv(trans, m, n, alpha, a, lda, x, incX, beta, y, incY)
}
