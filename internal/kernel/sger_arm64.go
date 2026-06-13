package kernel

// Sger overrides the generic rank-1 update A += alpha*x*y^T. Each column of A
// receives a scaled copy of x: A[:,j] += (alpha*y[j]) * x — exactly a saxpy.
// The unit-stride fast path reuses the NEON saxpy kernel, streaming each
// contiguous column of A through it. Strided x falls back to the reference.
func (k neonKernel) Sger(m, n int, alpha float32, x []float32, incX int, y []float32, incY int, a []float32, lda int) {
	if alpha == 0 || m == 0 || n == 0 {
		return
	}
	if incX != 1 {
		k.genericKernel.Sger(m, n, alpha, x, incX, y, incY, a, lda)
		return
	}
	jy := firstIndex(n, incY)
	for j := 0; j < n; j++ {
		if f := alpha * y[jy]; f != 0 {
			saxpyUnitNEON(m, f, &x[0], &a[j*lda])
		}
		jy += incY
	}
}
