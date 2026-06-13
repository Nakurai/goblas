package kernel

// Dger overrides the generic rank-1 update A += alpha*x*y^T. Each column of A
// receives a scaled copy of x: A[:,j] += (alpha*y[j]) * x — which is exactly a
// daxpy. The unit-stride fast path therefore reuses the NEON daxpy kernel,
// streaming each contiguous column of A through it. Strided x falls back to the
// reference (the column writeback would no longer be a unit-stride axpy).
func (k neonKernel) Dger(m, n int, alpha float64, x []float64, incX int, y []float64, incY int, a []float64, lda int) {
	if alpha == 0 || m == 0 || n == 0 {
		return
	}
	if incX != 1 {
		k.genericKernel.Dger(m, n, alpha, x, incX, y, incY, a, lda)
		return
	}
	jy := firstIndex(n, incY)
	for j := 0; j < n; j++ {
		if f := alpha * y[jy]; f != 0 {
			daxpyUnitNEON(m, f, &x[0], &a[j*lda])
		}
		jy += incY
	}
}
