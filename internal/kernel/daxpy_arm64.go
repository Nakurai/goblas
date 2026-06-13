package kernel

//go:noescape
func daxpyUnitNEON(n int, alpha float64, x, y *float64)

// Daxpy overrides the generic y += alpha*x with a NEON kernel on the unit-stride
// fast path, falling back to the reference for strided input.
func (k neonKernel) Daxpy(n int, alpha float64, x []float64, incX int, y []float64, incY int) {
	if alpha == 0 {
		return
	}
	if incX == 1 && incY == 1 {
		daxpyUnitNEON(n, alpha, &x[0], &y[0])
		return
	}
	k.genericKernel.Daxpy(n, alpha, x, incX, y, incY)
}
