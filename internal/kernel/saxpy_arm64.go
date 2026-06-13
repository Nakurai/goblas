package kernel

//go:noescape
func saxpyUnitNEON(n int, alpha float32, x, y *float32)

// Saxpy overrides the generic y += alpha*x with a NEON kernel on the
// unit-stride fast path, falling back to the reference for strided input.
func (k neonKernel) Saxpy(n int, alpha float32, x []float32, incX int, y []float32, incY int) {
	if alpha == 0 {
		return
	}
	if incX == 1 && incY == 1 {
		saxpyUnitNEON(n, alpha, &x[0], &y[0])
		return
	}
	k.genericKernel.Saxpy(n, alpha, x, incX, y, incY)
}
