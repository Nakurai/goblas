package kernel

// ddotUnitNEON computes the unit-stride dot product of n elements of x and y
// using NEON. Implemented in ddot_arm64.s.
//
//go:noescape
func ddotUnitNEON(n int, x, y *float64) float64

// Ddot overrides the generic dot product with a NEON kernel on the unit-stride
// fast path (the common case), falling back to the reference for strided input.
func (k neonKernel) Ddot(n int, x []float64, incX int, y []float64, incY int) float64 {
	if incX == 1 && incY == 1 {
		return ddotUnitNEON(n, &x[0], &y[0])
	}
	return k.genericKernel.Ddot(n, x, incX, y, incY)
}
