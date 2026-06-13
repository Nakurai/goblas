package kernel

// sdotUnitNEON computes the unit-stride float32 dot product of n elements of x
// and y using NEON. Implemented in sdot_arm64.s.
//
//go:noescape
func sdotUnitNEON(n int, x, y *float32) float32

// Sdot overrides the generic dot product with a NEON kernel on the unit-stride
// fast path, falling back to the reference for strided input.
func (k neonKernel) Sdot(n int, x []float32, incX int, y []float32, incY int) float32 {
	if incX == 1 && incY == 1 {
		return sdotUnitNEON(n, &x[0], &y[0])
	}
	return k.genericKernel.Sdot(n, x, incX, y, incY)
}
