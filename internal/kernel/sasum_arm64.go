package kernel

// sasumUnitNEON returns the sum of |x[i]| over n unit-stride elements.
// Implemented in sasum_arm64.s.
//
//go:noescape
func sasumUnitNEON(n int, x *float32) float32

// Sasum overrides the generic sum of absolute values with a NEON kernel on the
// unit-stride fast path, falling back to the reference for strided input.
func (k neonKernel) Sasum(n int, x []float32, incX int) float32 {
	if n < 1 {
		return 0
	}
	if incX == 1 {
		return sasumUnitNEON(n, &x[0])
	}
	return k.genericKernel.Sasum(n, x, incX)
}
