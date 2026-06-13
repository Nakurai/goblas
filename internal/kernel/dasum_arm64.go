package kernel

// dasumUnitNEON returns the sum of |x[i]| over n unit-stride elements.
// Implemented in dasum_arm64.s.
//
//go:noescape
func dasumUnitNEON(n int, x *float64) float64

// Dasum overrides the generic sum of absolute values with a NEON kernel on the
// unit-stride fast path, falling back to the reference for strided input.
func (k neonKernel) Dasum(n int, x []float64, incX int) float64 {
	if n < 1 {
		return 0
	}
	if incX == 1 {
		return dasumUnitNEON(n, &x[0])
	}
	return k.genericKernel.Dasum(n, x, incX)
}
