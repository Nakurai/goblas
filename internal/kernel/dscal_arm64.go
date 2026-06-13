package kernel

//go:noescape
func dscalUnitNEON(n int, alpha float64, x *float64)

// Dscal overrides the generic x *= alpha with a NEON kernel on the unit-stride
// fast path, falling back to the reference for strided input.
func (k neonKernel) Dscal(n int, alpha float64, x []float64, incX int) {
	if incX == 1 {
		dscalUnitNEON(n, alpha, &x[0])
		return
	}
	k.genericKernel.Dscal(n, alpha, x, incX)
}
