package kernel

//go:noescape
func sscalUnitNEON(n int, alpha float32, x *float32)

// Sscal overrides the generic x *= alpha with a NEON kernel on the unit-stride
// fast path, falling back to the reference for strided input.
func (k neonKernel) Sscal(n int, alpha float32, x []float32, incX int) {
	if incX == 1 {
		sscalUnitNEON(n, alpha, &x[0])
		return
	}
	k.genericKernel.Sscal(n, alpha, x, incX)
}
