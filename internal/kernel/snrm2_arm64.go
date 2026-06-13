package kernel

import "math"

// sssqUnitNEON returns the sum of squares of n unit-stride float32 elements.
// Implemented in snrm2_arm64.s.
//
//go:noescape
func sssqUnitNEON(n int, x *float32) float32

// Snrm2 overrides the generic Euclidean norm with a NEON sum-of-squares fast
// path. Naive sum of squares can overflow (huge elements) or underflow to zero
// (tiny elements), so the fast result is only trusted when it is a normal
// positive number; otherwise we fall back to the generic overflow-safe LAPACK
// scaled algorithm, which also handles the all-zero vector.
func (k neonKernel) Snrm2(n int, x []float32, incX int) float32 {
	if n < 1 {
		return 0
	}
	if incX == 1 && n >= 2 {
		ssq := sssqUnitNEON(n, &x[0])
		if ssq > 0 && !math.IsInf(float64(ssq), 1) {
			return float32(math.Sqrt(float64(ssq)))
		}
	}
	return k.genericKernel.Snrm2(n, x, incX)
}
