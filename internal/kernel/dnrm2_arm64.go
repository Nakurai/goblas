package kernel

import "math"

// dssqUnitNEON returns the sum of squares of n unit-stride elements of x.
// Implemented in dnrm2_arm64.s.
//
//go:noescape
func dssqUnitNEON(n int, x *float64) float64

// Dnrm2 overrides the generic Euclidean norm with a NEON sum-of-squares fast
// path. Naive sum of squares can overflow (huge elements) or underflow to zero
// (tiny elements), so the fast result is only trusted when it is a normal
// positive number; otherwise we fall back to the generic overflow-safe LAPACK
// scaled algorithm, which also handles the all-zero vector. For the common
// case of normal-range data this is one branchless pass.
func (k neonKernel) Dnrm2(n int, x []float64, incX int) float64 {
	if n < 1 {
		return 0
	}
	if incX == 1 && n >= 2 {
		ssq := dssqUnitNEON(n, &x[0])
		if ssq > 0 && !math.IsInf(ssq, 1) {
			return math.Sqrt(ssq)
		}
	}
	return k.genericKernel.Dnrm2(n, x, incX)
}
