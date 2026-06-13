package kernel

import "math"

// float is the element-type constraint shared by every generic primitive: the
// float64 (D-prefixed) and float32 (S-prefixed) BLAS routines are the same code
// instantiated at the two element types. Go monomorphizes the two as distinct
// gcshapes (different sizes), so there is no boxing or dictionary overhead on
// the arithmetic and the hand-written assembly kernels drop straight in.
type float interface {
	~float32 | ~float64
}

// absT is |x| for any float type (math.Abs is float64-only).
func absT[T float](x T) T {
	if x < 0 {
		return -x
	}
	return x
}

// sqrtT computes the square root in the element's own precision by routing
// through math.Sqrt; for float32 the round-trip matches a hardware single-
// precision sqrt to the last bit for in-range inputs.
func sqrtT[T float](x T) T {
	return T(math.Sqrt(float64(x)))
}
