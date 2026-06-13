package mat32

import (
	"math"

	"github.com/nakurai/goblas"
)

func gblasTrans(t bool) goblas.Transpose {
	if t {
		return goblas.Trans
	}
	return goblas.NoTrans
}

// Mul sets m = a*b, the matrix product. The inner dimensions must match. When
// both operands are Dense32-backed (directly or through a Transpose32 view) the
// product runs on the goblas Sgemm kernel via the row→column relabel identity
// (row-major C = op(A)·op(B) ⟺ column-major Cᵀ = op(B)ᵀ·op(A)ᵀ); otherwise a
// portable float32 triple-loop is used. The whole operation is float32 — no
// casting.
func (m *Dense32) Mul(a, b Matrix32) {
	ar, ac := a.Dims()
	br, bc := b.Dims()
	if ac != br {
		panic("mat32: dimension mismatch")
	}
	rows, cols, k := ar, bc, ac

	aD, aT, aok := underlying(a)
	bD, bT, bok := underlying(b)

	// If the receiver aliases an input, compute into a temporary first.
	if (aok && aD == m) || (bok && bD == m) {
		var tmp Dense32
		tmp.Mul(a, b)
		m.reuseAsNonZeroed(rows, cols)
		m.Copy(&tmp)
		return
	}
	m.reuseAsNonZeroed(rows, cols)

	if aok && bok {
		goblas.Sgemm(gblasTrans(bT), gblasTrans(aT), cols, rows, k,
			1, bD.data, bD.stride, aD.data, aD.stride, 0, m.data, m.stride)
		return
	}

	// Generic fallback: at least one operand is not Dense32-backed.
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			var s float32
			for l := 0; l < k; l++ {
				s += a.At(i, l) * b.At(l, j)
			}
			m.data[i*m.stride+j] = s
		}
	}
}

// MulVec sets the receiver vector to a*x. The matrix's column count must equal
// x's length. Dense32×VecDense32 runs on the goblas Sgemv kernel; other
// combinations use a portable float32 loop.
func (v *VecDense32) MulVec(a Matrix32, x Vector32) {
	ar, ac := a.Dims()
	if ac != x.Len() {
		panic("mat32: dimension mismatch")
	}
	aD, aT, aok := underlying(a)
	xv, xok := x.(*VecDense32)

	if aok && xok && xv != v {
		v.reuseAsNonZeroed(ar)
		// Mirror blasadapt.Dgemv: gonum Dgemv(tA, m0, n0) on the stored buffer
		// maps to goblas.Sgemv(!tA, n0, m0, ...).
		goblas.Sgemv(gblasTrans(!aT), aD.cols, aD.rows, 1, aD.data, aD.stride,
			xv.data, xv.inc, 0, mustVecResult(v, ar), v.inc)
		return
	}

	// Generic fallback.
	out := make([]float32, ar)
	for i := 0; i < ar; i++ {
		var s float32
		for j := 0; j < ac; j++ {
			s += a.At(i, j) * x.AtVec(j)
		}
		out[i] = s
	}
	v.reuseAsNonZeroed(ar)
	for i := 0; i < ar; i++ {
		v.data[i*v.inc] = out[i]
	}
}

// mustVecResult ensures v has a length-n unit-or-strided backing and returns it.
func mustVecResult(v *VecDense32, n int) []float32 {
	v.reuseAsNonZeroed(n)
	return v.data
}

// Add sets m = a + b (element-wise). a and b must have the same dimensions.
func (m *Dense32) Add(a, b Matrix32) {
	m.elementwise(a, b, func(x, y float32) float32 { return x + y })
}

// Sub sets m = a - b (element-wise).
func (m *Dense32) Sub(a, b Matrix32) {
	m.elementwise(a, b, func(x, y float32) float32 { return x - y })
}

// MulElem sets m = a ∘ b (element-wise / Hadamard product).
func (m *Dense32) MulElem(a, b Matrix32) {
	m.elementwise(a, b, func(x, y float32) float32 { return x * y })
}

// DivElem sets m = a / b (element-wise).
func (m *Dense32) DivElem(a, b Matrix32) {
	m.elementwise(a, b, func(x, y float32) float32 { return x / y })
}

func (m *Dense32) elementwise(a, b Matrix32, op func(x, y float32) float32) {
	ar, ac := a.Dims()
	br, bc := b.Dims()
	if ar != br || ac != bc {
		panic("mat32: dimension mismatch")
	}
	m.reuseAsNonZeroed(ar, ac)
	for i := 0; i < ar; i++ {
		for j := 0; j < ac; j++ {
			m.data[i*m.stride+j] = op(a.At(i, j), b.At(i, j))
		}
	}
}

// Scale sets m = f*a.
func (m *Dense32) Scale(f float32, a Matrix32) {
	ar, ac := a.Dims()
	m.reuseAsNonZeroed(ar, ac)
	for i := 0; i < ar; i++ {
		for j := 0; j < ac; j++ {
			m.data[i*m.stride+j] = f * a.At(i, j)
		}
	}
}

// Apply sets m[i,j] = fn(i, j, a[i,j]).
func (m *Dense32) Apply(fn func(i, j int, v float32) float32, a Matrix32) {
	ar, ac := a.Dims()
	m.reuseAsNonZeroed(ar, ac)
	for i := 0; i < ar; i++ {
		for j := 0; j < ac; j++ {
			m.data[i*m.stride+j] = fn(i, j, a.At(i, j))
		}
	}
}

// Outer sets m = alpha * x * yᵀ, an r×c rank-one matrix where r = x.Len() and
// c = y.Len(). It runs on the goblas Sger kernel.
func (m *Dense32) Outer(alpha float32, x, y Vector32) {
	r, c := x.Len(), y.Len()
	m.reuseAsNonZeroed(r, c)
	m.Zero()
	xv, xok := x.(*VecDense32)
	yv, yok := y.(*VecDense32)
	if xok && yok {
		// Mirror blasadapt.Dger: gonum A += alpha x yᵀ maps to a column-major
		// Sger with x/y swapped and dims swapped.
		goblas.Sger(c, r, alpha, yv.data, yv.inc, xv.data, xv.inc, m.data, m.stride)
		return
	}
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			m.data[i*m.stride+j] = alpha * x.AtVec(i) * y.AtVec(j)
		}
	}
}

// Norm returns the Frobenius norm of the matrix, sqrt(sum of squares). It uses
// the goblas Snrm2 kernel on the contiguous fast path.
func (m *Dense32) Norm() float32 {
	if m.stride == m.cols {
		return goblas.Snrm2(m.rows*m.cols, m.data, 1)
	}
	var ssq float64
	for i := 0; i < m.rows; i++ {
		row := m.data[i*m.stride : i*m.stride+m.cols]
		for _, v := range row {
			ssq += float64(v) * float64(v)
		}
	}
	return float32(math.Sqrt(ssq))
}
