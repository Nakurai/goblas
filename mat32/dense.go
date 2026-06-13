package mat32

import "gonum.org/v1/gonum/blas/blas32"

// Dense32 is a row-major dense float32 matrix: element (i,j) is stored at
// data[i*stride+j]. The zero value is an empty matrix that the in-place result
// methods (Mul, Add, …) will allocate on first use.
type Dense32 struct {
	rows, cols, stride int
	data               []float32
}

// NewDense32 creates an r×c matrix backed by data (row-major). If data is nil a
// new zeroed backing slice is allocated; otherwise len(data) must be r*c.
func NewDense32(r, c int, data []float32) *Dense32 {
	if r <= 0 || c <= 0 {
		panic("mat32: non-positive dimension")
	}
	if data == nil {
		data = make([]float32, r*c)
	} else if len(data) != r*c {
		panic("mat32: data length mismatch")
	}
	return &Dense32{rows: r, cols: c, stride: c, data: data}
}

// Dims returns the number of rows and columns.
func (m *Dense32) Dims() (r, c int) { return m.rows, m.cols }

// At returns the element at row i, column j.
func (m *Dense32) At(i, j int) float32 {
	if uint(i) >= uint(m.rows) || uint(j) >= uint(m.cols) {
		panic("mat32: index out of range")
	}
	return m.data[i*m.stride+j]
}

// Set assigns v to the element at row i, column j.
func (m *Dense32) Set(i, j int, v float32) {
	if uint(i) >= uint(m.rows) || uint(j) >= uint(m.cols) {
		panic("mat32: index out of range")
	}
	m.data[i*m.stride+j] = v
}

// T returns the transpose of the matrix as a lazy view.
func (m *Dense32) T() Matrix32 { return Transpose32{m} }

// RawMatrix returns the matrix as a gonum blas32.General sharing the same
// backing array (row-major), for interop with gonum's blas32 package.
func (m *Dense32) RawMatrix() blas32.General {
	return blas32.General{Rows: m.rows, Cols: m.cols, Stride: m.stride, Data: m.data}
}

// Reset returns the matrix to its empty zero state, releasing the backing
// slice so it can be reused as a fresh result destination.
func (m *Dense32) Reset() {
	m.rows, m.cols, m.stride, m.data = 0, 0, 0, nil
}

// Zero sets every element to zero, keeping the dimensions.
func (m *Dense32) Zero() {
	for i := 0; i < m.rows; i++ {
		row := m.data[i*m.stride : i*m.stride+m.cols]
		for j := range row {
			row[j] = 0
		}
	}
}

// Clone makes m a deep, tightly-packed copy of a.
func (m *Dense32) Clone(a Matrix32) {
	r, c := a.Dims()
	m.reuseAsNonZeroed(r, c)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			m.data[i*m.stride+j] = a.At(i, j)
		}
	}
}

// Copy copies the elements of a into the overlapping top-left block of m and
// returns the number of rows and columns copied (the element-wise minimum of
// the two shapes), matching mat.Dense.Copy.
func (m *Dense32) Copy(a Matrix32) (r, c int) {
	ar, ac := a.Dims()
	r, c = min(ar, m.rows), min(ac, m.cols)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			m.data[i*m.stride+j] = a.At(i, j)
		}
	}
	return r, c
}

// reuseAsNonZeroed prepares m to receive an r×c result: it allocates a backing
// slice for an empty (zero-value) matrix, or verifies the shape of an existing
// one. The contents are not cleared (callers overwrite every element).
func (m *Dense32) reuseAsNonZeroed(r, c int) {
	if m.rows == 0 && m.cols == 0 && m.data == nil {
		m.rows, m.cols, m.stride, m.data = r, c, c, make([]float32, r*c)
		return
	}
	if m.rows != r || m.cols != c {
		panic("mat32: dimension mismatch in receiver")
	}
}
