package mat32

// Matrix32 is the basic single-precision matrix interface, mirroring
// gonum/mat.Matrix at float32.
type Matrix32 interface {
	// Dims returns the dimensions of the matrix.
	Dims() (r, c int)
	// At returns the element at row i, column j.
	At(i, j int) float32
	// T returns the transpose of the matrix.
	T() Matrix32
}

// Vector32 is a float32 vector, usable anywhere a Matrix32 is (as a column).
type Vector32 interface {
	Matrix32
	// AtVec returns the i-th element.
	AtVec(i int) float32
	// Len returns the number of elements.
	Len() int
}

// Transpose32 is a lazy transpose view of a Matrix32: it swaps the index order
// without copying. It is the float32 analogue of mat.Transpose.
type Transpose32 struct {
	Matrix Matrix32
}

// Dims returns the dimensions of the transposed matrix.
func (t Transpose32) Dims() (r, c int) {
	c, r = t.Matrix.Dims()
	return r, c
}

// At returns the element at row i, column j of the transpose, i.e. element
// (j, i) of the underlying matrix.
func (t Transpose32) At(i, j int) float32 { return t.Matrix.At(j, i) }

// T returns the underlying matrix (untransposing).
func (t Transpose32) T() Matrix32 { return t.Matrix }

// Untranspose returns the matrix the transpose wraps.
func (t Transpose32) Untranspose() Matrix32 { return t.Matrix }

// underlying returns the backing *Dense32 of a matrix that is either a *Dense32
// or a Transpose32 of one, reporting whether the value is transposed relative to
// that backing store. ok is false for any other Matrix32 implementation.
func underlying(a Matrix32) (d *Dense32, trans, ok bool) {
	switch t := a.(type) {
	case *Dense32:
		return t, false, true
	case Transpose32:
		if d, ok := t.Matrix.(*Dense32); ok {
			return d, true, true
		}
	}
	return nil, false, false
}
