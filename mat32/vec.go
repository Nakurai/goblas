package mat32

// VecDense32 is a dense float32 column vector. As a Matrix32 it has dimensions
// (n, 1). It carries an increment so it can view strided data, mirroring
// mat.VecDense.
type VecDense32 struct {
	n    int
	inc  int
	data []float32
}

// NewVecDense32 creates a length-n column vector backed by data. If data is nil
// a new zeroed slice is allocated; otherwise len(data) must be n.
func NewVecDense32(n int, data []float32) *VecDense32 {
	if n <= 0 {
		panic("mat32: non-positive dimension")
	}
	if data == nil {
		data = make([]float32, n)
	} else if len(data) != n {
		panic("mat32: data length mismatch")
	}
	return &VecDense32{n: n, inc: 1, data: data}
}

// Dims returns (n, 1).
func (v *VecDense32) Dims() (r, c int) { return v.n, 1 }

// At returns the i-th element; j must be 0.
func (v *VecDense32) At(i, j int) float32 {
	if j != 0 {
		panic("mat32: column index out of range")
	}
	return v.AtVec(i)
}

// AtVec returns the i-th element.
func (v *VecDense32) AtVec(i int) float32 {
	if uint(i) >= uint(v.n) {
		panic("mat32: index out of range")
	}
	return v.data[i*v.inc]
}

// SetVec assigns the i-th element.
func (v *VecDense32) SetVec(i int, val float32) {
	if uint(i) >= uint(v.n) {
		panic("mat32: index out of range")
	}
	v.data[i*v.inc] = val
}

// Len returns the number of elements.
func (v *VecDense32) Len() int { return v.n }

// T returns the vector as a 1×n row (a lazy transpose view).
func (v *VecDense32) T() Matrix32 { return Transpose32{v} }

// RawVector returns the backing slice and increment.
func (v *VecDense32) rawVector() (data []float32, inc int) { return v.data, v.inc }

// reuseAsNonZeroed prepares v to receive a length-n result.
func (v *VecDense32) reuseAsNonZeroed(n int) {
	if v.n == 0 && v.data == nil {
		v.n, v.inc, v.data = n, 1, make([]float32, n)
		return
	}
	if v.n != n {
		panic("mat32: dimension mismatch in vector receiver")
	}
}
