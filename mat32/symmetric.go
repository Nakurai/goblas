package mat32

// SymDense32 is a symmetric float32 matrix with full row-major storage (both
// triangles held, kept equal). It is the input type for Cholesky32.
type SymDense32 struct {
	n      int
	stride int
	data   []float32
}

// NewSymDense32 creates an n×n symmetric matrix. If data is nil a zero matrix is
// allocated; otherwise len(data) must be n*n (row-major) and is assumed
// symmetric (only the values read by the factorization matter).
func NewSymDense32(n int, data []float32) *SymDense32 {
	if n <= 0 {
		panic("mat32: non-positive dimension")
	}
	if data == nil {
		data = make([]float32, n*n)
	} else if len(data) != n*n {
		panic("mat32: data length mismatch")
	}
	return &SymDense32{n: n, stride: n, data: data}
}

// SymDense32FromDense builds a symmetric matrix from the given triangle of a
// square matrix a, mirroring it into the other triangle. If upper is true the
// upper triangle of a is used, else the lower.
func SymDense32FromDense(a Matrix32, upper bool) *SymDense32 {
	n, c := a.Dims()
	if n != c {
		panic("mat32: matrix not square")
	}
	s := NewSymDense32(n, nil)
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			var v float32
			if upper {
				v = a.At(i, j)
			} else {
				v = a.At(j, i)
			}
			s.data[i*s.stride+j] = v
			s.data[j*s.stride+i] = v
		}
	}
	return s
}

// Dims returns (n, n).
func (s *SymDense32) Dims() (r, c int) { return s.n, s.n }

// Symmetric returns the order n of the matrix.
func (s *SymDense32) Symmetric() int { return s.n }

// At returns the element at row i, column j.
func (s *SymDense32) At(i, j int) float32 {
	if uint(i) >= uint(s.n) || uint(j) >= uint(s.n) {
		panic("mat32: index out of range")
	}
	return s.data[i*s.stride+j]
}

// SetSym sets the (i,j) and (j,i) elements to v, keeping symmetry.
func (s *SymDense32) SetSym(i, j int, v float32) {
	s.data[i*s.stride+j] = v
	s.data[j*s.stride+i] = v
}

// T returns the receiver (a symmetric matrix is its own transpose).
func (s *SymDense32) T() Matrix32 { return s }

// colMajor returns a fresh column-major copy of a (ld = rows). The conversion
// is a layout transpose only — still float32, no value casting.
func colMajor(a Matrix32) (buf []float32, n, ld int) {
	r, c := a.Dims()
	if r != c {
		panic("mat32: matrix not square")
	}
	n, ld = r, r
	buf = make([]float32, n*n)
	for j := 0; j < n; j++ {
		for i := 0; i < n; i++ {
			buf[i+j*ld] = a.At(i, j)
		}
	}
	return buf, n, ld
}

// colMajorRHS returns a fresh column-major copy of an n×nrhs right-hand side.
func colMajorRHS(b Matrix32) (buf []float32, n, nrhs, ld int) {
	n, nrhs = b.Dims()
	ld = n
	buf = make([]float32, n*nrhs)
	for j := 0; j < nrhs; j++ {
		for i := 0; i < n; i++ {
			buf[i+j*ld] = b.At(i, j)
		}
	}
	return buf, n, nrhs, ld
}

// denseFromColMajor builds a row-major Dense32 (n×c) from a column-major buffer.
func denseFromColMajor(buf []float32, n, c, ld int) *Dense32 {
	d := NewDense32(n, c, nil)
	for j := 0; j < c; j++ {
		for i := 0; i < n; i++ {
			d.data[i*d.stride+j] = buf[i+j*ld]
		}
	}
	return d
}
