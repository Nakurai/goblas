package mat32

import "gonum.org/v1/gonum/mat"

// QR32, SVD32, Eigen32 and EigenSym32 wrap gonum's float64 factorizations.
// IMPORTANT: these cast to float64 internally — they are NOT end-to-end float32
// (gonum has no float32 LAPACK). For the no-cast float32 solves use Cholesky32
// or LU32. Register blasadapt.Use() so the bridge's float64 BLAS runs on goblas.

// QR32 is the QR factorization A = Q·R (via the float64 bridge).
type QR32 struct{ qr mat.QR }

// Factorize computes the QR factorization of a (m ≥ n).
func (q *QR32) Factorize(a Matrix32) { q.qr.Factorize(toDense64(a)) }

// QTo writes the orthogonal matrix Q into dst.
func (q *QR32) QTo(dst *Dense32) {
	var m mat.Dense
	q.qr.QTo(&m)
	setFrom64(dst, &m)
}

// RTo writes the upper-triangular matrix R into dst.
func (q *QR32) RTo(dst *Dense32) {
	var m mat.Dense
	q.qr.RTo(&m)
	setFrom64(dst, &m)
}

// SolveTo solves the least-squares (or, with trans, the related) system and
// writes the result into dst.
func (q *QR32) SolveTo(dst *Dense32, trans bool, b Matrix32) error {
	var m mat.Dense
	if err := q.qr.SolveTo(&m, trans, toDense64(b)); err != nil {
		return err
	}
	setFrom64(dst, &m)
	return nil
}

// SVD32 is the singular value decomposition (via the float64 bridge). The kind
// argument is gonum's mat.SVDKind (e.g. mat.SVDThin, mat.SVDFull).
type SVD32 struct{ svd mat.SVD }

// Factorize computes the SVD of a, returning false if it did not converge.
func (s *SVD32) Factorize(a Matrix32, kind mat.SVDKind) (ok bool) {
	return s.svd.Factorize(toDense64(a), kind)
}

// Values returns the singular values in descending order. If dst is non-nil it
// is filled and returned.
func (s *SVD32) Values(dst []float32) []float32 {
	v := s.svd.Values(nil)
	if dst == nil {
		dst = make([]float32, len(v))
	}
	for i := range v {
		dst[i] = float32(v[i])
	}
	return dst
}

// UTo writes the left singular vectors into dst.
func (s *SVD32) UTo(dst *Dense32) {
	var m mat.Dense
	s.svd.UTo(&m)
	setFrom64(dst, &m)
}

// VTo writes the right singular vectors into dst.
func (s *SVD32) VTo(dst *Dense32) {
	var m mat.Dense
	s.svd.VTo(&m)
	setFrom64(dst, &m)
}

// EigenSym32 is the eigendecomposition of a symmetric matrix (real eigenvalues
// and eigenvectors), via the float64 bridge.
type EigenSym32 struct{ e mat.EigenSym }

// Factorize computes the eigendecomposition of the symmetric matrix a; pass
// vectors=true to also compute the eigenvectors.
func (e *EigenSym32) Factorize(a *SymDense32, vectors bool) (ok bool) {
	return e.e.Factorize(toSym64(a), vectors)
}

// Values returns the eigenvalues in ascending order.
func (e *EigenSym32) Values(dst []float32) []float32 {
	v := e.e.Values(nil)
	if dst == nil {
		dst = make([]float32, len(v))
	}
	for i := range v {
		dst[i] = float32(v[i])
	}
	return dst
}

// VectorsTo writes the eigenvectors (as columns) into dst.
func (e *EigenSym32) VectorsTo(dst *Dense32) {
	var m mat.Dense
	e.e.VectorsTo(&m)
	setFrom64(dst, &m)
}

// Eigen32 is the eigendecomposition of a general (possibly non-symmetric)
// matrix, via the float64 bridge. Eigenvalues are complex; vectors are not
// exposed here (use the symmetric path, or the float64 bridge directly, when
// eigenvectors are needed).
type Eigen32 struct{ e mat.Eigen }

// Factorize computes the eigenvalues (and, per kind, vectors internally) of a.
func (e *Eigen32) Factorize(a Matrix32, kind mat.EigenKind) (ok bool) {
	return e.e.Factorize(toDense64(a), kind)
}

// Values returns the complex eigenvalues as complex64.
func (e *Eigen32) Values(dst []complex64) []complex64 {
	v := e.e.Values(nil)
	if dst == nil {
		dst = make([]complex64, len(v))
	}
	for i := range v {
		dst[i] = complex64(v[i])
	}
	return dst
}
