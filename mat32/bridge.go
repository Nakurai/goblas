package mat32

import "gonum.org/v1/gonum/mat"

// This file holds the float64 bridge used by the advanced factorizations
// (QR/SVD/Eigen), which gonum implements only in float64. These conversions
// DO cast between float32 and float64 — unlike the rest of mat32, the QR/SVD/
// Eigen paths are not end-to-end float32. For an accelerated bridge, register
// goblas as the float64 BLAS with blasadapt.Use() so gonum's LAPACK runs on
// goblas kernels.

// toDense64 returns a float64 gonum Dense copy of a (casting each element).
func toDense64(a Matrix32) *mat.Dense {
	r, c := a.Dims()
	d := make([]float64, r*c)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			d[i*c+j] = float64(a.At(i, j))
		}
	}
	return mat.NewDense(r, c, d)
}

// toSym64 returns a float64 gonum SymDense copy of a.
func toSym64(a *SymDense32) *mat.SymDense {
	n := a.n
	d := make([]float64, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			d[i*n+j] = float64(a.At(i, j))
		}
	}
	return mat.NewSymDense(n, d)
}

// setFrom64 fills dst with the float32-cast contents of the gonum matrix src.
func setFrom64(dst *Dense32, src mat.Matrix) {
	r, c := src.Dims()
	dst.reuseAsNonZeroed(r, c)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			dst.data[i*dst.stride+j] = float32(src.At(i, j))
		}
	}
}
