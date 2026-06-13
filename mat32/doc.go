// Package mat32 is a single-precision (float32) dense matrix type built on the
// goblas S-kernels. It exists because gonum offers no high-level float32 linear
// algebra (there is no lapack32 and gonum/mat is float64-only), so float32 data
// — sensor streams, ML activations — would otherwise have to be cast to float64
// to get matrix operations.
//
// # Precision boundary
//
// Dense32 arithmetic (Mul/MulVec/Add/Scale/…) and the LU and Cholesky solves
// are end-to-end float32: no float64 casting, preserving the memory/bandwidth
// advantage of single precision. The trailing FLOPs run on the goblas Sgemm/
// Strsm/Ssyrk/Sgemv kernels.
//
// The advanced factorizations QR32, SVD32, Eigen32 and EigenSym32 are computed
// via a float64 bridge to gonum/mat (gonum has no float32 LAPACK): they cast to
// float64 internally and are NOT end-to-end float32. To make the bridge's
// float64 BLAS itself goblas-accelerated, register goblas with
// blasadapt.Use() at startup.
//
// # Conventions and caveats
//
// Matrices are stored row-major (element (i,j) at data[i*stride+j]), matching
// gonum/mat and gonum's blas32.General. The Matrix32 interface mirrors
// gonum/mat.Matrix at float32.
//
// Single precision carries ~7 significant digits, so results — especially from
// the factorizations on ill-conditioned inputs — are less accurate than
// float64. In particular, Det overflows to ±Inf for even moderately sized
// matrices; prefer the solves over forming determinants.
package mat32
