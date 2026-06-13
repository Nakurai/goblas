// Package kernel defines the numerical primitives behind goblas and the
// mechanism for selecting an implementation at runtime.
//
// The contract is a single Kernel interface implemented by genericKernel, a
// portable pure-Go reference that is always correct and always available.
// Accelerated implementations (e.g. ARM64 NEON) embed genericKernel and
// override only the routines they have hand-tuned assembly for, so partial
// coverage falls back to the reference automatically.
//
// All matrices are column-major: element A(i,j) lives at a[i+j*lda], where lda
// (the "leading dimension") is the column stride and must be >= the row count.
package kernel

// Kernel is the set of BLAS primitives goblas dispatches to. Implementations
// must be safe for concurrent use: they are stateless and operate only on the
// slices passed in.
type Kernel interface {
	// --- Level 1: vector-vector ---

	// Ddot returns the dot product of x and y.
	Ddot(n int, x []float64, incX int, y []float64, incY int) float64
	// Daxpy computes y += alpha*x.
	Daxpy(n int, alpha float64, x []float64, incX int, y []float64, incY int)
	// Dscal computes x *= alpha.
	Dscal(n int, alpha float64, x []float64, incX int)
	// Dnrm2 returns the Euclidean norm of x.
	Dnrm2(n int, x []float64, incX int) float64
	// Dasum returns the sum of the absolute values of x.
	Dasum(n int, x []float64, incX int) float64
	// Idamax returns the index of the element of x with the largest absolute
	// value. It returns -1 if n == 0.
	Idamax(n int, x []float64, incX int) int
	// Dcopy copies x into y.
	Dcopy(n int, x []float64, incX int, y []float64, incY int)
	// Dswap exchanges the contents of x and y.
	Dswap(n int, x []float64, incX int, y []float64, incY int)

	// --- Level 2: matrix-vector (column-major) ---

	// Dgemv computes y = alpha*op(A)*x + beta*y, where op(A) = A if trans is
	// false and A^T if trans is true. A is m rows by n columns with leading
	// dimension lda.
	Dgemv(trans bool, m, n int, alpha float64, a []float64, lda int, x []float64, incX int, beta float64, y []float64, incY int)
	// Dger computes A += alpha*x*y^T, where A is m by n.
	Dger(m, n int, alpha float64, x []float64, incX int, y []float64, incY int, a []float64, lda int)
	// Dtrsv solves op(A)*x = b in place (x holds b on entry, the solution on
	// return), where A is an n by n triangular matrix. upper selects the
	// stored triangle; unit means the diagonal is implicitly 1.
	Dtrsv(upper, transA, unit bool, n int, a []float64, lda int, x []float64, incX int)

	// --- Level 3: matrix-matrix (column-major) ---

	// Dgemm computes C = alpha*op(A)*op(B) + beta*C, where op(X) = X or X^T.
	// op(A) is m by k, op(B) is k by n, and C is m by n.
	Dgemm(transA, transB bool, m, n, k int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int)
	// Dsyrk computes C = alpha*A*A^T + beta*C (trans false, A is n by k) or
	// C = alpha*A^T*A + beta*C (trans true, A is k by n). Only the triangle
	// of C selected by upper is referenced and updated.
	Dsyrk(upper, trans bool, n, k int, alpha float64, a []float64, lda int, beta float64, c []float64, ldc int)
	// Dtrsm solves op(A)*X = alpha*B (left true) or X*op(A) = alpha*B (left
	// false) for X, overwriting B (m by n). A is triangular of order m (left)
	// or n (right); unit means an implicit unit diagonal.
	Dtrsm(left, upper, transA, unit bool, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int)
	// Dsymm computes C = alpha*A*B + beta*C (left true) or
	// C = alpha*B*A + beta*C (left false), where A is symmetric with only the
	// triangle selected by upper stored. C is m by n.
	Dsymm(left, upper bool, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int)
	// Dtrmm computes B = alpha*op(A)*B (left true) or B = alpha*B*op(A)
	// (left false) in place, where A is triangular. B is m by n.
	Dtrmm(left, upper, transA, unit bool, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int)
}

// Kernel32 is the float32 (single-precision, S-prefixed) counterpart of Kernel.
// It mirrors Kernel routine-for-routine; the two are the same algorithms
// instantiated at the two element types (see float.go). Implementations must be
// safe for concurrent use.
type Kernel32 interface {
	// --- Level 1: vector-vector ---

	Sdot(n int, x []float32, incX int, y []float32, incY int) float32
	Saxpy(n int, alpha float32, x []float32, incX int, y []float32, incY int)
	Sscal(n int, alpha float32, x []float32, incX int)
	Snrm2(n int, x []float32, incX int) float32
	Sasum(n int, x []float32, incX int) float32
	Isamax(n int, x []float32, incX int) int
	Scopy(n int, x []float32, incX int, y []float32, incY int)
	Sswap(n int, x []float32, incX int, y []float32, incY int)

	// --- Level 2: matrix-vector (column-major) ---

	Sgemv(trans bool, m, n int, alpha float32, a []float32, lda int, x []float32, incX int, beta float32, y []float32, incY int)
	Sger(m, n int, alpha float32, x []float32, incX int, y []float32, incY int, a []float32, lda int)
	Strsv(upper, transA, unit bool, n int, a []float32, lda int, x []float32, incX int)

	// --- Level 3: matrix-matrix (column-major) ---

	Sgemm(transA, transB bool, m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int)
	Ssyrk(upper, trans bool, n, k int, alpha float32, a []float32, lda int, beta float32, c []float32, ldc int)
	Strsm(left, upper, transA, unit bool, m, n int, alpha float32, a []float32, lda int, b []float32, ldb int)
	Ssymm(left, upper bool, m, n int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int)
	Strmm(left, upper, transA, unit bool, m, n int, alpha float32, a []float32, lda int, b []float32, ldb int)
}
