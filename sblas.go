package goblas

// This file is the float32 (single-precision, S-prefixed) public API. It
// mirrors the float64 routines in blas.go exactly — same argument validation,
// same column-major semantics — delegating to the active32 kernel.

// --- Level 1 ---

// Sdot returns the dot product x·y. incX and incY are the vector increments.
func Sdot(n int, x []float32, incX int, y []float32, incY int) float32 {
	checkVector32("x", n, x, incX)
	checkVector32("y", n, y, incY)
	if n == 0 {
		return 0
	}
	return active32.Sdot(n, x, incX, y, incY)
}

// Saxpy computes y = alpha*x + y.
func Saxpy(n int, alpha float32, x []float32, incX int, y []float32, incY int) {
	checkVector32("x", n, x, incX)
	checkVector32("y", n, y, incY)
	if n == 0 {
		return
	}
	active32.Saxpy(n, alpha, x, incX, y, incY)
}

// Sscal computes x = alpha*x. A negative incX is a no-op: for single-vector
// routines the reference BLAS treats incX <= 0 as out of domain (a reversed
// traversal is only meaningful for the paired two-vector routines).
func Sscal(n int, alpha float32, x []float32, incX int) {
	if incX < 0 {
		return
	}
	checkVector32("x", n, x, incX)
	if n == 0 {
		return
	}
	active32.Sscal(n, alpha, x, incX)
}

// Snrm2 returns the Euclidean norm sqrt(x·x), computed to avoid overflow.
// A negative incX returns 0 (see Sscal on the single-vector incX contract).
func Snrm2(n int, x []float32, incX int) float32 {
	if incX < 0 {
		return 0
	}
	checkVector32("x", n, x, incX)
	if n == 0 {
		return 0
	}
	return active32.Snrm2(n, x, incX)
}

// Sasum returns the sum of the absolute values of x.
// A negative incX returns 0 (see Sscal on the single-vector incX contract).
func Sasum(n int, x []float32, incX int) float32 {
	if incX < 0 {
		return 0
	}
	checkVector32("x", n, x, incX)
	if n == 0 {
		return 0
	}
	return active32.Sasum(n, x, incX)
}

// Isamax returns the index (in vector terms, 0-based) of the element of x with
// the largest absolute value, or -1 if n == 0 or incX < 0 (see Sscal on the
// single-vector incX contract).
func Isamax(n int, x []float32, incX int) int {
	if incX < 0 {
		return -1
	}
	checkVector32("x", n, x, incX)
	if n == 0 {
		return -1
	}
	return active32.Isamax(n, x, incX)
}

// Scopy copies x into y.
func Scopy(n int, x []float32, incX int, y []float32, incY int) {
	checkVector32("x", n, x, incX)
	checkVector32("y", n, y, incY)
	if n == 0 {
		return
	}
	active32.Scopy(n, x, incX, y, incY)
}

// Sswap exchanges the contents of x and y.
func Sswap(n int, x []float32, incX int, y []float32, incY int) {
	checkVector32("x", n, x, incX)
	checkVector32("y", n, y, incY)
	if n == 0 {
		return
	}
	active32.Sswap(n, x, incX, y, incY)
}

// --- Level 2 ---

// Sgemv computes y = alpha*op(A)*x + beta*y, where op(A) = A or Aᵀ per trans.
// A is m rows by n columns, column-major with leading dimension lda >= max(1,m).
// When trans is NoTrans, x has length n and y has length m; when Trans, the
// lengths swap.
func Sgemv(trans Transpose, m, n int, alpha float32, a []float32, lda int, x []float32, incX int, beta float32, y []float32, incY int) {
	if m < 0 || n < 0 {
		panic("goblas: negative dimension")
	}
	if lda < max(1, m) {
		panic("goblas: bad lda")
	}
	lenX, lenY := n, m
	if trans == Trans {
		lenX, lenY = m, n
	}
	checkVector32("x", lenX, x, incX)
	checkVector32("y", lenY, y, incY)
	if m > 0 && n > 0 {
		checkMatrix32("a", m, n, a, lda)
	}
	if m == 0 || n == 0 {
		return
	}
	active32.Sgemv(bool(trans), m, n, alpha, a, lda, x, incX, beta, y, incY)
}

// Sger computes A = alpha*x*yᵀ + A, where A is m×n, x has length m and y has
// length n.
func Sger(m, n int, alpha float32, x []float32, incX int, y []float32, incY int, a []float32, lda int) {
	if m < 0 || n < 0 {
		panic("goblas: negative dimension")
	}
	if lda < max(1, m) {
		panic("goblas: bad lda")
	}
	checkVector32("x", m, x, incX)
	checkVector32("y", n, y, incY)
	if m == 0 || n == 0 {
		return
	}
	checkMatrix32("a", m, n, a, lda)
	active32.Sger(m, n, alpha, x, incX, y, incY, a, lda)
}

// Strsv solves op(A)*x = b for x, where A is an n×n triangular matrix and b is
// stored in x on entry; the solution overwrites x.
func Strsv(ul Uplo, transA Transpose, d Diag, n int, a []float32, lda int, x []float32, incX int) {
	if n < 0 {
		panic("goblas: negative dimension")
	}
	if lda < max(1, n) {
		panic("goblas: bad lda")
	}
	checkVector32("x", n, x, incX)
	if n == 0 {
		return
	}
	checkMatrix32("a", n, n, a, lda)
	active32.Strsv(ul == Upper, bool(transA), d == Unit, n, a, lda, x, incX)
}

// --- Level 3 ---

// Sgemm computes C = alpha*op(A)*op(B) + beta*C, where op(X) = X or Xᵀ.
// op(A) is m by k, op(B) is k by n, and C is m by n. All matrices are
// column-major; lda, ldb, ldc are the leading dimensions of A, B, C as stored.
func Sgemm(transA, transB Transpose, m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	if m < 0 || n < 0 || k < 0 {
		panic("goblas: negative dimension")
	}
	rowsA, colsA := m, k
	if transA == Trans {
		rowsA, colsA = k, m
	}
	rowsB, colsB := k, n
	if transB == Trans {
		rowsB, colsB = n, k
	}
	if lda < max(1, rowsA) {
		panic("goblas: bad lda")
	}
	if ldb < max(1, rowsB) {
		panic("goblas: bad ldb")
	}
	if ldc < max(1, m) {
		panic("goblas: bad ldc")
	}
	if m > 0 && n > 0 {
		checkMatrix32("c", m, n, c, ldc)
	}
	if m == 0 || n == 0 {
		return
	}
	if k > 0 {
		checkMatrix32("a", rowsA, colsA, a, lda)
		checkMatrix32("b", rowsB, colsB, b, ldb)
	}
	active32.Sgemm(bool(transA), bool(transB), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

// Ssyrk computes C = alpha*A*Aᵀ + beta*C (trans == NoTrans, A is n×k) or
// C = alpha*Aᵀ*A + beta*C (trans == Trans, A is k×n). C is n×n and only the
// triangle selected by ul is referenced and updated.
func Ssyrk(ul Uplo, trans Transpose, n, k int, alpha float32, a []float32, lda int, beta float32, c []float32, ldc int) {
	if n < 0 || k < 0 {
		panic("goblas: negative dimension")
	}
	rowsA, colsA := n, k
	if trans == Trans {
		rowsA, colsA = k, n
	}
	if lda < max(1, rowsA) {
		panic("goblas: bad lda")
	}
	if ldc < max(1, n) {
		panic("goblas: bad ldc")
	}
	if n == 0 {
		return
	}
	if k > 0 {
		checkMatrix32("a", rowsA, colsA, a, lda)
	}
	checkMatrix32("c", n, n, c, ldc)
	active32.Ssyrk(ul == Upper, bool(trans), n, k, alpha, a, lda, beta, c, ldc)
}

// Strsm solves op(A)*X = alpha*B (s == Left) or X*op(A) = alpha*B (s == Right)
// for X, overwriting B (m×n) with the solution. A is triangular of order m
// (Left) or n (Right).
func Strsm(s Side, ul Uplo, transA Transpose, d Diag, m, n int, alpha float32, a []float32, lda int, b []float32, ldb int) {
	if m < 0 || n < 0 {
		panic("goblas: negative dimension")
	}
	k := n
	if s == Left {
		k = m
	}
	if lda < max(1, k) {
		panic("goblas: bad lda")
	}
	if ldb < max(1, m) {
		panic("goblas: bad ldb")
	}
	if m == 0 || n == 0 {
		return
	}
	checkMatrix32("a", k, k, a, lda)
	checkMatrix32("b", m, n, b, ldb)
	active32.Strsm(s == Left, ul == Upper, bool(transA), d == Unit, m, n, alpha, a, lda, b, ldb)
}

// Ssymm computes C = alpha*A*B + beta*C (s == Left) or C = alpha*B*A + beta*C
// (s == Right), where A is symmetric with only the ul triangle stored. C and B
// are m×n; A is m×m (Left) or n×n (Right).
func Ssymm(s Side, ul Uplo, m, n int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	if m < 0 || n < 0 {
		panic("goblas: negative dimension")
	}
	k := n
	if s == Left {
		k = m
	}
	if lda < max(1, k) {
		panic("goblas: bad lda")
	}
	if ldb < max(1, m) {
		panic("goblas: bad ldb")
	}
	if ldc < max(1, m) {
		panic("goblas: bad ldc")
	}
	if m == 0 || n == 0 {
		return
	}
	checkMatrix32("a", k, k, a, lda)
	checkMatrix32("b", m, n, b, ldb)
	checkMatrix32("c", m, n, c, ldc)
	active32.Ssymm(s == Left, ul == Upper, m, n, alpha, a, lda, b, ldb, beta, c, ldc)
}

// Strmm computes B = alpha*op(A)*B (s == Left) or B = alpha*B*op(A)
// (s == Right) in place, where A is triangular. B is m×n.
func Strmm(s Side, ul Uplo, transA Transpose, d Diag, m, n int, alpha float32, a []float32, lda int, b []float32, ldb int) {
	if m < 0 || n < 0 {
		panic("goblas: negative dimension")
	}
	k := n
	if s == Left {
		k = m
	}
	if lda < max(1, k) {
		panic("goblas: bad lda")
	}
	if ldb < max(1, m) {
		panic("goblas: bad ldb")
	}
	if m == 0 || n == 0 {
		return
	}
	checkMatrix32("a", k, k, a, lda)
	checkMatrix32("b", m, n, b, ldb)
	active32.Strmm(s == Left, ul == Upper, bool(transA), d == Unit, m, n, alpha, a, lda, b, ldb)
}

// --- validation helpers (float32 mirrors of checkVector/checkMatrix) ---

func checkVector32(name string, n int, x []float32, incX int) {
	if n < 0 {
		panic("goblas: negative vector length")
	}
	if incX == 0 {
		panic("goblas: zero increment for " + name)
	}
	if n == 0 {
		return
	}
	inc := incX
	if inc < 0 {
		inc = -inc
	}
	if need := (n-1)*inc + 1; len(x) < need {
		panic("goblas: short vector " + name)
	}
}

func checkMatrix32(name string, rows, cols int, m []float32, ld int) {
	if need := (cols-1)*ld + rows; len(m) < need {
		panic("goblas: short matrix " + name)
	}
}
