package goblas

// Transpose specifies whether a matrix operand is used as-is or transposed.
type Transpose bool

const (
	NoTrans Transpose = false // use op(A) = A
	Trans   Transpose = true  // use op(A) = Aᵀ
)

// Uplo specifies which triangle of a triangular or symmetric matrix is stored.
type Uplo bool

const (
	Upper Uplo = true  // the upper triangle is stored
	Lower Uplo = false // the lower triangle is stored
)

// Side specifies whether a matrix appears on the left or right of the operand.
type Side bool

const (
	Left  Side = true  // op(A)*X or A*B
	Right Side = false // X*op(A) or B*A
)

// Diag specifies whether a triangular matrix has an implicit unit diagonal.
type Diag bool

const (
	NonUnit Diag = false // the diagonal is stored
	Unit    Diag = true  // the diagonal is implicitly all ones
)

// --- Level 1 ---

// Ddot returns the dot product x·y. incX and incY are the vector increments.
func Ddot(n int, x []float64, incX int, y []float64, incY int) float64 {
	checkVector("x", n, x, incX)
	checkVector("y", n, y, incY)
	if n == 0 {
		return 0
	}
	return active.Ddot(n, x, incX, y, incY)
}

// Daxpy computes y = alpha*x + y.
func Daxpy(n int, alpha float64, x []float64, incX int, y []float64, incY int) {
	checkVector("x", n, x, incX)
	checkVector("y", n, y, incY)
	if n == 0 {
		return
	}
	active.Daxpy(n, alpha, x, incX, y, incY)
}

// Dscal computes x = alpha*x. A negative incX is a no-op: for single-vector
// routines the reference BLAS treats incX <= 0 as out of domain (a reversed
// traversal is only meaningful for the paired two-vector routines).
func Dscal(n int, alpha float64, x []float64, incX int) {
	if incX < 0 {
		return
	}
	checkVector("x", n, x, incX)
	if n == 0 {
		return
	}
	active.Dscal(n, alpha, x, incX)
}

// Dnrm2 returns the Euclidean norm sqrt(x·x), computed to avoid overflow.
// A negative incX returns 0 (see Dscal on the single-vector incX contract).
func Dnrm2(n int, x []float64, incX int) float64 {
	if incX < 0 {
		return 0
	}
	checkVector("x", n, x, incX)
	if n == 0 {
		return 0
	}
	return active.Dnrm2(n, x, incX)
}

// Dasum returns the sum of the absolute values of x.
// A negative incX returns 0 (see Dscal on the single-vector incX contract).
func Dasum(n int, x []float64, incX int) float64 {
	if incX < 0 {
		return 0
	}
	checkVector("x", n, x, incX)
	if n == 0 {
		return 0
	}
	return active.Dasum(n, x, incX)
}

// Idamax returns the index (in vector terms, 0-based) of the element of x with
// the largest absolute value, or -1 if n == 0 or incX < 0 (see Dscal on the
// single-vector incX contract).
func Idamax(n int, x []float64, incX int) int {
	if incX < 0 {
		return -1
	}
	checkVector("x", n, x, incX)
	if n == 0 {
		return -1
	}
	return active.Idamax(n, x, incX)
}

// Dcopy copies x into y.
func Dcopy(n int, x []float64, incX int, y []float64, incY int) {
	checkVector("x", n, x, incX)
	checkVector("y", n, y, incY)
	if n == 0 {
		return
	}
	active.Dcopy(n, x, incX, y, incY)
}

// Dswap exchanges the contents of x and y.
func Dswap(n int, x []float64, incX int, y []float64, incY int) {
	checkVector("x", n, x, incX)
	checkVector("y", n, y, incY)
	if n == 0 {
		return
	}
	active.Dswap(n, x, incX, y, incY)
}

// --- Level 2 ---

// Dgemv computes y = alpha*op(A)*x + beta*y, where op(A) = A or Aᵀ per trans.
// A is m rows by n columns, column-major with leading dimension lda >= max(1,m).
// When trans is NoTrans, x has length n and y has length m; when Trans, the
// lengths swap.
func Dgemv(trans Transpose, m, n int, alpha float64, a []float64, lda int, x []float64, incX int, beta float64, y []float64, incY int) {
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
	checkVector("x", lenX, x, incX)
	checkVector("y", lenY, y, incY)
	if m > 0 && n > 0 {
		checkMatrix("a", m, n, a, lda)
	}
	if m == 0 || n == 0 {
		return
	}
	active.Dgemv(bool(trans), m, n, alpha, a, lda, x, incX, beta, y, incY)
}

// Dger computes A = alpha*x*yᵀ + A, where A is m×n, x has length m and y has
// length n.
func Dger(m, n int, alpha float64, x []float64, incX int, y []float64, incY int, a []float64, lda int) {
	if m < 0 || n < 0 {
		panic("goblas: negative dimension")
	}
	if lda < max(1, m) {
		panic("goblas: bad lda")
	}
	checkVector("x", m, x, incX)
	checkVector("y", n, y, incY)
	if m == 0 || n == 0 {
		return
	}
	checkMatrix("a", m, n, a, lda)
	active.Dger(m, n, alpha, x, incX, y, incY, a, lda)
}

// Dtrsv solves op(A)*x = b for x, where A is an n×n triangular matrix and b is
// stored in x on entry; the solution overwrites x.
func Dtrsv(ul Uplo, transA Transpose, d Diag, n int, a []float64, lda int, x []float64, incX int) {
	if n < 0 {
		panic("goblas: negative dimension")
	}
	if lda < max(1, n) {
		panic("goblas: bad lda")
	}
	checkVector("x", n, x, incX)
	if n == 0 {
		return
	}
	checkMatrix("a", n, n, a, lda)
	active.Dtrsv(ul == Upper, bool(transA), d == Unit, n, a, lda, x, incX)
}

// --- Level 3 ---

// Dgemm computes C = alpha*op(A)*op(B) + beta*C, where op(X) = X or Xᵀ.
// op(A) is m by k, op(B) is k by n, and C is m by n. All matrices are
// column-major; lda, ldb, ldc are the leading dimensions of A, B, C as stored.
func Dgemm(transA, transB Transpose, m, n, k int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int) {
	if m < 0 || n < 0 || k < 0 {
		panic("goblas: negative dimension")
	}
	// Stored (pre-transpose) row counts of A and B.
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
		checkMatrix("c", m, n, c, ldc)
	}
	if m == 0 || n == 0 {
		return
	}
	if k > 0 {
		checkMatrix("a", rowsA, colsA, a, lda)
		checkMatrix("b", rowsB, colsB, b, ldb)
	}
	active.Dgemm(bool(transA), bool(transB), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

// Dsyrk computes C = alpha*A*Aᵀ + beta*C (trans == NoTrans, A is n×k) or
// C = alpha*Aᵀ*A + beta*C (trans == Trans, A is k×n). C is n×n and only the
// triangle selected by ul is referenced and updated.
func Dsyrk(ul Uplo, trans Transpose, n, k int, alpha float64, a []float64, lda int, beta float64, c []float64, ldc int) {
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
		checkMatrix("a", rowsA, colsA, a, lda)
	}
	checkMatrix("c", n, n, c, ldc)
	active.Dsyrk(ul == Upper, bool(trans), n, k, alpha, a, lda, beta, c, ldc)
}

// Dtrsm solves op(A)*X = alpha*B (s == Left) or X*op(A) = alpha*B (s == Right)
// for X, overwriting B (m×n) with the solution. A is triangular of order m
// (Left) or n (Right).
func Dtrsm(s Side, ul Uplo, transA Transpose, d Diag, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int) {
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
	checkMatrix("a", k, k, a, lda)
	checkMatrix("b", m, n, b, ldb)
	active.Dtrsm(s == Left, ul == Upper, bool(transA), d == Unit, m, n, alpha, a, lda, b, ldb)
}

// Dsymm computes C = alpha*A*B + beta*C (s == Left) or C = alpha*B*A + beta*C
// (s == Right), where A is symmetric with only the ul triangle stored. C and B
// are m×n; A is m×m (Left) or n×n (Right).
func Dsymm(s Side, ul Uplo, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int) {
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
	checkMatrix("a", k, k, a, lda)
	checkMatrix("b", m, n, b, ldb)
	checkMatrix("c", m, n, c, ldc)
	active.Dsymm(s == Left, ul == Upper, m, n, alpha, a, lda, b, ldb, beta, c, ldc)
}

// Dtrmm computes B = alpha*op(A)*B (s == Left) or B = alpha*B*op(A)
// (s == Right) in place, where A is triangular. B is m×n.
func Dtrmm(s Side, ul Uplo, transA Transpose, d Diag, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int) {
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
	checkMatrix("a", k, k, a, lda)
	checkMatrix("b", m, n, b, ldb)
	active.Dtrmm(s == Left, ul == Upper, bool(transA), d == Unit, m, n, alpha, a, lda, b, ldb)
}

// --- validation helpers ---

// checkVector panics if incX is zero or x is too short to hold n elements at
// the given increment.
func checkVector(name string, n int, x []float64, incX int) {
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

// checkMatrix panics if a column-major rows×cols matrix does not fit in m.
func checkMatrix(name string, rows, cols int, m []float64, ld int) {
	if need := (cols-1)*ld + rows; len(m) < need {
		panic("goblas: short matrix " + name)
	}
}
