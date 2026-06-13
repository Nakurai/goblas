package kernel

// This file exposes the float32 (S-prefixed) kernel methods on genericKernel.
// Each one is a thin wrapper over the element-generic free function it shares
// with its float64 (D) sibling, so the two precisions stay in lock-step by
// construction. Accelerated kernels (neonKernel, avx2Kernel) embed
// genericKernel and inherit these until they override individual routines with
// single-precision assembly.

// Generic32 returns the pure-Go float32 kernel. Like Generic, it is exported so
// tests and benchmarks can force the reference implementation regardless of CPU.
func Generic32() Kernel32 { return genericKernel{} }

// --- Level 1 ---

func (genericKernel) Sdot(n int, x []float32, incX int, y []float32, incY int) float32 {
	return ddotGeneric(n, x, incX, y, incY)
}

func (genericKernel) Saxpy(n int, alpha float32, x []float32, incX int, y []float32, incY int) {
	daxpyGeneric(n, alpha, x, incX, y, incY)
}

func (genericKernel) Sscal(n int, alpha float32, x []float32, incX int) {
	dscalGeneric(n, alpha, x, incX)
}

func (genericKernel) Snrm2(n int, x []float32, incX int) float32 {
	return dnrm2Generic(n, x, incX)
}

func (genericKernel) Sasum(n int, x []float32, incX int) float32 {
	return dasumGeneric(n, x, incX)
}

func (genericKernel) Isamax(n int, x []float32, incX int) int {
	return idamaxGeneric(n, x, incX)
}

func (genericKernel) Scopy(n int, x []float32, incX int, y []float32, incY int) {
	dcopyGeneric(n, x, incX, y, incY)
}

func (genericKernel) Sswap(n int, x []float32, incX int, y []float32, incY int) {
	dswapGeneric(n, x, incX, y, incY)
}

// --- Level 2 ---

func (genericKernel) Sgemv(trans bool, m, n int, alpha float32, a []float32, lda int, x []float32, incX int, beta float32, y []float32, incY int) {
	dgemvGeneric(trans, m, n, alpha, a, lda, x, incX, beta, y, incY)
}

func (genericKernel) Sger(m, n int, alpha float32, x []float32, incX int, y []float32, incY int, a []float32, lda int) {
	dgerGeneric(m, n, alpha, x, incX, y, incY, a, lda)
}

func (genericKernel) Strsv(upper, transA, unit bool, n int, a []float32, lda int, x []float32, incX int) {
	dtrsvGeneric(upper, transA, unit, n, a, lda, x, incX)
}

// --- Level 3 ---

func (g genericKernel) Sgemm(transA, transB bool, m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	gemmGeneric(transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

func (g genericKernel) Ssyrk(upper, trans bool, n, k int, alpha float32, a []float32, lda int, beta float32, c []float32, ldc int) {
	dsyrkRec(g.Sgemm, upper, trans, n, k, alpha, a, lda, beta, c, ldc)
}

func (g genericKernel) Strsm(left, upper, transA, unit bool, m, n int, alpha float32, a []float32, lda int, b []float32, ldb int) {
	dtrsmRec(g.Sgemm, left, upper, transA, unit, m, n, alpha, a, lda, b, ldb)
}

func (g genericKernel) Ssymm(left, upper bool, m, n int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	dsymmRec(g.Sgemm, left, upper, m, n, alpha, a, lda, b, ldb, beta, c, ldc)
}

func (g genericKernel) Strmm(left, upper, transA, unit bool, m, n int, alpha float32, a []float32, lda int, b []float32, ldb int) {
	dtrmmRec(g.Sgemm, left, upper, transA, unit, m, n, alpha, a, lda, b, ldb)
}
