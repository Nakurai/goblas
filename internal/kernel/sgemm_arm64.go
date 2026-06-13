package kernel

// sgemmKernel8x8 computes C[8x8] += Apanel * Bpanel for packed micro-panels
// (A packed 8-wide, B packed 8-wide), in single precision. Implemented in
// sgemm8x8_arm64.s.
//
//go:noescape
func sgemmKernel8x8(k int, a, b, c *float32, ldc int)

// sgemmKernel8x12 computes C[8x12] += Apanel * Bpanel (B packed 12-wide). It
// reuses each A load across 12 columns (24 accumulators) for higher FMA density
// than the 8x8 kernel. Implemented in sgemm8x12_arm64.s.
//
//go:noescape
func sgemmKernel8x12(k int, a, b, c *float32, ldc int)

// neonSMicroKernel/neonSNR select the float32 NEON micro-kernel for the shared
// blocked driver. The 8x8 kernel is the default: the wider 8x12 kernel (24
// accumulators, like the float64 8x6) was measured tied-to-slightly-slower
// (BenchmarkSgemmKernelSweep), because float32 GEMM is already memory/cache-
// bandwidth bound at these sizes, so more FMA density per A load buys nothing.
// Both are kept; variables so the tuning benchmark can swap them.
var (
	neonSMicroKernel microKernel[float32] = sgemmKernel8x8
	neonSNR                               = 8
)

// Sgemm overrides the generic single-precision matrix multiply with the shared
// blocked driver running the NEON micro-kernel.
func (nk neonKernel) Sgemm(transA, transB bool, m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	gemmBlocked(neonSMicroKernel, neonSNR, transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

// Ssyrk/Strsm/Ssymm/Strmm route through the shared recursive blocking fed with
// the NEON Sgemm, so the gemm-shaped bulk runs on assembly.

func (nk neonKernel) Ssyrk(upper, trans bool, n, k int, alpha float32, a []float32, lda int, beta float32, c []float32, ldc int) {
	dsyrkRec(nk.Sgemm, upper, trans, n, k, alpha, a, lda, beta, c, ldc)
}

func (nk neonKernel) Strsm(left, upper, transA, unit bool, m, n int, alpha float32, a []float32, lda int, b []float32, ldb int) {
	dtrsmRec(nk.Sgemm, left, upper, transA, unit, m, n, alpha, a, lda, b, ldb)
}

func (nk neonKernel) Ssymm(left, upper bool, m, n int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	dsymmRec(nk.Sgemm, left, upper, m, n, alpha, a, lda, b, ldb, beta, c, ldc)
}

func (nk neonKernel) Strmm(left, upper, transA, unit bool, m, n int, alpha float32, a []float32, lda int, b []float32, ldb int) {
	dtrmmRec(nk.Sgemm, left, upper, transA, unit, m, n, alpha, a, lda, b, ldb)
}
