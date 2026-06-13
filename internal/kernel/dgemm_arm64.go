package kernel

// dgemmKernel8x4 computes C[8x4] += Apanel * Bpanel for packed micro-panels.
// Implemented in dgemm_arm64.s.
//
//go:noescape
func dgemmKernel8x4(k int, a, b, c *float64, ldc int)

// dgemmKernel8x6 computes C[8x6] += Apanel * Bpanel for packed micro-panels
// (B packed 6-wide). Implemented in dgemm8x6_arm64.s.
//
//go:noescape
func dgemmKernel8x6(k int, a, b, c *float64, ldc int)

// neonMicroKernel/neonNR select which NEON micro-kernel the blocked driver
// runs. The 8x6 shape reuses each A load across 6 columns instead of 4
// (higher FMA density per load); the Phase 12 sweep picks the default.
// Variables so the tuning benchmarks can swap them.
var (
	neonMicroKernel microKernel = dgemmKernel8x6
	neonNR                      = 6
)

// Dgemm overrides the generic matrix multiply with the shared blocked driver
// running the NEON micro-kernel.
func (nk neonKernel) Dgemm(transA, transB bool, m, n, k int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int) {
	dgemmBlocked(neonMicroKernel, neonNR, transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}
