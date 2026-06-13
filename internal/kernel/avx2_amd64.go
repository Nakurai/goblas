//go:build amd64

package kernel

import "github.com/nakurai/goblas/internal/cpu"

// dgemmKernel8x4AVX2 computes C[8x4] += Apanel * Bpanel for packed
// micro-panels using AVX2+FMA (4 float64 per YMM register). Implemented in
// dgemm_amd64.s.
//
//go:noescape
func dgemmKernel8x4AVX2(k int, a, b, c *float64, ldc int)

// avx2Kernel is the x86-64 AVX2-accelerated kernel. It embeds genericKernel
// and overrides only the routines with assembly, mirroring neonKernel on
// ARM64.
type avx2Kernel struct {
	genericKernel
}

func (ak avx2Kernel) Dgemm(transA, transB bool, m, n, k int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int) {
	dgemmBlocked(dgemmKernel8x4AVX2, dgemmNR, transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

// Dsyrk and Dtrsm reuse the shared recursive blocking fed with the AVX2 Dgemm
// so the gemm-shaped bulk runs on assembly.

func (ak avx2Kernel) Dsyrk(upper, trans bool, n, k int, alpha float64, a []float64, lda int, beta float64, c []float64, ldc int) {
	dsyrkRec(ak.Dgemm, upper, trans, n, k, alpha, a, lda, beta, c, ldc)
}

func (ak avx2Kernel) Dtrsm(left, upper, transA, unit bool, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int) {
	dtrsmRec(ak.Dgemm, left, upper, transA, unit, m, n, alpha, a, lda, b, ldb)
}

// Select routes AVX2+FMA-capable x86-64 hosts to the AVX2 kernel; anything
// older (pre-2013) gets the portable reference. Blocking is conservative for
// the typical 32 KB x86 L1d: a packed A block is mc*kc*8 = 16*256*8 = 32 KB.
func Select(c cpu.CPU) Kernel {
	if !c.HasAVX2FMA {
		return genericKernel{}
	}
	dgemmKC, dgemmMC = 256, 16
	if c.L1DBytes > 0 {
		dgemmMC = max(dgemmMR, c.L1DBytes/(dgemmKC*8)/dgemmMR*dgemmMR)
	}
	return avx2Kernel{}
}
