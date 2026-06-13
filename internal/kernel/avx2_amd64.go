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
	gemmBlocked(microKernel[float64](dgemmKernel8x4AVX2), dgemmNR, transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

// Dsyrk and Dtrsm reuse the shared recursive blocking fed with the AVX2 Dgemm
// so the gemm-shaped bulk runs on assembly.

func (ak avx2Kernel) Dsyrk(upper, trans bool, n, k int, alpha float64, a []float64, lda int, beta float64, c []float64, ldc int) {
	dsyrkRec(ak.Dgemm, upper, trans, n, k, alpha, a, lda, beta, c, ldc)
}

func (ak avx2Kernel) Dtrsm(left, upper, transA, unit bool, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int) {
	dtrsmRec(ak.Dgemm, left, upper, transA, unit, m, n, alpha, a, lda, b, ldb)
}

func (ak avx2Kernel) Dsymm(left, upper bool, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int) {
	dsymmRec(ak.Dgemm, left, upper, m, n, alpha, a, lda, b, ldb, beta, c, ldc)
}

func (ak avx2Kernel) Dtrmm(left, upper, transA, unit bool, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int) {
	dtrmmRec(ak.Dgemm, left, upper, transA, unit, m, n, alpha, a, lda, b, ldb)
}

// --- float32 (single precision) AVX2 ---

// sgemmKernel8x8AVX2 computes C[8x8] += Apanel * Bpanel for packed micro-panels
// (A and B packed 8-wide) in single precision using AVX2+FMA (8 float32 per YMM
// register). Implemented in sgemm_amd64.s.
//
//go:noescape
func sgemmKernel8x8AVX2(k int, a, b, c *float32, ldc int)

var (
	avx2SMicroKernel microKernel[float32] = sgemmKernel8x8AVX2
	avx2SNR                               = 8
)

func (ak avx2Kernel) Sgemm(transA, transB bool, m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	gemmBlocked(avx2SMicroKernel, avx2SNR, transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

func (ak avx2Kernel) Ssyrk(upper, trans bool, n, k int, alpha float32, a []float32, lda int, beta float32, c []float32, ldc int) {
	dsyrkRec(ak.Sgemm, upper, trans, n, k, alpha, a, lda, beta, c, ldc)
}

func (ak avx2Kernel) Strsm(left, upper, transA, unit bool, m, n int, alpha float32, a []float32, lda int, b []float32, ldb int) {
	dtrsmRec(ak.Sgemm, left, upper, transA, unit, m, n, alpha, a, lda, b, ldb)
}

func (ak avx2Kernel) Ssymm(left, upper bool, m, n int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	dsymmRec(ak.Sgemm, left, upper, m, n, alpha, a, lda, b, ldb, beta, c, ldc)
}

func (ak avx2Kernel) Strmm(left, upper, transA, unit bool, m, n int, alpha float32, a []float32, lda int, b []float32, ldb int) {
	dtrmmRec(ak.Sgemm, left, upper, transA, unit, m, n, alpha, a, lda, b, ldb)
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

// Select32 mirrors Select for the float32 kernel; avx2Kernel currently serves
// float32 through the embedded pure-Go reference until single-precision AVX2
// assembly lands.
func Select32(c cpu.CPU) Kernel32 {
	if !c.HasAVX2FMA {
		return genericKernel{}
	}
	return avx2Kernel{}
}
