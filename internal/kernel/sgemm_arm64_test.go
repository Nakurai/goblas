//go:build arm64

package kernel

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// sCloseRel reports float32 agreement within a relative tolerance (the blocked
// kernel reorders summation, and single precision carries ~7 digits).
func sCloseRel(got, want float32) bool {
	d := math.Abs(float64(got) - float64(want))
	return d/math.Max(1, math.Abs(float64(want))) <= 1e-3
}

func sRandSlice(seed int64, n int) []float32 {
	r := rand.New(rand.NewSource(seed))
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(r.NormFloat64())
	}
	return v
}

// TestSgemmNEONMatchesGeneric mirrors the float64 dgemm test: the tiled NEON
// sgemm against the naive reference across full tiles, edge tiles in both
// dimensions, multiple kc/mc blocks, all transpose combos, and betas.
func TestSgemmNEONMatchesGeneric(t *testing.T) {
	nk := neonKernel{}

	dims := []struct{ m, n, k int }{
		{1, 1, 1},
		{8, 8, 16}, // exactly one full tile
		{7, 5, 5},  // single partial tile
		{16, 16, 32},
		{17, 9, 33}, // full tiles + edges on every dimension
		{64, 64, 64},
		{100, 50, 300}, // k > kc: multiple pc blocks
		{530, 30, 40},  // m > mc: multiple ic blocks
	}
	alphas := []float32{1, 0.5}
	betas := []float32{0, 1, -0.5}

	for _, d := range dims {
		for _, transA := range []bool{false, true} {
			for _, transB := range []bool{false, true} {
				for _, alpha := range alphas {
					for _, beta := range betas {
						rowsA, colsA := d.m, d.k
						if transA {
							rowsA, colsA = d.k, d.m
						}
						rowsB, colsB := d.k, d.n
						if transB {
							rowsB, colsB = d.n, d.k
						}
						lda := rowsA + 2
						ldb := rowsB + 1
						ldc := d.m + 3

						a := sRandSlice(int64(1+d.m), lda*colsA)
						b := sRandSlice(int64(2+d.n), ldb*colsB)
						cg := sRandSlice(int64(3+d.k), ldc*d.n)
						cn := append([]float32(nil), cg...)

						gemmNaive(transA, transB, d.m, d.n, d.k, alpha, a, lda, b, ldb, beta, cg, ldc)
						nk.Sgemm(transA, transB, d.m, d.n, d.k, alpha, a, lda, b, ldb, beta, cn, ldc)

						for i := range cg {
							if !sCloseRel(cn[i], cg[i]) {
								t.Fatalf("m=%d n=%d k=%d tA=%v tB=%v alpha=%v beta=%v idx=%d: neon=%v generic=%v",
									d.m, d.n, d.k, transA, transB, alpha, beta, i, cn[i], cg[i])
							}
						}
					}
				}
			}
		}
	}
}

// BenchmarkSgemmGenericVsNEON reports float32 GFLOPS for the generic and NEON
// kernels; compare against BenchmarkDgemmGenericVsNEON to see the single-
// precision speedup (4 lanes per .S4 register vs 2 per .D2).
func BenchmarkSgemmGenericVsNEON(b *testing.B) {
	g := genericKernel{}
	nk := neonKernel{}
	for _, n := range []int{64, 256, 512, 1024} {
		a := sRandSlice(1, n*n)
		bb := sRandSlice(2, n*n)
		c := make([]float32, n*n)
		flops := 2 * float64(n) * float64(n) * float64(n)

		b.Run(fmt.Sprintf("generic/%d", n), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				g.Sgemm(false, false, n, n, n, 1, a, n, bb, n, 0, c, n)
				kSink += float64(c[n*n-1])
			}
			b.ReportMetric(flops/kSecPerOp(b)/1e9, "GFLOPS")
		})
		b.Run(fmt.Sprintf("neon/%d", n), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				nk.Sgemm(false, false, n, n, n, 1, a, n, bb, n, 0, c, n)
				kSink += float64(c[n*n-1])
			}
			b.ReportMetric(flops/kSecPerOp(b)/1e9, "GFLOPS")
		})
	}
}

// BenchmarkSgemmBlockSweep sweeps kc/mc for the float32 kernel at n=1024 to
// re-tune blocking for the smaller element (packed A block = mc*kc*4 bytes).
func BenchmarkSgemmBlockSweep(b *testing.B) {
	nk := neonKernel{}
	n := 1024
	a := sRandSlice(1, n*n)
	bb := sRandSlice(2, n*n)
	c := make([]float32, n*n)
	flops := 2 * float64(n) * float64(n) * float64(n)
	savedKC, savedMC := dgemmKC, dgemmMC
	defer func() { dgemmKC, dgemmMC = savedKC, savedMC }()
	for _, kc := range []int{384, 512, 640, 768} {
		for _, mc := range []int{24, 32, 48, 64} {
			b.Run(fmt.Sprintf("kc%d/mc%d", kc, mc), func(b *testing.B) {
				dgemmKC, dgemmMC = kc, mc
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					nk.Sgemm(false, false, n, n, n, 1, a, n, bb, n, 0, c, n)
					kSink += float64(c[n*n-1])
				}
				b.ReportMetric(flops/kSecPerOp(b)/1e9, "GFLOPS")
			})
		}
	}
}

// BenchmarkSgemmKernelSweep compares the 8x8 and 8x12 float32 NEON kernels
// interleaved (per-size A/B) so thermal drift hits both equally.
func BenchmarkSgemmKernelSweep(b *testing.B) {
	nk := neonKernel{}
	savedMK, savedNR := neonSMicroKernel, neonSNR
	defer func() { neonSMicroKernel, neonSNR = savedMK, savedNR }()
	kernels := []struct {
		name string
		mk   microKernel[float32]
		nr   int
	}{
		{"8x8", sgemmKernel8x8, 8},
		{"8x12", sgemmKernel8x12, 12},
	}
	for _, n := range []int{256, 512, 1024} {
		a := sRandSlice(1, n*n)
		bb := sRandSlice(2, n*n)
		c := make([]float32, n*n)
		flops := 2 * float64(n) * float64(n) * float64(n)
		for _, kn := range kernels {
			b.Run(fmt.Sprintf("%s/%d", kn.name, n), func(b *testing.B) {
				neonSMicroKernel, neonSNR = kn.mk, kn.nr
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					nk.Sgemm(false, false, n, n, n, 1, a, n, bb, n, 0, c, n)
					kSink += float64(c[n*n-1])
				}
				b.ReportMetric(flops/kSecPerOp(b)/1e9, "GFLOPS")
			})
		}
	}
}
