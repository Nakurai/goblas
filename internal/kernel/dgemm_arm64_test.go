//go:build arm64

package kernel

import (
	"fmt"
	"math/rand"
	"testing"
)

// TestDgemmNEONMatchesGeneric exercises the tiled NEON dgemm against the
// naive reference across sizes that cover full tiles, edge tiles in both
// dimensions, multiple kc/mc blocks, all transpose combinations, and betas.
func TestDgemmNEONMatchesGeneric(t *testing.T) {
	r := rand.New(rand.NewSource(11))
	nk := neonKernel{}

	dims := []struct{ m, n, k int }{
		{1, 1, 1},
		{8, 4, 16},   // exactly one full tile
		{7, 3, 5},    // single partial tile
		{16, 8, 32},  // multiple full tiles
		{17, 9, 33},  // full tiles + edges on every dimension
		{64, 64, 64},
		{100, 50, 300}, // k > kc: multiple pc blocks
		{530, 30, 40},  // m > mc: multiple ic blocks
	}
	alphas := []float64{1, 0.5}
	betas := []float64{0, 1, -0.5}

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

						a := make([]float64, lda*colsA)
						b := make([]float64, ldb*colsB)
						cg := make([]float64, ldc*d.n)
						cn := make([]float64, ldc*d.n)
						for i := range a {
							a[i] = r.NormFloat64()
						}
						for i := range b {
							b[i] = r.NormFloat64()
						}
						for i := range cg {
							v := r.NormFloat64()
							cg[i], cn[i] = v, v
						}

						dgemmNaive(transA, transB, d.m, d.n, d.k, alpha, a, lda, b, ldb, beta, cg, ldc)
						nk.Dgemm(transA, transB, d.m, d.n, d.k, alpha, a, lda, b, ldb, beta, cn, ldc)

						for i := range cg {
							if !close10(cn[i], cg[i]) {
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

func BenchmarkDgemmGenericVsNEON(b *testing.B) {
	g := genericKernel{}
	nk := neonKernel{}
	for _, n := range []int{64, 256, 512, 1024} {
		a := kRandSlice(1, n*n)
		bb := kRandSlice(2, n*n)
		c := make([]float64, n*n)
		flops := 2 * float64(n) * float64(n) * float64(n)

		b.Run(fmt.Sprintf("generic/%d", n), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				g.Dgemm(false, false, n, n, n, 1, a, n, bb, n, 0, c, n)
				kSink += c[n*n-1]
			}
			b.ReportMetric(flops/kSecPerOp(b)/1e9, "GFLOPS")
		})
		b.Run(fmt.Sprintf("neon/%d", n), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				nk.Dgemm(false, false, n, n, n, 1, a, n, bb, n, 0, c, n)
				kSink += c[n*n-1]
			}
			b.ReportMetric(flops/kSecPerOp(b)/1e9, "GFLOPS")
		})
	}
}
