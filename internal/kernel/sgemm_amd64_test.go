//go:build amd64

package kernel

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/nakurai/goblas/internal/cpu"
)

func sClose5(a, b float32) bool {
	return math.Abs(float64(a)-float64(b)) <= 1e-3*(1+math.Abs(float64(b)))
}

// TestSgemmAVX2MatchesGeneric exercises the tiled AVX2 sgemm against the naive
// reference across full tiles, edge tiles, multiple kc/mc blocks, all transpose
// combos, and betas. It skips on hosts without AVX2+FMA, so on the ARM64 dev
// machine the path is exercised only by cross-compile + go vet (asmdecl); it
// must be run on real x86 to graduate from experimental.
func TestSgemmAVX2MatchesGeneric(t *testing.T) {
	if !cpu.Detect().HasAVX2FMA {
		t.Skip("host has no AVX2+FMA")
	}
	r := rand.New(rand.NewSource(11))
	ak := avx2Kernel{}

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

						a := make([]float32, lda*colsA)
						b := make([]float32, ldb*colsB)
						cg := make([]float32, ldc*d.n)
						cn := make([]float32, ldc*d.n)
						for i := range a {
							a[i] = float32(r.NormFloat64())
						}
						for i := range b {
							b[i] = float32(r.NormFloat64())
						}
						for i := range cg {
							v := float32(r.NormFloat64())
							cg[i], cn[i] = v, v
						}

						gemmNaive(transA, transB, d.m, d.n, d.k, alpha, a, lda, b, ldb, beta, cg, ldc)
						ak.Sgemm(transA, transB, d.m, d.n, d.k, alpha, a, lda, b, ldb, beta, cn, ldc)

						for i := range cg {
							if !sClose5(cn[i], cg[i]) {
								t.Fatalf("m=%d n=%d k=%d tA=%v tB=%v alpha=%v beta=%v idx=%d: avx2=%v generic=%v",
									d.m, d.n, d.k, transA, transB, alpha, beta, i, cn[i], cg[i])
							}
						}
					}
				}
			}
		}
	}
}

func BenchmarkSgemmGenericVsAVX2(b *testing.B) {
	if !cpu.Detect().HasAVX2FMA {
		b.Skip("host has no AVX2+FMA")
	}
	g := genericKernel{}
	ak := avx2Kernel{}
	r := rand.New(rand.NewSource(1))
	for _, n := range []int{64, 256, 512, 1024} {
		a := make([]float32, n*n)
		bb := make([]float32, n*n)
		c := make([]float32, n*n)
		for i := range a {
			a[i], bb[i] = float32(r.NormFloat64()), float32(r.NormFloat64())
		}
		flops := 2 * float64(n) * float64(n) * float64(n)

		for _, k := range []struct {
			name string
			kern Kernel32
		}{{"generic", g}, {"avx2", ak}} {
			b.Run(fmt.Sprintf("%s/%d", k.name, n), func(b *testing.B) {
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					k.kern.Sgemm(false, false, n, n, n, 1, a, n, bb, n, 0, c, n)
					amdSink += float64(c[n*n-1])
				}
				b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
			})
		}
	}
}
