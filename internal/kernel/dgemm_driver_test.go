package kernel

import (
	"math"
	"math/rand"
	"testing"
)

// TestDgemmGenericBlockedMatchesNaive verifies the portable blocked dgemm
// (pure-Go micro-kernel + parallel driver) against the naive reference. This
// test runs on every architecture, exercising the path that hosts without
// assembly kernels rely on.
func TestDgemmGenericBlockedMatchesNaive(t *testing.T) {
	r := rand.New(rand.NewSource(12))
	g := genericKernel{}

	dims := []struct{ m, n, k int }{
		{1, 1, 1},
		{8, 4, 16},
		{7, 3, 5},
		{17, 9, 33},
		{64, 64, 64}, // crosses the tiny-problem cutoff into the blocked path
		{100, 50, 600},
		{530, 30, 40},
	}
	for _, d := range dims {
		for _, transA := range []bool{false, true} {
			for _, transB := range []bool{false, true} {
				rowsA, colsA := d.m, d.k
				if transA {
					rowsA, colsA = d.k, d.m
				}
				rowsB, colsB := d.k, d.n
				if transB {
					rowsB, colsB = d.n, d.k
				}
				lda, ldb, ldc := rowsA+2, rowsB+1, d.m+3

				a := make([]float64, lda*colsA)
				b := make([]float64, ldb*colsB)
				cw := make([]float64, ldc*d.n)
				cb := make([]float64, ldc*d.n)
				for i := range a {
					a[i] = r.NormFloat64()
				}
				for i := range b {
					b[i] = r.NormFloat64()
				}
				for i := range cw {
					v := r.NormFloat64()
					cw[i], cb[i] = v, v
				}

				dgemmNaive(transA, transB, d.m, d.n, d.k, 1.2, a, lda, b, ldb, -0.5, cw, ldc)
				g.Dgemm(transA, transB, d.m, d.n, d.k, 1.2, a, lda, b, ldb, -0.5, cb, ldc)

				for i := range cw {
					if math.Abs(cb[i]-cw[i]) > 1e-10*(1+math.Abs(cw[i])) {
						t.Fatalf("m=%d n=%d k=%d tA=%v tB=%v idx=%d: blocked=%v naive=%v",
							d.m, d.n, d.k, transA, transB, i, cb[i], cw[i])
					}
				}
			}
		}
	}
}
