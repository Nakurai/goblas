//go:build arm64

package kernel

import (
	"math/rand"
	"testing"
)

func TestDgemvNoTransNEONMatchesGeneric(t *testing.T) {
	r := rand.New(rand.NewSource(10))
	g := genericKernel{}
	alpha := 1.3

	// Cover row-count remainders around the 4-wide and 2-wide loops.
	for m := 1; m <= 33; m++ {
		for _, n := range []int{1, 3, 7, 8, 16} {
			lda := m + 3 // exercise non-compact leading dimension

			a := make([]float64, lda*n)
			x := make([]float64, n)
			yg := make([]float64, m)
			yn := make([]float64, m)

			for i := range a {
				a[i] = r.NormFloat64()
			}
			for i := range x {
				x[i] = r.NormFloat64()
			}
			for i := range yg {
				v := r.NormFloat64()
				yg[i] = v
				yn[i] = v
			}

			// beta=0: y = alpha*A*x (start from same y then zero it)
			g.Dgemv(false, m, n, alpha, a, lda, x, 1, 0, yg, 1)
			scaleStrided(m, 0, yn, 1)
			dgemvNoTransNEON(m, n, alpha, &a[0], lda, &x[0], &yn[0])

			for i := range yg {
				if !close10(yn[i], yg[i]) {
					t.Fatalf("m=%d n=%d i=%d: neon=%v generic=%v", m, n, i, yn[i], yg[i])
				}
			}
		}
	}
}
