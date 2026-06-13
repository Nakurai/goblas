//go:build arm64

package kernel

import (
	"math"
	"math/rand"
	"testing"
)

func close10(a, b float64) bool { return math.Abs(a-b) <= 1e-10*(1+math.Abs(b)) }

func TestDdotUnitNEONMatchesGeneric(t *testing.T) {
	r := rand.New(rand.NewSource(7))
	g := genericKernel{}
	// Cover every remainder class around the 8-wide and 2-wide loops.
	for n := 0; n <= 40; n++ {
		x := make([]float64, n+1)
		y := make([]float64, n+1)
		for i := range x {
			x[i] = r.NormFloat64()
			y[i] = r.NormFloat64()
		}
		want := g.Ddot(n, x, 1, y, 1)
		var got float64
		if n > 0 {
			got = ddotUnitNEON(n, &x[0], &y[0])
		}
		if !close10(got, want) {
			t.Fatalf("n=%d: neon=%v generic=%v", n, got, want)
		}
	}
}

func TestDaxpyUnitNEONMatchesGeneric(t *testing.T) {
	r := rand.New(rand.NewSource(8))
	g := genericKernel{}
	alpha := 1.75
	for n := 1; n <= 40; n++ {
		x := make([]float64, n)
		yg := make([]float64, n)
		yn := make([]float64, n)
		for i := range x {
			x[i] = r.NormFloat64()
			v := r.NormFloat64()
			yg[i], yn[i] = v, v
		}
		g.Daxpy(n, alpha, x, 1, yg, 1)
		daxpyUnitNEON(n, alpha, &x[0], &yn[0])
		for i := range yg {
			if !close10(yn[i], yg[i]) {
				t.Fatalf("daxpy n=%d i=%d: neon=%v generic=%v", n, i, yn[i], yg[i])
			}
		}
	}
}

func TestDscalUnitNEONMatchesGeneric(t *testing.T) {
	r := rand.New(rand.NewSource(9))
	g := genericKernel{}
	alpha := -0.5
	for n := 1; n <= 40; n++ {
		xg := make([]float64, n)
		xn := make([]float64, n)
		for i := range xg {
			v := r.NormFloat64()
			xg[i], xn[i] = v, v
		}
		g.Dscal(n, alpha, xg, 1)
		dscalUnitNEON(n, alpha, &xn[0])
		for i := range xg {
			if !close10(xn[i], xg[i]) {
				t.Fatalf("dscal n=%d i=%d: neon=%v generic=%v", n, i, xn[i], xg[i])
			}
		}
	}
}
