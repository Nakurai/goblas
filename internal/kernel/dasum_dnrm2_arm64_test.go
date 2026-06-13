//go:build arm64

package kernel

import (
	"math"
	"math/rand"
	"testing"
)

func TestDasumUnitNEONMatchesGeneric(t *testing.T) {
	r := rand.New(rand.NewSource(40))
	g := genericKernel{}
	// Cover every remainder class around the 8-wide and 2-wide loops.
	for n := 0; n <= 40; n++ {
		x := make([]float64, n+1)
		for i := range x {
			x[i] = r.NormFloat64()
		}
		want := g.Dasum(n, x, 1)
		var got float64
		if n > 0 {
			got = dasumUnitNEON(n, &x[0])
		}
		if !close10(got, want) {
			t.Fatalf("dasum n=%d: neon=%v generic=%v", n, got, want)
		}
	}
}

func TestDnrm2NEONMatchesGeneric(t *testing.T) {
	r := rand.New(rand.NewSource(41))
	g := genericKernel{}
	nk := neonKernel{}
	for n := 0; n <= 40; n++ {
		x := make([]float64, n+1)
		for i := range x {
			x[i] = r.NormFloat64()
		}
		want := g.Dnrm2(n, x, 1)
		got := nk.Dnrm2(n, x, 1)
		if !close10(got, want) {
			t.Fatalf("dnrm2 n=%d: neon=%v generic=%v", n, got, want)
		}
	}
}

// TestDnrm2NEONFallback exercises the overflow/underflow/all-zero cases where
// the naive sum-of-squares is unsafe and Dnrm2 must defer to the generic
// scaled algorithm.
func TestDnrm2NEONFallback(t *testing.T) {
	g := genericKernel{}
	nk := neonKernel{}
	cases := map[string][]float64{
		"overflow":  {1e200, 2e200, 3e200},       // sum of squares is +Inf
		"underflow": {1e-180, 2e-180, 3e-180},    // sum of squares underflows to 0
		"allzero":   {0, 0, 0, 0},                // genuinely zero
		"mixed":     {1e200, 1e-180, 3.5, 0, -7}, // wide dynamic range
	}
	for name, x := range cases {
		want := g.Dnrm2(len(x), x, 1)
		got := nk.Dnrm2(len(x), x, 1)
		if math.Abs(got-want) > 1e-12*(1+math.Abs(want)) {
			t.Fatalf("dnrm2 %s: neon=%v generic=%v", name, got, want)
		}
	}
}
