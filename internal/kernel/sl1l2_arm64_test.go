//go:build arm64

package kernel

import (
	"math"
	"math/rand"
	"testing"
)

// These tests pit the float32 NEON L1/L2 kernels against the generic reference
// across lengths that exercise the 16/8-wide main loops, the 4-wide step, and
// the scalar tail, plus strided inputs that must hit the generic fallback.

func sClose(a, b float32, tol float64) bool {
	d := math.Abs(float64(a) - float64(b))
	return d/math.Max(1, math.Abs(float64(b))) <= tol
}

func TestSL1NEONMatchesGeneric(t *testing.T) {
	g := genericKernel{}
	nk := neonKernel{}
	r := rand.New(rand.NewSource(21))
	lens := []int{0, 1, 3, 4, 7, 8, 15, 16, 17, 31, 33, 100, 1000}

	for _, n := range lens {
		x := make([]float32, n+1)
		y := make([]float32, n+1)
		for i := range x {
			x[i] = float32(r.NormFloat64())
			y[i] = float32(r.NormFloat64())
		}

		// Sdot
		if n > 0 {
			if got, want := nk.Sdot(n, x, 1, y, 1), g.Sdot(n, x, 1, y, 1); !sClose(got, want, 1e-4) {
				t.Errorf("Sdot n=%d: neon=%v generic=%v", n, got, want)
			}
		}
		// Sasum / Snrm2 / Sscal / Saxpy
		if got, want := nk.Sasum(n, x, 1), g.Sasum(n, x, 1); !sClose(got, want, 1e-4) {
			t.Errorf("Sasum n=%d: neon=%v generic=%v", n, got, want)
		}
		if got, want := nk.Snrm2(n, x, 1), g.Snrm2(n, x, 1); !sClose(got, want, 1e-4) {
			t.Errorf("Snrm2 n=%d: neon=%v generic=%v", n, got, want)
		}
		xs1 := append([]float32(nil), x...)
		xs2 := append([]float32(nil), x...)
		nk.Sscal(n, 1.7, xs1, 1)
		g.Sscal(n, 1.7, xs2, 1)
		for i := 0; i < n; i++ {
			if !sClose(xs1[i], xs2[i], 1e-5) {
				t.Fatalf("Sscal n=%d[%d]: neon=%v generic=%v", n, i, xs1[i], xs2[i])
			}
		}
		ya1 := append([]float32(nil), y...)
		ya2 := append([]float32(nil), y...)
		nk.Saxpy(n, -0.9, x, 1, ya1, 1)
		g.Saxpy(n, -0.9, x, 1, ya2, 1)
		for i := 0; i < n; i++ {
			if !sClose(ya1[i], ya2[i], 1e-5) {
				t.Fatalf("Saxpy n=%d[%d]: neon=%v generic=%v", n, i, ya1[i], ya2[i])
			}
		}
	}

	// Strided inputs must agree with the reference (fallback path).
	n := 50
	x := make([]float32, 2*n)
	y := make([]float32, 3*n)
	for i := range x {
		x[i] = float32(r.NormFloat64())
	}
	for i := range y {
		y[i] = float32(r.NormFloat64())
	}
	if got, want := nk.Sdot(n, x, 2, y, 3), g.Sdot(n, x, 2, y, 3); !sClose(got, want, 1e-4) {
		t.Errorf("Sdot strided: neon=%v generic=%v", got, want)
	}
	if got, want := nk.Sasum(n, x, 2), g.Sasum(n, x, 2); !sClose(got, want, 1e-4) {
		t.Errorf("Sasum strided: neon=%v generic=%v", got, want)
	}
}

func TestSgemvSgerStrsvNEONMatchesGeneric(t *testing.T) {
	g := genericKernel{}
	nk := neonKernel{}
	r := rand.New(rand.NewSource(22))

	for _, dim := range [][2]int{{1, 1}, {8, 8}, {17, 9}, {64, 50}, {130, 77}} {
		m, n := dim[0], dim[1]
		a := make([]float32, m*n)
		for i := range a {
			a[i] = float32(r.NormFloat64())
		}
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(r.NormFloat64())
		}
		y := make([]float32, m)
		for i := range y {
			y[i] = float32(r.NormFloat64())
		}
		// Sgemv NoTrans (NEON path) vs generic.
		y1 := append([]float32(nil), y...)
		y2 := append([]float32(nil), y...)
		nk.Sgemv(false, m, n, 1.3, a, m, x, 1, -0.5, y1, 1)
		g.Sgemv(false, m, n, 1.3, a, m, x, 1, -0.5, y2, 1)
		for i := range y1 {
			if !sClose(y1[i], y2[i], 1e-3) {
				t.Fatalf("Sgemv m=%d n=%d[%d]: neon=%v generic=%v", m, n, i, y1[i], y2[i])
			}
		}
		// Sger (reuses NEON saxpy) vs generic.
		a1 := append([]float32(nil), a...)
		a2 := append([]float32(nil), a...)
		xm := make([]float32, m)
		for i := range xm {
			xm[i] = float32(r.NormFloat64())
		}
		nk.Sger(m, n, 0.7, xm, 1, x, 1, a1, m)
		g.Sger(m, n, 0.7, xm, 1, x, 1, a2, m)
		for i := range a1 {
			if !sClose(a1[i], a2[i], 1e-3) {
				t.Fatalf("Sger m=%d n=%d[%d]: neon=%v generic=%v", m, n, i, a1[i], a2[i])
			}
		}
	}

	// Strsv: well-conditioned triangular system, NEON vs generic, all 4 modes.
	n := 80
	tri := make([]float32, n*n)
	for j := 0; j < n; j++ {
		for i := 0; i < n; i++ {
			tri[i+j*n] = float32(r.NormFloat64()) * 0.3
		}
		tri[j+j*n] = float32(2 + r.Float64())
	}
	for _, upper := range []bool{true, false} {
		for _, trans := range []bool{true, false} {
			b := make([]float32, n)
			for i := range b {
				b[i] = float32(r.NormFloat64())
			}
			x1 := append([]float32(nil), b...)
			x2 := append([]float32(nil), b...)
			nk.Strsv(upper, trans, false, n, tri, n, x1, 1)
			g.Strsv(upper, trans, false, n, tri, n, x2, 1)
			for i := range x1 {
				if !sClose(x1[i], x2[i], 5e-3) {
					t.Fatalf("Strsv up=%v tr=%v[%d]: neon=%v generic=%v", upper, trans, i, x1[i], x2[i])
				}
			}
		}
	}
}

func BenchmarkSL1GenericVsNEON(b *testing.B) {
	g := genericKernel{}
	nk := neonKernel{}
	n := 65536
	x := sRandSlice(1, n)
	y := sRandSlice(2, n)
	bench := func(name string, fn func()) {
		b.Run(name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				fn()
			}
			b.ReportMetric(float64(n)/kSecPerOp(b)/1e9, "Gelem/s")
		})
	}
	bench("Sdot/generic", func() { kSink += float64(g.Sdot(n, x, 1, y, 1)) })
	bench("Sdot/neon", func() { kSink += float64(nk.Sdot(n, x, 1, y, 1)) })
	bench("Sasum/generic", func() { kSink += float64(g.Sasum(n, x, 1)) })
	bench("Sasum/neon", func() { kSink += float64(nk.Sasum(n, x, 1)) })
	bench("Snrm2/generic", func() { kSink += float64(g.Snrm2(n, x, 1)) })
	bench("Snrm2/neon", func() { kSink += float64(nk.Snrm2(n, x, 1)) })
}
