package blasadapt

import (
	"math"
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas32"
)

// TestUse32Integration registers goblas via blas32.Use and drives it through
// gonum's row-major blas32 wrappers (blas32.General / blas32.Gemm) — the real
// path a float32 caller uses — checking the product against a naive reference.
func TestUse32Integration(t *testing.T) {
	Use32()
	if _, ok := blas32.Implementation().(Implementation); !ok {
		t.Fatalf("blas32.Use did not register the goblas adapter")
	}

	r := rand.New(rand.NewSource(140))
	m, n, k := 48, 36, 40
	mk := func(rows, cols int) blas32.General {
		g := blas32.General{Rows: rows, Cols: cols, Stride: cols, Data: make([]float32, rows*cols)}
		for i := range g.Data {
			g.Data[i] = float32(r.NormFloat64())
		}
		return g
	}
	a := mk(m, k)
	b := mk(k, n)
	c := mk(m, n)

	// Naive row-major reference: C = 1*A*B + 0*C.
	want := make([]float32, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var s float32
			for l := 0; l < k; l++ {
				s += a.Data[i*a.Stride+l] * b.Data[l*b.Stride+j]
			}
			want[i*n+j] = s
		}
	}

	blas32.Gemm(blas.NoTrans, blas.NoTrans, 1, a, b, 0, c)

	for i := range want {
		if math.Abs(float64(c.Data[i]-want[i])) > 1e-3*(1+math.Abs(float64(want[i]))) {
			t.Fatalf("blas32.Gemm idx=%d: got %v want %v", i, c.Data[i], want[i])
		}
	}
}
