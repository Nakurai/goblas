package mat32

import (
	"math"
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/mat"
)

// spd builds a random n×n symmetric positive-definite float32 matrix (and the
// matching float64 data) as MᵀM + n·I.
func spd(r *rand.Rand, n int) (*SymDense32, []float64) {
	m32 := make([]float32, n*n)
	for i := range m32 {
		m32[i] = float32(r.NormFloat64())
	}
	a := make([]float32, n*n)
	a64 := make([]float64, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			var s float32
			for k := 0; k < n; k++ {
				s += m32[k*n+i] * m32[k*n+j] // (MᵀM)[i,j]
			}
			if i == j {
				s += float32(n)
			}
			a[i*n+j] = s
			a64[i*n+j] = float64(s)
		}
	}
	return NewSymDense32(n, a), a64
}

func TestCholeskySolve(t *testing.T) {
	r := rand.New(rand.NewSource(10))
	for _, n := range []int{1, 5, 32, 64, 100, 200} {
		a, a64 := spd(r, n)
		var chol Cholesky32
		if !chol.Factorize(a) {
			t.Fatalf("n=%d: not positive definite", n)
		}
		// Random RHS.
		nrhs := 3
		bd := make([]float32, n*nrhs)
		for i := range bd {
			bd[i] = float32(r.NormFloat64())
		}
		b := NewDense32(n, nrhs, bd)
		var x Dense32
		if err := chol.SolveTo(&x, b); err != nil {
			t.Fatal(err)
		}
		// Residual A·x ≈ b.
		var ax Dense32
		ax.Mul(a, &x)
		for i := 0; i < n; i++ {
			for j := 0; j < nrhs; j++ {
				if d := math.Abs(float64(ax.At(i, j) - b.At(i, j))); d > 1e-2*(1+math.Abs(float64(b.At(i, j)))) {
					t.Fatalf("Cholesky n=%d residual[%d,%d]=%v", n, i, j, d)
				}
			}
		}
		// Det vs gonum float64 (relative). Only at small n: the float32
		// determinant overflows to +Inf for larger matrices (~λⁿ growth).
		if n <= 16 {
			var c64 mat.Cholesky
			c64.Factorize(mat.NewSymDense(n, a64))
			if w := c64.Det(); w != 0 {
				got := float64(chol.Det())
				if rel := math.Abs(got-w) / math.Abs(w); rel > 5e-2 {
					t.Errorf("Cholesky n=%d Det rel err %v (got %v want %v)", n, rel, got, w)
				}
			}
		}
	}
}

func TestLUSolve(t *testing.T) {
	r := rand.New(rand.NewSource(11))
	for _, n := range []int{1, 4, 17, 64, 128, 200} {
		// Diagonally dominant → well conditioned.
		ad := make([]float32, n*n)
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				ad[i*n+j] = float32(r.NormFloat64())
			}
			ad[i*n+i] += float32(n)
		}
		a := NewDense32(n, n, ad)
		bd := make([]float32, n)
		for i := range bd {
			bd[i] = float32(r.NormFloat64())
		}
		b := NewDense32(n, 1, bd)
		var x Dense32
		if err := x.Solve(a, b); err != nil {
			t.Fatal(err)
		}
		var ax Dense32
		ax.Mul(a, &x)
		for i := 0; i < n; i++ {
			if d := math.Abs(float64(ax.At(i, 0) - b.At(i, 0))); d > 1e-2*(1+math.Abs(float64(b.At(i, 0)))) {
				t.Fatalf("LU n=%d residual[%d]=%v", n, i, d)
			}
		}
	}
}

func TestLUDet(t *testing.T) {
	r := rand.New(rand.NewSource(12))
	n := 12 // small enough that the float32 determinant does not overflow
	ad := make([]float32, n*n)
	a64 := make([]float64, n*n)
	for i := range ad {
		v := float32(r.NormFloat64())
		ad[i] = v
		a64[i] = float64(v)
	}
	for i := 0; i < n; i++ {
		ad[i*n+i] += float32(n)
		a64[i*n+i] += float64(n)
	}
	var lu LU32
	lu.Factorize(NewDense32(n, n, ad))
	var lu64 mat.LU
	lu64.Factorize(mat.NewDense(n, n, a64))
	got, want := float64(lu.Det()), lu64.Det()
	if rel := math.Abs(got-want) / math.Abs(want); rel > 5e-2 {
		t.Errorf("LU Det rel err %v (got %v want %v)", rel, got, want)
	}
}
