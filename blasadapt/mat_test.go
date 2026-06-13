package blasadapt

import (
	"math"
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/blas/blas64"
	gonumblas "gonum.org/v1/gonum/blas/gonum"
	"gonum.org/v1/gonum/mat"
)

// withStock runs f with Gonum's stock BLAS registered; withOurs with goblas.
// blas64.Use is a process-wide switch, so tests must always restore stock.
func withOurs(f func()) {
	Use()
	defer blas64.Use(gonumblas.Implementation{})
	f()
}

func randDense(r *rand.Rand, m, n int) *mat.Dense {
	d := mat.NewDense(m, n, nil)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			d.Set(i, j, r.NormFloat64())
		}
	}
	return d
}

// spd builds a symmetric positive-definite matrix A = MᵀM + n·I.
func spd(r *rand.Rand, n int) *mat.SymDense {
	m := randDense(r, n, n)
	var p mat.Dense
	p.Mul(m.T(), m)
	s := mat.NewSymDense(n, nil)
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			v := p.At(i, j)
			if i == j {
				v += float64(n)
			}
			s.SetSym(i, j, v)
		}
	}
	return s
}

func maxAbsDiff(a, b mat.Matrix) float64 {
	ra, ca := a.Dims()
	var d float64
	for i := 0; i < ra; i++ {
		for j := 0; j < ca; j++ {
			d = math.Max(d, math.Abs(a.At(i, j)-b.At(i, j)))
		}
	}
	return d
}

// TestMatOnGoblas runs gonum/mat's high-level operations with goblas
// registered as the BLAS and checks the results agree with stock Gonum.
// This exercises Gonum's pure-Go LAPACK (getrf, potrf, geqrf, gesvd, ...)
// issuing its BLAS calls into goblas kernels.
func TestMatOnGoblas(t *testing.T) {
	r := rand.New(rand.NewSource(40))
	n := 120
	a := randDense(r, n, n)
	b := randDense(r, n, 3)
	s := spd(r, n)

	// --- Stock results first ---
	var solveWant mat.Dense
	if err := solveWant.Solve(a, b); err != nil {
		t.Fatal(err)
	}
	var luWant mat.LU
	luWant.Factorize(a)
	detWant := luWant.Det()

	var cholWant mat.Cholesky
	if !cholWant.Factorize(s) {
		t.Fatal("stock cholesky failed")
	}
	var cholSolveWant mat.Dense
	if err := cholWant.SolveTo(&cholSolveWant, b); err != nil {
		t.Fatal(err)
	}

	var qrWant mat.QR
	qrWant.Factorize(a)
	var qrSolveWant mat.Dense
	if err := qrWant.SolveTo(&qrSolveWant, false, b); err != nil {
		t.Fatal(err)
	}

	var svdWant mat.SVD
	if !svdWant.Factorize(a, mat.SVDThin) {
		t.Fatal("stock svd failed")
	}
	svWant := svdWant.Values(nil)

	var mulWant mat.Dense
	mulWant.Mul(a, b)

	// --- Same operations with goblas registered ---
	withOurs(func() {
		var got mat.Dense
		if err := got.Solve(a, b); err != nil {
			t.Fatal(err)
		}
		if d := maxAbsDiff(&got, &solveWant); d > 1e-8 {
			t.Errorf("Solve differs from stock by %v", d)
		}

		var lu mat.LU
		lu.Factorize(a)
		if det := lu.Det(); math.Abs(det-detWant) > 1e-6*(1+math.Abs(detWant)) {
			t.Errorf("LU det: got %v want %v", det, detWant)
		}

		var chol mat.Cholesky
		if !chol.Factorize(s) {
			t.Fatal("cholesky on goblas failed")
		}
		var cholSolve mat.Dense
		if err := chol.SolveTo(&cholSolve, b); err != nil {
			t.Fatal(err)
		}
		if d := maxAbsDiff(&cholSolve, &cholSolveWant); d > 1e-8 {
			t.Errorf("Cholesky solve differs from stock by %v", d)
		}

		var qr mat.QR
		qr.Factorize(a)
		var qrSolve mat.Dense
		if err := qr.SolveTo(&qrSolve, false, b); err != nil {
			t.Fatal(err)
		}
		if d := maxAbsDiff(&qrSolve, &qrSolveWant); d > 1e-8 {
			t.Errorf("QR solve differs from stock by %v", d)
		}

		var svd mat.SVD
		if !svd.Factorize(a, mat.SVDThin) {
			t.Fatal("svd on goblas failed")
		}
		sv := svd.Values(nil)
		for i := range sv {
			if math.Abs(sv[i]-svWant[i]) > 1e-8*(1+svWant[i]) {
				t.Errorf("singular value %d: got %v want %v", i, sv[i], svWant[i])
			}
		}

		var mul mat.Dense
		mul.Mul(a, b)
		if d := maxAbsDiff(&mul, &mulWant); d > 1e-10 {
			t.Errorf("Mul differs from stock by %v", d)
		}
	})
}

// Benchmarks: the same mat operation with stock Gonum vs goblas registered.

func benchMatMul(b *testing.B, n int) {
	r := rand.New(rand.NewSource(1))
	x := randDense(r, n, n)
	y := randDense(r, n, n)
	var z mat.Dense
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		z.Mul(x, y)
	}
}

func BenchmarkMatMul1024Stock(b *testing.B) { benchMatMul(b, 1024) }
func BenchmarkMatMul1024Goblas(b *testing.B) {
	Use()
	defer blas64.Use(gonumblas.Implementation{})
	benchMatMul(b, 1024)
}

func benchCholesky(b *testing.B, n int) {
	r := rand.New(rand.NewSource(2))
	s := spd(r, n)
	var c mat.Cholesky
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if !c.Factorize(s) {
			b.Fatal("factorization failed")
		}
	}
}

func BenchmarkCholesky1024Stock(b *testing.B) { benchCholesky(b, 1024) }
func BenchmarkCholesky1024Goblas(b *testing.B) {
	Use()
	defer blas64.Use(gonumblas.Implementation{})
	benchCholesky(b, 1024)
}

func benchSolve(b *testing.B, n int) {
	r := rand.New(rand.NewSource(3))
	a := randDense(r, n, n)
	rhs := randDense(r, n, 8)
	var x mat.Dense
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := x.Solve(a, rhs); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSolve1024Stock(b *testing.B) { benchSolve(b, 1024) }
func BenchmarkSolve1024Goblas(b *testing.B) {
	Use()
	defer blas64.Use(gonumblas.Implementation{})
	benchSolve(b, 1024)
}
