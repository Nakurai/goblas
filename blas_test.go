package goblas

import (
	"math"
	"math/rand"
	"testing"
)

const tol = 1e-12

func closeEnough(a, b, tol float64) bool {
	d := math.Abs(a - b)
	return d <= tol*(1+math.Abs(a)+math.Abs(b))
}

func randSlice(r *rand.Rand, n int) []float64 {
	s := make([]float64, n)
	for i := range s {
		s[i] = r.NormFloat64()
	}
	return s
}

// at returns element (i,j) of a column-major rows×cols matrix with leading
// dimension ld, optionally transposed.
func at(m []float64, ld, i, j int, trans bool) float64 {
	if trans {
		return m[j+i*ld]
	}
	return m[i+j*ld]
}

func TestDdot(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	n := 50
	x, y := randSlice(r, n), randSlice(r, n)
	var want float64
	for i := 0; i < n; i++ {
		want += x[i] * y[i]
	}
	if got := Ddot(n, x, 1, y, 1); !closeEnough(got, want, tol) {
		t.Errorf("Ddot = %v, want %v", got, want)
	}
}

func TestDdotStrided(t *testing.T) {
	r := rand.New(rand.NewSource(2))
	n := 20
	incX, incY := 2, 3
	x := randSlice(r, n*incX)
	y := randSlice(r, n*incY)
	var want float64
	for i := 0; i < n; i++ {
		want += x[i*incX] * y[i*incY]
	}
	if got := Ddot(n, x, incX, y, incY); !closeEnough(got, want, tol) {
		t.Errorf("strided Ddot = %v, want %v", got, want)
	}
}

func TestDaxpy(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	n := 40
	x, y := randSlice(r, n), randSlice(r, n)
	alpha := 1.7
	want := make([]float64, n)
	for i := range want {
		want[i] = alpha*x[i] + y[i]
	}
	Daxpy(n, alpha, x, 1, y, 1)
	for i := range want {
		if !closeEnough(y[i], want[i], tol) {
			t.Fatalf("Daxpy[%d] = %v, want %v", i, y[i], want[i])
		}
	}
}

func TestDnrm2(t *testing.T) {
	r := rand.New(rand.NewSource(4))
	n := 33
	x := randSlice(r, n)
	var ss float64
	for _, v := range x {
		ss += v * v
	}
	want := math.Sqrt(ss)
	if got := Dnrm2(n, x, 1); !closeEnough(got, want, 1e-10) {
		t.Errorf("Dnrm2 = %v, want %v", got, want)
	}
}

func TestDasumIdamax(t *testing.T) {
	x := []float64{1, -4, 3, -2}
	if got := Dasum(4, x, 1); !closeEnough(got, 10, tol) {
		t.Errorf("Dasum = %v, want 10", got)
	}
	if got := Idamax(4, x, 1); got != 1 {
		t.Errorf("Idamax = %v, want 1", got)
	}
}

func TestDgemv(t *testing.T) {
	r := rand.New(rand.NewSource(5))
	for _, trans := range []Transpose{NoTrans, Trans} {
		m, n := 7, 5
		lda := m + 2 // exercise leading dimension > rows
		a := randSlice(r, lda*n)
		lenX, lenY := n, m
		if trans == Trans {
			lenX, lenY = m, n
		}
		x := randSlice(r, lenX)
		y := randSlice(r, lenY)
		alpha, beta := 1.3, -0.7

		want := make([]float64, lenY)
		for i := 0; i < lenY; i++ {
			var s float64
			for l := 0; l < lenX; l++ {
				// op(A)(i,l): A(i,l) for NoTrans, A(l,i) for Trans.
				if trans == NoTrans {
					s += a[i+l*lda] * x[l]
				} else {
					s += a[l+i*lda] * x[l]
				}
			}
			want[i] = alpha*s + beta*y[i]
		}
		Dgemv(trans, m, n, alpha, a, lda, x, 1, beta, y, 1)
		for i := range want {
			if !closeEnough(y[i], want[i], tol) {
				t.Fatalf("Dgemv trans=%v y[%d] = %v, want %v", bool(trans), i, y[i], want[i])
			}
		}
	}
}

func TestDgemm(t *testing.T) {
	r := rand.New(rand.NewSource(6))
	for _, ta := range []Transpose{NoTrans, Trans} {
		for _, tb := range []Transpose{NoTrans, Trans} {
			m, n, k := 6, 4, 5
			lda := rowsLD(m, k, ta) + 1
			ldb := rowsLD(k, n, tb) + 1
			ldc := m + 2

			var a, b []float64
			if ta == NoTrans {
				a = randSlice(r, lda*k)
			} else {
				a = randSlice(r, lda*m)
			}
			if tb == NoTrans {
				b = randSlice(r, ldb*n)
			} else {
				b = randSlice(r, ldb*k)
			}
			c := randSlice(r, ldc*n)
			alpha, beta := 1.1, 0.5

			want := make([]float64, len(c))
			copy(want, c)
			for j := 0; j < n; j++ {
				for i := 0; i < m; i++ {
					var s float64
					for l := 0; l < k; l++ {
						s += at(a, lda, i, l, bool(ta)) * at(b, ldb, l, j, bool(tb))
					}
					want[i+j*ldc] = alpha*s + beta*want[i+j*ldc]
				}
			}
			Dgemm(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
			for j := 0; j < n; j++ {
				for i := 0; i < m; i++ {
					idx := i + j*ldc
					if !closeEnough(c[idx], want[idx], tol) {
						t.Fatalf("Dgemm ta=%v tb=%v C(%d,%d) = %v, want %v",
							bool(ta), bool(tb), i, j, c[idx], want[idx])
					}
				}
			}
		}
	}
}

// rowsLD returns the stored row count of a matrix whose op() shape is r×c.
func rowsLD(r, c int, trans Transpose) int {
	if trans == Trans {
		return c
	}
	return r
}

func TestShortVectorPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("expected panic on short vector")
		}
	}()
	Ddot(10, make([]float64, 3), 1, make([]float64, 10), 1)
}
