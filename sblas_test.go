package goblas

import (
	"math"
	"math/rand"
	"testing"
)

// These tests validate the float32 (S) public API against independent naive
// float32 references. The S and D routines share the same generic engine
// (already exercised by the float64 suite), so the focus here is the float32
// instantiation, the S-API validation/wiring, and the float32 helpers.

// relClose reports whether got and want agree within a float32-appropriate
// relative tolerance (single precision carries ~7 significant digits, and the
// blocked kernels reorder summation, so exact equality is not expected).
func relClose(got, want float32, tol float64) bool {
	d := math.Abs(float64(got) - float64(want))
	scale := math.Max(1, math.Abs(float64(want)))
	return d/scale <= tol
}

func randVec32(rng *rand.Rand, n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(rng.NormFloat64())
	}
	return v
}

func TestSdot(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for _, n := range []int{0, 1, 3, 8, 17, 100, 1000} {
		x, y := randVec32(rng, n+1), randVec32(rng, n+1)
		var want float32
		for i := 0; i < n; i++ {
			want += x[i] * y[i]
		}
		if got := Sdot(n, x, 1, y, 1); !relClose(got, want, 1e-4) {
			t.Errorf("Sdot n=%d: got %v want %v", n, got, want)
		}
	}
}

func TestSaxpySscal(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	n := 257
	x := randVec32(rng, n)
	y := randVec32(rng, n)
	want := make([]float32, n)
	const alpha float32 = 1.5
	for i := range want {
		want[i] = alpha*x[i] + y[i]
	}
	Saxpy(n, alpha, x, 1, y, 1)
	for i := range y {
		if !relClose(y[i], want[i], 1e-5) {
			t.Fatalf("Saxpy[%d]: got %v want %v", i, y[i], want[i])
		}
	}

	x2 := randVec32(rng, n)
	want2 := make([]float32, n)
	for i := range want2 {
		want2[i] = alpha * x2[i]
	}
	Sscal(n, alpha, x2, 1)
	for i := range x2 {
		if !relClose(x2[i], want2[i], 1e-5) {
			t.Fatalf("Sscal[%d]: got %v want %v", i, x2[i], want2[i])
		}
	}
}

func TestSnrm2Sasum(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for _, n := range []int{1, 7, 64, 1000} {
		x := randVec32(rng, n)
		var ss, asum float64
		for _, v := range x {
			ss += float64(v) * float64(v)
			asum += math.Abs(float64(v))
		}
		if got, want := Snrm2(n, x, 1), float32(math.Sqrt(ss)); !relClose(got, want, 1e-4) {
			t.Errorf("Snrm2 n=%d: got %v want %v", n, got, want)
		}
		if got, want := Sasum(n, x, 1), float32(asum); !relClose(got, want, 1e-4) {
			t.Errorf("Sasum n=%d: got %v want %v", n, got, want)
		}
	}
}

func TestIsamax(t *testing.T) {
	x := []float32{0.5, -2.0, 1.0, -3.5, 3.4}
	if got := Isamax(len(x), x, 1); got != 3 {
		t.Errorf("Isamax: got %d want 3", got)
	}
	if got := Isamax(0, nil, 1); got != -1 {
		t.Errorf("Isamax empty: got %d want -1", got)
	}
}

func TestScopySswap(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	n := 33
	x := randVec32(rng, n)
	y := make([]float32, n)
	Scopy(n, x, 1, y, 1)
	for i := range x {
		if x[i] != y[i] {
			t.Fatalf("Scopy[%d]: got %v want %v", i, y[i], x[i])
		}
	}
	a := randVec32(rng, n)
	b := randVec32(rng, n)
	a0 := append([]float32(nil), a...)
	b0 := append([]float32(nil), b...)
	Sswap(n, a, 1, b, 1)
	for i := range a {
		if a[i] != b0[i] || b[i] != a0[i] {
			t.Fatalf("Sswap[%d] mismatch", i)
		}
	}
}

// sgemmNaive is an independent column-major reference for Sgemm.
func sgemmNaive(tA, tB Transpose, m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	for j := 0; j < n; j++ {
		for i := 0; i < m; i++ {
			c[i+j*ldc] *= beta
		}
	}
	for j := 0; j < n; j++ {
		for i := 0; i < m; i++ {
			var s float32
			for l := 0; l < k; l++ {
				var av, bv float32
				if tA == NoTrans {
					av = a[i+l*lda]
				} else {
					av = a[l+i*lda]
				}
				if tB == NoTrans {
					bv = b[l+j*ldb]
				} else {
					bv = b[j+l*ldb]
				}
				s += av * bv
			}
			c[i+j*ldc] += alpha * s
		}
	}
}

func TestSgemm(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	for _, tA := range []Transpose{NoTrans, Trans} {
		for _, tB := range []Transpose{NoTrans, Trans} {
			for _, sz := range [][3]int{{8, 8, 8}, {17, 13, 9}, {64, 64, 64}, {200, 96, 128}} {
				m, n, k := sz[0], sz[1], sz[2]
				ra, ca := m, k
				if tA == Trans {
					ra, ca = k, m
				}
				rb, cb := k, n
				if tB == Trans {
					rb, cb = n, k
				}
				a := randVec32(rng, ra*ca)
				b := randVec32(rng, rb*cb)
				c := randVec32(rng, m*n)
				want := append([]float32(nil), c...)
				const alpha, beta float32 = 1.3, -0.7
				sgemmNaive(tA, tB, m, n, k, alpha, a, ra, b, rb, beta, want, m)
				Sgemm(tA, tB, m, n, k, alpha, a, ra, b, rb, beta, c, m)
				for i := range c {
					if !relClose(c[i], want[i], 1e-3) {
						t.Fatalf("Sgemm tA=%v tB=%v %dx%dx%d [%d]: got %v want %v", tA, tB, m, n, k, i, c[i], want[i])
					}
				}
			}
		}
	}
}

func TestSgemv(t *testing.T) {
	rng := rand.New(rand.NewSource(6))
	m, n := 50, 37
	a := randVec32(rng, m*n)
	for _, tr := range []Transpose{NoTrans, Trans} {
		lenX, lenY := n, m
		if tr == Trans {
			lenX, lenY = m, n
		}
		x := randVec32(rng, lenX)
		y := randVec32(rng, lenY)
		want := append([]float32(nil), y...)
		const alpha, beta float32 = 0.9, 0.4
		for i := 0; i < lenY; i++ {
			var s float32
			for j := 0; j < lenX; j++ {
				if tr == NoTrans {
					s += a[i+j*m] * x[j]
				} else {
					s += a[j+i*m] * x[j]
				}
			}
			want[i] = alpha*s + beta*want[i]
		}
		Sgemv(tr, m, n, alpha, a, m, x, 1, beta, y, 1)
		for i := range y {
			if !relClose(y[i], want[i], 1e-3) {
				t.Fatalf("Sgemv tr=%v [%d]: got %v want %v", tr, i, y[i], want[i])
			}
		}
	}
}

func TestSsyrkStrsmAgainstDense(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	n, k := 96, 40
	// Ssyrk: C = A*Aᵀ, lower; compare against a dense float32 product.
	a := randVec32(rng, n*k)
	c := make([]float32, n*n)
	Ssyrk(Lower, NoTrans, n, k, 1, a, n, 0, c, n)
	for j := 0; j < n; j++ {
		for i := j; i < n; i++ {
			var s float32
			for l := 0; l < k; l++ {
				s += a[i+l*n] * a[j+l*n]
			}
			if !relClose(c[i+j*n], s, 1e-3) {
				t.Fatalf("Ssyrk[%d,%d]: got %v want %v", i, j, c[i+j*n], s)
			}
		}
	}

	// Strsm: solve A*X = B (lower, non-unit), then check A*X ≈ B.
	m, nb := 64, 20
	tri := make([]float32, m*m)
	for j := 0; j < m; j++ {
		for i := j; i < m; i++ {
			tri[i+j*m] = float32(rng.NormFloat64()) * 0.5
		}
		tri[j+j*m] = float32(2 + rng.Float64()) // well-conditioned diagonal
	}
	bOrig := randVec32(rng, m*nb)
	x := append([]float32(nil), bOrig...)
	Strsm(Left, Lower, NoTrans, NonUnit, m, nb, 1, tri, m, x, m)
	for j := 0; j < nb; j++ {
		for i := 0; i < m; i++ {
			var s float32
			for l := 0; l <= i; l++ {
				s += tri[i+l*m] * x[l+j*m]
			}
			if !relClose(s, bOrig[i+j*m], 5e-3) {
				t.Fatalf("Strsm residual [%d,%d]: A*X=%v B=%v", i, j, s, bOrig[i+j*m])
			}
		}
	}
}

func TestSsymmStrmmAgainstNaive(t *testing.T) {
	rng := rand.New(rand.NewSource(8))
	m, n := 80, 48
	// Ssymm left/upper: C = A*B, A symmetric (upper stored).
	a := make([]float32, m*m)
	for j := 0; j < m; j++ {
		for i := 0; i <= j; i++ {
			a[i+j*m] = float32(rng.NormFloat64())
		}
	}
	symAt := func(i, j int) float32 {
		if i <= j {
			return a[i+j*m]
		}
		return a[j+i*m]
	}
	b := randVec32(rng, m*n)
	c := randVec32(rng, m*n)
	want := append([]float32(nil), c...)
	const alpha, beta float32 = 1.1, 0.3
	for j := 0; j < n; j++ {
		for i := 0; i < m; i++ {
			var s float32
			for l := 0; l < m; l++ {
				s += symAt(i, l) * b[l+j*m]
			}
			want[i+j*m] = alpha*s + beta*want[i+j*m]
		}
	}
	Ssymm(Left, Upper, m, n, alpha, a, m, b, m, beta, c, m)
	for i := range c {
		if !relClose(c[i], want[i], 2e-3) {
			t.Fatalf("Ssymm [%d]: got %v want %v", i, c[i], want[i])
		}
	}

	// Strmm left/upper/non-unit: B = alpha*A*B in place.
	tri := make([]float32, m*m)
	for j := 0; j < m; j++ {
		for i := 0; i <= j; i++ {
			tri[i+j*m] = float32(rng.NormFloat64())
		}
	}
	bb := randVec32(rng, m*n)
	want2 := make([]float32, m*n)
	for j := 0; j < n; j++ {
		for i := 0; i < m; i++ {
			var s float32
			for l := i; l < m; l++ { // upper triangular
				s += tri[i+l*m] * bb[l+j*m]
			}
			want2[i+j*m] = alpha * s
		}
	}
	Strmm(Left, Upper, NoTrans, NonUnit, m, n, alpha, tri, m, bb, m)
	for i := range bb {
		if !relClose(bb[i], want2[i], 2e-3) {
			t.Fatalf("Strmm [%d]: got %v want %v", i, bb[i], want2[i])
		}
	}
}

func TestStrsv(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	n := 64
	tri := make([]float32, n*n)
	for j := 0; j < n; j++ {
		for i := 0; i <= j; i++ {
			tri[i+j*n] = float32(rng.NormFloat64()) * 0.5
		}
		tri[j+j*n] = float32(2 + rng.Float64())
	}
	bOrig := randVec32(rng, n)
	x := append([]float32(nil), bOrig...)
	Strsv(Upper, NoTrans, NonUnit, n, tri, n, x, 1)
	// Check A*x ≈ b.
	for i := 0; i < n; i++ {
		var s float32
		for l := i; l < n; l++ {
			s += tri[i+l*n] * x[l]
		}
		if !relClose(s, bOrig[i], 5e-3) {
			t.Fatalf("Strsv residual [%d]: A*x=%v b=%v", i, s, bOrig[i])
		}
	}
}
