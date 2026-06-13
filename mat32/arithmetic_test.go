package mat32

import (
	"math"
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/mat"
)

// mat32 results are validated against gonum/mat float64 computations on the
// same data, within a float32 relative tolerance.

func randData(r *rand.Rand, n int) ([]float32, []float64) {
	f32 := make([]float32, n)
	f64 := make([]float64, n)
	for i := range f32 {
		v := r.NormFloat64()
		f32[i] = float32(v)
		f64[i] = float64(float32(v)) // same bits the float32 path sees
	}
	return f32, f64
}

func closeMat(t *testing.T, name string, got *Dense32, want mat.Matrix, tol float64) {
	t.Helper()
	r, c := got.Dims()
	wr, wc := want.Dims()
	if r != wr || c != wc {
		t.Fatalf("%s: dims (%d,%d) want (%d,%d)", name, r, c, wr, wc)
	}
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			g := float64(got.At(i, j))
			w := want.At(i, j)
			if math.Abs(g-w) > tol*(1+math.Abs(w)) {
				t.Fatalf("%s[%d,%d]: got %v want %v", name, i, j, g, w)
			}
		}
	}
}

func TestMul(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	for _, d := range []struct{ m, k, n int }{{1, 1, 1}, {3, 4, 2}, {8, 8, 8}, {17, 13, 9}, {64, 50, 40}} {
		af32, af64 := randData(r, d.m*d.k)
		bf32, bf64 := randData(r, d.k*d.n)
		a := NewDense32(d.m, d.k, af32)
		b := NewDense32(d.k, d.n, bf32)
		var got Dense32
		got.Mul(a, b)

		wa := mat.NewDense(d.m, d.k, af64)
		wb := mat.NewDense(d.k, d.n, bf64)
		var want mat.Dense
		want.Mul(wa, wb)
		closeMat(t, "Mul", &got, &want, 1e-3)

		// Transposed operands: (Aᵀ)ᵀ · B style — use Aᵀ where shapes allow.
		af32b, af64b := randData(r, d.k*d.m) // k×m, so its transpose is m×k
		at := NewDense32(d.k, d.m, af32b)
		var gotT Dense32
		gotT.Mul(at.T(), b)
		wat := mat.NewDense(d.k, d.m, af64b)
		var wantT mat.Dense
		wantT.Mul(wat.T(), wb)
		closeMat(t, "Mul(Aᵀ,B)", &gotT, &wantT, 1e-3)
	}
}

func TestMulVec(t *testing.T) {
	r := rand.New(rand.NewSource(2))
	for _, d := range []struct{ m, n int }{{1, 1}, {5, 3}, {40, 37}, {128, 64}} {
		af32, af64 := randData(r, d.m*d.n)
		xf32, xf64 := randData(r, d.n)
		a := NewDense32(d.m, d.n, af32)
		x := NewVecDense32(d.n, xf32)
		var got VecDense32
		got.MulVec(a, x)

		wa := mat.NewDense(d.m, d.n, af64)
		wx := mat.NewVecDense(d.n, xf64)
		var want mat.VecDense
		want.MulVec(wa, wx)
		for i := 0; i < d.m; i++ {
			g, w := float64(got.AtVec(i)), want.AtVec(i)
			if math.Abs(g-w) > 1e-3*(1+math.Abs(w)) {
				t.Fatalf("MulVec[%d]: got %v want %v", i, g, w)
			}
		}
	}
}

func TestAddSubScale(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	m, n := 12, 9
	af32, af64 := randData(r, m*n)
	bf32, bf64 := randData(r, m*n)
	a, b := NewDense32(m, n, af32), NewDense32(m, n, bf32)
	wa, wb := mat.NewDense(m, n, af64), mat.NewDense(m, n, bf64)

	var add Dense32
	add.Add(a, b)
	var wadd mat.Dense
	wadd.Add(wa, wb)
	closeMat(t, "Add", &add, &wadd, 1e-5)

	var sub Dense32
	sub.Sub(a, b)
	var wsub mat.Dense
	wsub.Sub(wa, wb)
	closeMat(t, "Sub", &sub, &wsub, 1e-5)

	var sc Dense32
	sc.Scale(2.5, a)
	var wsc mat.Dense
	wsc.Scale(2.5, wa)
	closeMat(t, "Scale", &sc, &wsc, 1e-5)
}

func TestMulElemApply(t *testing.T) {
	r := rand.New(rand.NewSource(4))
	m, n := 7, 5
	af32, af64 := randData(r, m*n)
	bf32, bf64 := randData(r, m*n)
	a, b := NewDense32(m, n, af32), NewDense32(m, n, bf32)
	wa, wb := mat.NewDense(m, n, af64), mat.NewDense(m, n, bf64)

	var he Dense32
	he.MulElem(a, b)
	var whe mat.Dense
	whe.MulElem(wa, wb)
	closeMat(t, "MulElem", &he, &whe, 1e-4)

	var ap Dense32
	ap.Apply(func(i, j int, v float32) float32 { return v*v + 1 }, a)
	var wap mat.Dense
	wap.Apply(func(i, j int, v float64) float64 { return v*v + 1 }, wa)
	closeMat(t, "Apply", &ap, &wap, 1e-4)
}

func TestOuterNorm(t *testing.T) {
	r := rand.New(rand.NewSource(5))
	m, n := 6, 4
	xf32, xf64 := randData(r, m)
	yf32, yf64 := randData(r, n)
	x, y := NewVecDense32(m, xf32), NewVecDense32(n, yf32)
	var o Dense32
	o.Outer(1.5, x, y)
	var wo mat.Dense
	wo.Outer(1.5, mat.NewVecDense(m, xf64), mat.NewVecDense(n, yf64))
	closeMat(t, "Outer", &o, &wo, 1e-4)

	af32, af64 := randData(r, m*n)
	a := NewDense32(m, n, af32)
	wa := mat.NewDense(m, n, af64)
	if g, w := float64(a.Norm()), mat.Norm(wa, 2); math.Abs(g-w) > 1e-4*(1+w) {
		t.Fatalf("Norm: got %v want %v", g, w)
	}
}

func TestMulAlias(t *testing.T) {
	// Receiver aliasing an input must still produce the correct product.
	r := rand.New(rand.NewSource(6))
	n := 16
	af32, af64 := randData(r, n*n)
	a := NewDense32(n, n, af32)
	wa := mat.NewDense(n, n, af64)
	a.Mul(a, a) // c == a == b
	var want mat.Dense
	want.Mul(wa, wa)
	closeMat(t, "Mul(alias)", a, &want, 1e-3)
}
