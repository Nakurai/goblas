package mat32

import (
	"math"
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestQRSolve(t *testing.T) {
	r := rand.New(rand.NewSource(20))
	m, n := 40, 12 // overdetermined least squares
	ad := make([]float32, m*n)
	a64 := make([]float64, m*n)
	for i := range ad {
		v := float32(r.NormFloat64())
		ad[i], a64[i] = v, float64(v)
	}
	bd := make([]float32, m)
	b64 := make([]float64, m)
	for i := range bd {
		v := float32(r.NormFloat64())
		bd[i], b64[i] = v, float64(v)
	}
	var qr QR32
	qr.Factorize(NewDense32(m, n, ad))
	var x Dense32
	if err := qr.SolveTo(&x, false, NewDense32(m, 1, bd)); err != nil {
		t.Fatal(err)
	}
	var wqr mat.QR
	wqr.Factorize(mat.NewDense(m, n, a64))
	var wx mat.Dense
	if err := wqr.SolveTo(&wx, false, mat.NewVecDense(m, b64)); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < n; i++ {
		if d := math.Abs(float64(x.At(i, 0)) - wx.At(i, 0)); d > 1e-2*(1+math.Abs(wx.At(i, 0))) {
			t.Fatalf("QR Solve[%d]: got %v want %v", i, x.At(i, 0), wx.At(i, 0))
		}
	}
}

func TestSVDValues(t *testing.T) {
	r := rand.New(rand.NewSource(21))
	m, n := 30, 20
	ad := make([]float32, m*n)
	a64 := make([]float64, m*n)
	for i := range ad {
		v := float32(r.NormFloat64())
		ad[i], a64[i] = v, float64(v)
	}
	var svd SVD32
	if !svd.Factorize(NewDense32(m, n, ad), mat.SVDThin) {
		t.Fatal("SVD did not converge")
	}
	got := svd.Values(nil)
	var wsvd mat.SVD
	wsvd.Factorize(mat.NewDense(m, n, a64), mat.SVDThin)
	want := wsvd.Values(nil)
	for i := range want {
		if d := math.Abs(float64(got[i]) - want[i]); d > 1e-2*(1+want[i]) {
			t.Fatalf("SVD value[%d]: got %v want %v", i, got[i], want[i])
		}
	}
}

func TestEigenSym(t *testing.T) {
	r := rand.New(rand.NewSource(22))
	n := 16
	a, a64 := spd(r, n) // symmetric PD reuse helper
	var es EigenSym32
	if !es.Factorize(a, true) {
		t.Fatal("EigenSym failed")
	}
	got := es.Values(nil)
	var wes mat.EigenSym
	wes.Factorize(mat.NewSymDense(n, a64), true)
	want := wes.Values(nil)
	for i := range want {
		if d := math.Abs(float64(got[i]) - want[i]); d > 1e-2*(1+math.Abs(want[i])) {
			t.Fatalf("EigenSym value[%d]: got %v want %v", i, got[i], want[i])
		}
	}
	// Eigenvectors should be retrievable and unit-ish.
	var vec Dense32
	es.VectorsTo(&vec)
	if vr, vc := vec.Dims(); vr != n || vc != n {
		t.Fatalf("EigenSym vectors dims (%d,%d)", vr, vc)
	}
}
