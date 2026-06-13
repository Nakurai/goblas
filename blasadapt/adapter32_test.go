package blasadapt

import (
	"math"
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/blas"
)

// The float32 adapter must agree with gonum's own row-major float32 BLAS on
// identical inputs — this is what validates the row→column relabeling rules for
// the S routines (they mirror the D rules, so a mistake shows up immediately).

// Compile-time check that the adapter satisfies the float32 BLAS interface.
var _ blas.Float32 = Implementation{}

func randSlice32(r *rand.Rand, n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32(r.NormFloat64())
	}
	return s
}

func almostEqual32(t *testing.T, name string, got, want []float32, tol float64) {
	t.Helper()
	for i := range want {
		if math.Abs(float64(got[i])-float64(want[i])) > tol*(1+math.Abs(float64(want[i]))) {
			t.Fatalf("%s: idx=%d got %v want %v", name, i, got[i], want[i])
		}
	}
}

func TestSgemmMatchesGonum(t *testing.T) {
	r := rand.New(rand.NewSource(130))
	for _, d := range []struct{ m, n, k int }{{1, 1, 1}, {7, 5, 9}, {32, 17, 23}, {64, 64, 64}} {
		for _, tA := range transposes {
			for _, tB := range transposes {
				rA, cA := d.m, d.k
				if tA == blas.Trans {
					rA, cA = d.k, d.m
				}
				rB, cB := d.k, d.n
				if tB == blas.Trans {
					rB, cB = d.n, d.k
				}
				lda, ldb, ldc := cA+1, cB+2, d.n+1
				a := randSlice32(r, rA*lda)
				b := randSlice32(r, rB*ldb)
				cw := randSlice32(r, d.m*ldc)
				cg := append([]float32(nil), cw...)
				ours.Sgemm(tA, tB, d.m, d.n, d.k, 1.2, a, lda, b, ldb, -0.4, cw, ldc)
				stock.Sgemm(tA, tB, d.m, d.n, d.k, 1.2, a, lda, b, ldb, -0.4, cg, ldc)
				almostEqual32(t, "Sgemm", cw, cg, 1e-3)
			}
		}
	}
}

func TestSgemvSgerMatchGonum(t *testing.T) {
	r := rand.New(rand.NewSource(131))
	m, n := 40, 25
	for _, tA := range transposes {
		lda := n + 1
		a := randSlice32(r, m*lda)
		lenX, lenY := n, m
		if tA == blas.Trans {
			lenX, lenY = m, n
		}
		x := randSlice32(r, lenX)
		yw := randSlice32(r, lenY)
		yg := append([]float32(nil), yw...)
		ours.Sgemv(tA, m, n, 0.9, a, lda, x, 1, 0.5, yw, 1)
		stock.Sgemv(tA, m, n, 0.9, a, lda, x, 1, 0.5, yg, 1)
		almostEqual32(t, "Sgemv", yw, yg, 1e-3)
	}
	// Sger.
	lda := n + 2
	x := randSlice32(r, m)
	y := randSlice32(r, n)
	aw := randSlice32(r, m*lda)
	ag := append([]float32(nil), aw...)
	ours.Sger(m, n, 0.7, x, 1, y, 1, aw, lda)
	stock.Sger(m, n, 0.7, x, 1, y, 1, ag, lda)
	almostEqual32(t, "Sger", aw, ag, 1e-3)
}

func TestSsyrkSsymmMatchGonum(t *testing.T) {
	r := rand.New(rand.NewSource(132))
	n, k := 48, 30
	for _, ul := range uplos {
		for _, tr := range transposes {
			rA, cA := n, k
			if tr == blas.Trans {
				rA, cA = k, n
			}
			lda := cA + 1
			a := randSlice32(r, rA*lda)
			ldc := n + 1
			cw := randSlice32(r, n*ldc)
			cg := append([]float32(nil), cw...)
			ours.Ssyrk(ul, tr, n, k, 1.1, a, lda, -0.3, cw, ldc)
			stock.Ssyrk(ul, tr, n, k, 1.1, a, lda, -0.3, cg, ldc)
			almostEqual32(t, "Ssyrk", cw, cg, 1e-3)
		}
	}
	// Ssymm left/right × uplo.
	m, n2 := 32, 24
	for _, s := range sides {
		for _, ul := range uplos {
			d := m
			if s == blas.Right {
				d = n2
			}
			lda := d + 1
			a := randSlice32(r, d*lda)
			ldb, ldc := n2+1, n2+2
			b := randSlice32(r, m*ldb)
			cw := randSlice32(r, m*ldc)
			cg := append([]float32(nil), cw...)
			ours.Ssymm(s, ul, m, n2, 0.8, a, lda, b, ldb, 0.6, cw, ldc)
			stock.Ssymm(s, ul, m, n2, 0.8, a, lda, b, ldb, 0.6, cg, ldc)
			almostEqual32(t, "Ssymm", cw, cg, 1e-3)
		}
	}
}

func TestStrsmStrmmStrsvMatchGonum(t *testing.T) {
	r := rand.New(rand.NewSource(133))
	m, n := 32, 20
	for _, s := range sides {
		for _, ul := range uplos {
			for _, tA := range transposes {
				for _, dg := range diags {
					d := m
					if s == blas.Right {
						d = n
					}
					lda := d + 1
					// Diagonally dominant triangle for a well-conditioned solve.
					a := randSlice32(r, d*lda)
					for i := 0; i < d; i++ {
						a[i*lda+i] = float32(3 + r.Float64())
					}
					ldb := n + 1
					bw := randSlice32(r, m*ldb)
					bg := append([]float32(nil), bw...)
					ours.Strsm(s, ul, tA, dg, m, n, 1.0, a, lda, bw, ldb)
					stock.Strsm(s, ul, tA, dg, m, n, 1.0, a, lda, bg, ldb)
					almostEqual32(t, "Strsm", bw, bg, 2e-3)

					bw2 := randSlice32(r, m*ldb)
					bg2 := append([]float32(nil), bw2...)
					ours.Strmm(s, ul, tA, dg, m, n, 0.9, a, lda, bw2, ldb)
					stock.Strmm(s, ul, tA, dg, m, n, 0.9, a, lda, bg2, ldb)
					almostEqual32(t, "Strmm", bw2, bg2, 2e-3)
				}
			}
		}
	}
	// Strsv: single RHS, all uplo×trans×diag.
	nn := 50
	lda := nn + 1
	a := randSlice32(r, nn*lda)
	for i := 0; i < nn; i++ {
		a[i*lda+i] = float32(3 + r.Float64())
	}
	for _, ul := range uplos {
		for _, tA := range transposes {
			for _, dg := range diags {
				xw := randSlice32(r, nn)
				xg := append([]float32(nil), xw...)
				ours.Strsv(ul, tA, dg, nn, a, lda, xw, 1)
				stock.Strsv(ul, tA, dg, nn, a, lda, xg, 1)
				almostEqual32(t, "Strsv", xw, xg, 2e-3)
			}
		}
	}
}

func TestSL1MatchesGonum(t *testing.T) {
	r := rand.New(rand.NewSource(134))
	n := 257
	x := randSlice32(r, n)
	y := randSlice32(r, n)
	if got, want := ours.Sdot(n, x, 1, y, 1), stock.Sdot(n, x, 1, y, 1); math.Abs(float64(got-want)) > 1e-3*(1+math.Abs(float64(want))) {
		t.Errorf("Sdot: got %v want %v", got, want)
	}
	yw := append([]float32(nil), y...)
	yg := append([]float32(nil), y...)
	ours.Saxpy(n, 1.5, x, 1, yw, 1)
	stock.Saxpy(n, 1.5, x, 1, yg, 1)
	almostEqual32(t, "Saxpy", yw, yg, 1e-4)
	xw := append([]float32(nil), x...)
	xg := append([]float32(nil), x...)
	ours.Sscal(n, 0.3, xw, 1)
	stock.Sscal(n, 0.3, xg, 1)
	almostEqual32(t, "Sscal", xw, xg, 1e-4)
}
