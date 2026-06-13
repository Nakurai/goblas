package blasadapt

import (
	"math"
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/blas"
	gonumblas "gonum.org/v1/gonum/blas/gonum"
)

// The adapter must agree with Gonum's own row-major BLAS on identical inputs.
// Any error in the row→column relabeling rules shows up immediately here.

var (
	ours  = Implementation{}
	stock = gonumblas.Implementation{}
)

func randSlice(r *rand.Rand, n int) []float64 {
	s := make([]float64, n)
	for i := range s {
		s[i] = r.NormFloat64()
	}
	return s
}

func almostEqual(t *testing.T, name string, got, want []float64, tol float64) {
	t.Helper()
	for i := range want {
		if math.Abs(got[i]-want[i]) > tol*(1+math.Abs(want[i])) {
			t.Fatalf("%s: idx=%d got %v want %v", name, i, got[i], want[i])
		}
	}
}

var transposes = []blas.Transpose{blas.NoTrans, blas.Trans}
var uplos = []blas.Uplo{blas.Upper, blas.Lower}
var sides = []blas.Side{blas.Left, blas.Right}
var diags = []blas.Diag{blas.NonUnit, blas.Unit}

func TestDgemmMatchesGonum(t *testing.T) {
	r := rand.New(rand.NewSource(30))
	for _, d := range []struct{ m, n, k int }{{1, 1, 1}, {7, 5, 9}, {32, 17, 23}, {64, 64, 64}} {
		for _, tA := range transposes {
			for _, tB := range transposes {
				// Row-major dims: op(A) is m x k, op(B) is k x n.
				rA, cA := d.m, d.k
				if tA == blas.Trans {
					rA, cA = d.k, d.m
				}
				rB, cB := d.k, d.n
				if tB == blas.Trans {
					rB, cB = d.n, d.k
				}
				lda, ldb, ldc := cA+1, cB+2, d.n+1
				a := randSlice(r, rA*lda)
				b := randSlice(r, rB*ldb)
				cw := randSlice(r, d.m*ldc)
				cg := make([]float64, len(cw))
				copy(cg, cw)

				stock.Dgemm(tA, tB, d.m, d.n, d.k, 1.3, a, lda, b, ldb, -0.5, cw, ldc)
				ours.Dgemm(tA, tB, d.m, d.n, d.k, 1.3, a, lda, b, ldb, -0.5, cg, ldc)
				almostEqual(t, "dgemm", cg, cw, 1e-10)
			}
		}
	}
}

func TestDgemvMatchesGonum(t *testing.T) {
	r := rand.New(rand.NewSource(31))
	for _, d := range []struct{ m, n int }{{1, 1}, {7, 5}, {33, 17}} {
		for _, tA := range transposes {
			lenX, lenY := d.n, d.m
			if tA == blas.Trans {
				lenX, lenY = d.m, d.n
			}
			lda := d.n + 1 // row-major: lda >= cols
			a := randSlice(r, d.m*lda)
			x := randSlice(r, lenX)
			yw := randSlice(r, lenY)
			yg := make([]float64, len(yw))
			copy(yg, yw)

			stock.Dgemv(tA, d.m, d.n, 0.9, a, lda, x, 1, 1.1, yw, 1)
			ours.Dgemv(tA, d.m, d.n, 0.9, a, lda, x, 1, 1.1, yg, 1)
			almostEqual(t, "dgemv", yg, yw, 1e-10)
		}
	}
}

func TestDgerMatchesGonum(t *testing.T) {
	r := rand.New(rand.NewSource(35))
	for _, d := range []struct{ m, n int }{{1, 1}, {7, 5}, {33, 17}} {
		lda := d.n + 1 // row-major: lda >= cols
		x := randSlice(r, d.m)
		y := randSlice(r, d.n)
		aw := randSlice(r, d.m*lda)
		ag := make([]float64, len(aw))
		copy(ag, aw)

		stock.Dger(d.m, d.n, 1.3, x, 1, y, 1, aw, lda)
		ours.Dger(d.m, d.n, 1.3, x, 1, y, 1, ag, lda)
		almostEqual(t, "dger", ag, aw, 1e-10)
	}
}

func TestDtrsvMatchesGonum(t *testing.T) {
	r := rand.New(rand.NewSource(36))
	for _, n := range []int{1, 5, 17, 40} {
		for _, ul := range uplos {
			for _, tA := range transposes {
				for _, dg := range diags {
					lda := n + 1
					// Well-conditioned triangular A (row-major) in the ul triangle.
					a := make([]float64, n*lda)
					scale := 0.4 / math.Sqrt(float64(n))
					for i := 0; i < n; i++ {
						for j := 0; j < n; j++ {
							if (j > i) == (ul == blas.Upper) || i == j {
								a[i*lda+j] = scale * r.NormFloat64()
							}
						}
						a[i*lda+i] = 3 + math.Abs(r.NormFloat64())
					}
					xw := randSlice(r, n)
					xg := make([]float64, len(xw))
					copy(xg, xw)

					stock.Dtrsv(ul, tA, dg, n, a, lda, xw, 1)
					ours.Dtrsv(ul, tA, dg, n, a, lda, xg, 1)
					almostEqual(t, "dtrsv", xg, xw, 1e-9)
				}
			}
		}
	}
}

func TestDsyrkMatchesGonum(t *testing.T) {
	r := rand.New(rand.NewSource(32))
	for _, d := range []struct{ n, k int }{{1, 1}, {7, 5}, {33, 17}, {64, 40}} {
		for _, ul := range uplos {
			for _, tr := range transposes {
				rA, cA := d.n, d.k
				if tr == blas.Trans {
					rA, cA = d.k, d.n
				}
				lda, ldc := cA+1, d.n+2
				a := randSlice(r, rA*lda)
				cw := randSlice(r, d.n*ldc)
				cg := make([]float64, len(cw))
				copy(cg, cw)

				stock.Dsyrk(ul, tr, d.n, d.k, 1.2, a, lda, 0.7, cw, ldc)
				ours.Dsyrk(ul, tr, d.n, d.k, 1.2, a, lda, 0.7, cg, ldc)
				almostEqual(t, "dsyrk", cg, cw, 1e-9)
			}
		}
	}
}

func TestDtrsmMatchesGonum(t *testing.T) {
	r := rand.New(rand.NewSource(33))
	for _, d := range []struct{ m, n int }{{1, 1}, {7, 5}, {33, 17}, {64, 40}} {
		for _, s := range sides {
			for _, ul := range uplos {
				for _, tA := range transposes {
					for _, dg := range diags {
						ta := d.n
						if s == blas.Left {
							ta = d.m
						}
						lda, ldb := ta+1, d.n+2
						// Well-conditioned triangular A (row-major).
						a := make([]float64, ta*lda)
						scale := 0.5 / math.Sqrt(float64(ta))
						for i := 0; i < ta; i++ {
							for j := 0; j < ta; j++ {
								if (j > i) == (ul == blas.Upper) || i == j {
									a[i*lda+j] = scale * r.NormFloat64()
								}
							}
							a[i*lda+i] = 3 + math.Abs(r.NormFloat64())
						}
						bw := randSlice(r, d.m*ldb)
						bg := make([]float64, len(bw))
						copy(bg, bw)

						stock.Dtrsm(s, ul, tA, dg, d.m, d.n, 1.4, a, lda, bw, ldb)
						ours.Dtrsm(s, ul, tA, dg, d.m, d.n, 1.4, a, lda, bg, ldb)
						almostEqual(t, "dtrsm", bg, bw, 1e-8)
					}
				}
			}
		}
	}
}

func TestDsymmDtrmmMatchGonum(t *testing.T) {
	r := rand.New(rand.NewSource(34))
	for _, d := range []struct{ m, n int }{{1, 1}, {7, 5}, {33, 17}} {
		for _, s := range sides {
			for _, ul := range uplos {
				ta := d.n
				if s == blas.Left {
					ta = d.m
				}
				lda, ldb, ldc := ta+1, d.n+1, d.n+2

				a := randSlice(r, ta*lda)
				b := randSlice(r, d.m*ldb)
				cw := randSlice(r, d.m*ldc)
				cg := make([]float64, len(cw))
				copy(cg, cw)

				stock.Dsymm(s, ul, d.m, d.n, 1.1, a, lda, b, ldb, 0.6, cw, ldc)
				ours.Dsymm(s, ul, d.m, d.n, 1.1, a, lda, b, ldb, 0.6, cg, ldc)
				almostEqual(t, "dsymm", cg, cw, 1e-10)

				for _, tA := range transposes {
					for _, dg := range diags {
						bw := randSlice(r, d.m*ldb)
						bg2 := make([]float64, len(bw))
						copy(bg2, bw)
						stock.Dtrmm(s, ul, tA, dg, d.m, d.n, 0.8, a, lda, bw, ldb)
						ours.Dtrmm(s, ul, tA, dg, d.m, d.n, 0.8, a, lda, bg2, ldb)
						almostEqual(t, "dtrmm", bg2, bw, 1e-9)
					}
				}
			}
		}
	}
}
