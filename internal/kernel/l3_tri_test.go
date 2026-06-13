package kernel

import (
	"math"
	"math/rand"
	"testing"
)

// activeForTest returns the kernels to verify: generic everywhere, plus the
// platform kernel when it differs (covered by arm64-specific selection).
func kernelsUnderTest() map[string]Kernel {
	ks := map[string]Kernel{"generic": genericKernel{}}
	for name, k := range platformKernels() {
		ks[name] = k
	}
	return ks
}

// triMatrix builds a well-conditioned n x n triangular matrix (diagonally
// dominant so solves are stable) with leading dimension lda.
func triMatrix(r *rand.Rand, n, lda int, upper, unit bool) []float64 {
	a := make([]float64, lda*n)
	// Small off-diagonal entries keep the matrix well-conditioned even with a
	// unit diagonal (random unit-triangular matrices are exponentially
	// ill-conditioned otherwise, which would defeat residual checks).
	scale := 0.5 / math.Sqrt(float64(n))
	for j := 0; j < n; j++ {
		lo, hi := j, n
		if upper {
			lo, hi = 0, j+1
		}
		for i := lo; i < hi; i++ {
			a[i+j*lda] = scale * r.NormFloat64()
		}
		// Dominant diagonal keeps inverse well-behaved. For unit-diagonal
		// solves the stored diagonal is ignored, so leave the random value.
		if !unit {
			a[j+j*lda] = 4 + math.Abs(r.NormFloat64())
		}
	}
	return a
}

// denseTri materializes the full dense matrix that op(A) represents,
// honoring the implicit unit diagonal, as an n x n column-major matrix.
func denseTri(a []float64, lda, n int, upper, unit bool) []float64 {
	d := make([]float64, n*n)
	for j := 0; j < n; j++ {
		lo, hi := j, n
		if upper {
			lo, hi = 0, j+1
		}
		for i := lo; i < hi; i++ {
			d[i+j*n] = a[i+j*lda]
		}
		if unit {
			d[j+j*n] = 1
		}
	}
	return d
}

func TestDtrsmSolvesSystem(t *testing.T) {
	r := rand.New(rand.NewSource(20))
	for name, k := range kernelsUnderTest() {
		for _, d := range []struct{ m, n int }{{1, 1}, {5, 3}, {16, 8}, {33, 17}, {64, 64}, {100, 37}} {
			for _, left := range []bool{true, false} {
				for _, upper := range []bool{true, false} {
					for _, trans := range []bool{true, false} {
						for _, unit := range []bool{true, false} {
							ta := d.n
							if left {
								ta = d.m
							}
							lda := ta + 2
							ldb := d.m + 1
							alpha := 1.5

							a := triMatrix(r, ta, lda, upper, unit)
							b := make([]float64, ldb*d.n)
							for i := range b {
								b[i] = r.NormFloat64()
							}
							borig := make([]float64, len(b))
							copy(borig, b)

							k.Dtrsm(left, upper, trans, unit, d.m, d.n, alpha, a, lda, b, ldb)

							// Residual: op(A)*X (or X*op(A)) must equal alpha*B.
							da := denseTri(a, lda, ta, upper, unit)
							got := make([]float64, ldb*d.n)
							if left {
								dgemmNaive(trans, false, d.m, d.n, d.m, 1, da, ta, b, ldb, 0, got, ldb)
							} else {
								dgemmNaive(false, trans, d.m, d.n, d.n, 1, b, ldb, da, ta, 0, got, ldb)
							}
							for j := 0; j < d.n; j++ {
								for i := 0; i < d.m; i++ {
									want := alpha * borig[i+j*ldb]
									if math.Abs(got[i+j*ldb]-want) > 1e-8*(1+math.Abs(want)) {
										t.Fatalf("%s trsm m=%d n=%d L=%v U=%v T=%v unit=%v (%d,%d): residual %v want %v",
											name, d.m, d.n, left, upper, trans, unit, i, j, got[i+j*ldb], want)
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

func TestDsyrkMatchesDense(t *testing.T) {
	r := rand.New(rand.NewSource(21))
	for name, k := range kernelsUnderTest() {
		for _, d := range []struct{ n, k int }{{1, 1}, {5, 7}, {17, 9}, {64, 33}, {100, 64}} {
			for _, upper := range []bool{true, false} {
				for _, trans := range []bool{true, false} {
					rowsA, colsA := d.n, d.k
					if trans {
						rowsA, colsA = d.k, d.n
					}
					lda := rowsA + 1
					ldc := d.n + 2
					alpha, beta := 1.3, -0.4

					a := make([]float64, lda*colsA)
					for i := range a {
						a[i] = r.NormFloat64()
					}
					c := make([]float64, ldc*d.n)
					for i := range c {
						c[i] = r.NormFloat64()
					}
					// Dense reference: full C' = alpha*op(A)*op(A)^T + beta*C.
					want := make([]float64, len(c))
					copy(want, c)
					dgemmNaive(trans, !trans, d.n, d.n, d.k, alpha, a, lda, a, lda, beta, want, ldc)

					k.Dsyrk(upper, trans, d.n, d.k, alpha, a, lda, beta, c, ldc)

					for j := 0; j < d.n; j++ {
						lo, hi := j, d.n
						if upper {
							lo, hi = 0, j+1
						}
						for i := lo; i < hi; i++ {
							if math.Abs(c[i+j*ldc]-want[i+j*ldc]) > 1e-9*(1+math.Abs(want[i+j*ldc])) {
								t.Fatalf("%s syrk n=%d k=%d U=%v T=%v (%d,%d): got %v want %v",
									name, d.n, d.k, upper, trans, i, j, c[i+j*ldc], want[i+j*ldc])
							}
						}
					}
				}
			}
		}
	}
}

func TestDsymmMatchesDense(t *testing.T) {
	r := rand.New(rand.NewSource(22))
	for name, g := range kernelsUnderTest() {
		for _, d := range []struct{ m, n int }{{1, 1}, {5, 3}, {17, 9}, {40, 25}, {128, 96}, {200, 64}} {
			for _, left := range []bool{true, false} {
				for _, upper := range []bool{true, false} {
					ta := d.n
					if left {
						ta = d.m
					}
					lda := ta + 1
					ldb := d.m + 2
					ldc := d.m + 3
					alpha, beta := 0.7, 1.1

					// Symmetric A stored in one triangle; densify for reference.
					a := make([]float64, lda*ta)
					da := make([]float64, ta*ta)
					r2 := rand.New(rand.NewSource(int64(d.m*100 + d.n)))
					for j := 0; j < ta; j++ {
						for i := 0; i <= j; i++ {
							v := r2.NormFloat64()
							da[i+j*ta], da[j+i*ta] = v, v
							if upper {
								a[i+j*lda] = v
							} else {
								a[j+i*lda] = v
							}
						}
					}
					b := make([]float64, ldb*d.n)
					for i := range b {
						b[i] = r.NormFloat64()
					}
					c := make([]float64, ldc*d.n)
					want := make([]float64, len(c))
					for i := range c {
						c[i] = r.NormFloat64()
						want[i] = c[i]
					}
					if left {
						dgemmNaive(false, false, d.m, d.n, d.m, alpha, da, ta, b, ldb, beta, want, ldc)
					} else {
						dgemmNaive(false, false, d.m, d.n, d.n, alpha, b, ldb, da, ta, beta, want, ldc)
					}

					g.Dsymm(left, upper, d.m, d.n, alpha, a, lda, b, ldb, beta, c, ldc)

					for j := 0; j < d.n; j++ {
						for i := 0; i < d.m; i++ {
							if math.Abs(c[i+j*ldc]-want[i+j*ldc]) > 1e-10*(1+math.Abs(want[i+j*ldc])) {
								t.Fatalf("%s symm m=%d n=%d L=%v U=%v (%d,%d): got %v want %v",
									name, d.m, d.n, left, upper, i, j, c[i+j*ldc], want[i+j*ldc])
							}
						}
					}
				}
			}
		}
	}
}

func TestDtrmmMatchesDense(t *testing.T) {
	r := rand.New(rand.NewSource(23))
	for name, g := range kernelsUnderTest() {
		for _, d := range []struct{ m, n int }{{1, 1}, {5, 3}, {17, 9}, {40, 25}, {128, 96}, {200, 64}} {
			for _, left := range []bool{true, false} {
				for _, upper := range []bool{true, false} {
					for _, trans := range []bool{true, false} {
						for _, unit := range []bool{true, false} {
							ta := d.n
							if left {
								ta = d.m
							}
							lda := ta + 2
							ldb := d.m + 1
							alpha := 1.4

							a := triMatrix(r, ta, lda, upper, unit)
							da := denseTri(a, lda, ta, upper, unit)
							b := make([]float64, ldb*d.n)
							for i := range b {
								b[i] = r.NormFloat64()
							}
							want := make([]float64, len(b))
							if left {
								dgemmNaive(trans, false, d.m, d.n, d.m, alpha, da, ta, b, ldb, 0, want, ldb)
							} else {
								dgemmNaive(false, trans, d.m, d.n, d.n, alpha, b, ldb, da, ta, 0, want, ldb)
							}

							g.Dtrmm(left, upper, trans, unit, d.m, d.n, alpha, a, lda, b, ldb)

							for j := 0; j < d.n; j++ {
								for i := 0; i < d.m; i++ {
									if math.Abs(b[i+j*ldb]-want[i+j*ldb]) > 1e-9*(1+math.Abs(want[i+j*ldb])) {
										t.Fatalf("%s trmm m=%d n=%d L=%v U=%v T=%v unit=%v (%d,%d): got %v want %v",
											name, d.m, d.n, left, upper, trans, unit, i, j, b[i+j*ldb], want[i+j*ldb])
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

func TestDgerDtrsv(t *testing.T) {
	r := rand.New(rand.NewSource(24))
	for name, k := range kernelsUnderTest() {
		// Dger: A += alpha*x*y^T. incX=1 exercises the NEON daxpy-per-column
		// fast path; incX=2 exercises the strided fallback. Sizes span several
		// NEON vector widths plus a tail.
		for _, m := range []int{1, 7, 33, 100} {
			for _, incX := range []int{1, 2} {
				n := 5
				lda := m + 2
				a := make([]float64, lda*n)
				want := make([]float64, len(a))
				for i := range a {
					a[i] = r.NormFloat64()
					want[i] = a[i]
				}
				x := make([]float64, m*incX)
				y := make([]float64, n)
				for i := range x {
					x[i] = r.NormFloat64()
				}
				for i := range y {
					y[i] = r.NormFloat64()
				}
				alpha := 1.7
				for j := 0; j < n; j++ {
					for i := 0; i < m; i++ {
						want[i+j*lda] += alpha * x[i*incX] * y[j]
					}
				}
				k.Dger(m, n, alpha, x, incX, y, 1, a, lda)
				for i := range a {
					if math.Abs(a[i]-want[i]) > 1e-12*(1+math.Abs(want[i])) {
						t.Fatalf("%s ger m=%d incX=%d idx=%d: got %v want %v", name, m, incX, i, a[i], want[i])
					}
				}
			}
		}

		// Dtrsv: residual check op(A)*x == b across all flag combos. Sizes past
		// the NEON vector width exercise the daxpy/ddot spans plus their tails.
		for _, nn := range []int{1, 4, 13, 40, 100} {
			for _, upper := range []bool{true, false} {
				for _, trans := range []bool{true, false} {
					for _, unit := range []bool{true, false} {
						lda := nn + 1
						ta := triMatrix(r, nn, lda, upper, unit)
						da := denseTri(ta, lda, nn, upper, unit)
						b := make([]float64, nn)
						for i := range b {
							b[i] = r.NormFloat64()
						}
						xv := make([]float64, nn)
						copy(xv, b)
						k.Dtrsv(upper, trans, unit, nn, ta, lda, xv, 1)
						// Verify op(A)*x == b.
						got := make([]float64, nn)
						dgemmNaive(trans, false, nn, 1, nn, 1, da, nn, xv, nn, 0, got, nn)
						for i := range b {
							if math.Abs(got[i]-b[i]) > 1e-8*(1+math.Abs(b[i])) {
								t.Fatalf("%s trsv n=%d U=%v T=%v unit=%v i=%d: residual %v want %v",
									name, nn, upper, trans, unit, i, got[i], b[i])
							}
						}
					}
				}
			}
		}
	}
}

// triSink defeats dead-code elimination in the benchmarks below.
var triSink float64

func triBenchSlice(seed int64, n int) []float64 {
	r := rand.New(rand.NewSource(seed))
	s := make([]float64, n)
	for i := range s {
		s[i] = r.NormFloat64()
	}
	return s
}

// BenchmarkDsymm / BenchmarkDtrmm report GFLOPS for each available kernel
// (generic plus the platform kernel) so the Phase-13 Dgemm-routing speedup is
// reproducible. Run e.g.:
//
//	go test -run '^$' -bench 'Dsymm|Dtrmm' -benchtime=2s ./internal/kernel/
func BenchmarkDsymm(b *testing.B) {
	const n = 1024
	a := triBenchSlice(1, n*n)
	bb := triBenchSlice(2, n*n)
	c := make([]float64, n*n)
	flops := 2 * float64(n) * float64(n) * float64(n) // C = A*B, A symmetric m x m
	for name, k := range kernelsUnderTest() {
		b.Run(name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				k.Dsymm(true, true, n, n, 1, a, n, bb, n, 0, c, n)
				triSink += c[0]
			}
			b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
		})
	}
}

func BenchmarkDtrmm(b *testing.B) {
	const n = 1024
	a := triBenchSlice(1, n*n)
	bb := triBenchSlice(2, n*n)
	work := make([]float64, n*n)
	flops := float64(n) * float64(n) * float64(n) // B = A*B, A triangular (~half of gemm)
	for name, k := range kernelsUnderTest() {
		b.Run(name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				copy(work, bb) // Dtrmm is in place; restore B each iteration
				k.Dtrmm(true, true, false, false, n, n, 1, a, n, work, n)
				triSink += work[0]
			}
			b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
		})
	}
}

// BenchmarkDger / BenchmarkDtrsv report GFLOPS per kernel (generic plus the
// platform kernel) so the Phase-14 L2 speedup is reproducible. Run e.g.:
//
//	go test -run '^$' -bench 'Dger|Dtrsv' -benchtime=1s ./internal/kernel/
func BenchmarkDger(b *testing.B) {
	const m, n = 2048, 2048
	a := triBenchSlice(1, m*n)
	x := triBenchSlice(2, m)
	y := triBenchSlice(3, n)
	flops := 2 * float64(m) * float64(n)
	for name, k := range kernelsUnderTest() {
		b.Run(name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				k.Dger(m, n, 1.0001, x, 1, y, 1, a, m)
				triSink += a[0]
			}
			b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
		})
	}
}

func BenchmarkDtrsv(b *testing.B) {
	const n = 4096
	a := make([]float64, n*n)
	rr := rand.New(rand.NewSource(2))
	for j := 0; j < n; j++ {
		for i := j; i < n; i++ {
			a[i+j*n] = 0.1 * rr.NormFloat64()
		}
		a[j+j*n] = 4 + math.Abs(rr.NormFloat64()) // diagonally dominant
	}
	x0 := triBenchSlice(3, n)
	x := make([]float64, n)
	flops := float64(n) * float64(n) // ~one multiply-add per below-diagonal entry
	for name, k := range kernelsUnderTest() {
		b.Run(name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				copy(x, x0)
				k.Dtrsv(false, false, false, n, a, n, x, 1)
				triSink += x[0]
			}
			b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
		})
	}
}
