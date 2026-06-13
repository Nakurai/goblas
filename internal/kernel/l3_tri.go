package kernel

// This file implements the triangular and symmetric Level-3 routines.
//
// Dsyrk and Dtrsm use recursive blocking: the triangular matrix is split in
// half, the off-diagonal block update becomes a Dgemm (where the FLOPs are),
// and the two diagonal blocks recurse down to a small column-oriented base
// case. The recursion is parameterized by the gemm to use, so the generic
// kernel feeds it the portable blocked gemm and the NEON kernel feeds it the
// assembly-backed one — the heavy lifting always lands on the fastest gemm
// available. Dsymm and Dtrmm are reference column-oriented loops (they are
// rarely hot in LAPACK workloads; blocked variants can come later).

// dgemmFunc is the signature shared by every Dgemm implementation.
type dgemmFunc func(transA, transB bool, m, n, k int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int)

// triBase is the cutoff below which Dsyrk/Dtrsm stop recursing and run the
// column-oriented base case.
const triBase = 32

// --- generic kernel methods ---

func (g genericKernel) Dsyrk(upper, trans bool, n, k int, alpha float64, a []float64, lda int, beta float64, c []float64, ldc int) {
	dsyrkRec(g.Dgemm, upper, trans, n, k, alpha, a, lda, beta, c, ldc)
}

func (g genericKernel) Dtrsm(left, upper, transA, unit bool, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int) {
	dtrsmRec(g.Dgemm, left, upper, transA, unit, m, n, alpha, a, lda, b, ldb)
}

func (genericKernel) Dsymm(left, upper bool, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int) {
	dsymmRef(left, upper, m, n, alpha, a, lda, b, ldb, beta, c, ldc)
}

func (genericKernel) Dtrmm(left, upper, transA, unit bool, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int) {
	dtrmmRef(left, upper, transA, unit, m, n, alpha, a, lda, b, ldb)
}

// --- Dsyrk: C = alpha*A*A^T + beta*C (or A^T*A), one triangle of C ---

// dsyrkRec recursively splits C into [C11 C12; C21 C22]. The diagonal blocks
// are syrk subproblems; the off-diagonal block in the stored triangle is a
// plain gemm between the two halves of A.
func dsyrkRec(gemm dgemmFunc, upper, trans bool, n, k int, alpha float64, a []float64, lda int, beta float64, c []float64, ldc int) {
	if n == 0 {
		return
	}
	if n <= triBase {
		dsyrkBase(upper, trans, n, k, alpha, a, lda, beta, c, ldc)
		return
	}
	n1 := n / 2
	n2 := n - n1

	// Halves of A: rows (trans=false) or columns (trans=true) [0,n1) and [n1,n).
	var a1, a2 []float64
	if !trans {
		a1, a2 = a, a[n1:] // A is n x k: split rows
	} else {
		a1, a2 = a, a[n1*lda:] // A is k x n: split columns
	}

	dsyrkRec(gemm, upper, trans, n1, k, alpha, a1, lda, beta, c, ldc)
	dsyrkRec(gemm, upper, trans, n2, k, alpha, a2, lda, beta, c[n1+n1*ldc:], ldc)
	if !upper {
		// C21 (n2 x n1) = alpha * op2(A2) * op1(A1)^T + beta*C21.
		if !trans {
			gemm(false, true, n2, n1, k, alpha, a2, lda, a1, lda, beta, c[n1:], ldc)
		} else {
			gemm(true, false, n2, n1, k, alpha, a2, lda, a1, lda, beta, c[n1:], ldc)
		}
	} else {
		// C12 (n1 x n2) = alpha * op1(A1) * op2(A2)^T + beta*C12.
		if !trans {
			gemm(false, true, n1, n2, k, alpha, a1, lda, a2, lda, beta, c[n1*ldc:], ldc)
		} else {
			gemm(true, false, n1, n2, k, alpha, a1, lda, a2, lda, beta, c[n1*ldc:], ldc)
		}
	}
}

// dsyrkBase is the column-oriented reference syrk for small n.
func dsyrkBase(upper, trans bool, n, k int, alpha float64, a []float64, lda int, beta float64, c []float64, ldc int) {
	// Scale the stored triangle of C by beta.
	for j := 0; j < n; j++ {
		lo, hi := 0, j+1 // upper: rows [0, j]
		if !upper {
			lo, hi = j, n // lower: rows [j, n)
		}
		col := c[j*ldc:]
		if beta == 0 {
			for i := lo; i < hi; i++ {
				col[i] = 0
			}
		} else if beta != 1 {
			for i := lo; i < hi; i++ {
				col[i] *= beta
			}
		}
	}
	if alpha == 0 || k == 0 {
		return
	}

	if !trans {
		// C += alpha * A * A^T, A is n x k: rank-1 updates per column of A.
		for l := 0; l < k; l++ {
			al := a[l*lda:]
			for j := 0; j < n; j++ {
				f := alpha * al[j]
				if f == 0 {
					continue
				}
				col := c[j*ldc:]
				if upper {
					for i := 0; i <= j; i++ {
						col[i] += f * al[i]
					}
				} else {
					for i := j; i < n; i++ {
						col[i] += f * al[i]
					}
				}
			}
		}
		return
	}
	// C += alpha * A^T * A, A is k x n: C(i,j) += alpha * dot(A(:,i), A(:,j)).
	for j := 0; j < n; j++ {
		aj := a[j*lda : j*lda+k]
		col := c[j*ldc:]
		lo, hi := 0, j+1
		if !upper {
			lo, hi = j, n
		}
		for i := lo; i < hi; i++ {
			ai := a[i*lda : i*lda+k]
			var s float64
			for l := range aj {
				s += ai[l] * aj[l]
			}
			col[i] += alpha * s
		}
	}
}

// --- Dtrsm: solve op(A)X = alpha*B or X op(A) = alpha*B in place ---

// dtrsmRec recursively splits the triangular matrix A into
// [A11 A12; A21 A22] (A12 = 0 for lower, A21 = 0 for upper). One diagonal
// block is solved first, the off-diagonal block update is a gemm against the
// not-yet-solved part of B, then the other diagonal block is solved.
func dtrsmRec(gemm dgemmFunc, left, upper, transA, unit bool, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int) {
	if m == 0 || n == 0 {
		return
	}
	d := m
	if !left {
		d = n
	}
	if d <= triBase {
		dtrsmBase(left, upper, transA, unit, m, n, alpha, a, lda, b, ldb)
		return
	}
	d1 := d / 2
	d2 := d - d1

	a11 := a
	a22 := a[d1+d1*lda:]
	aOff := a[d1:]      // A21 (lower): d2 x d1
	if upper {
		aOff = a[d1*lda:] // A12 (upper): d1 x d2
	}

	if left {
		b1 := b           // rows [0, d1)
		b2 := b[d1:]      // rows [d1, m)
		// Solve order depends on which off-diagonal block is nonzero and
		// whether A is transposed: solve the block whose equation involves
		// only itself first.
		solveFirst1 := upper == transA // lower+notrans or upper+trans: top first
		if solveFirst1 {
			dtrsmRec(gemm, left, upper, transA, unit, d1, n, alpha, a11, lda, b1, ldb)
			// B2 = alpha*B2 - op(Aoff)*X1, where op(Aoff) is d2 x d1.
			if !transA {
				gemm(false, false, d2, n, d1, -1, aOff, lda, b1, ldb, alpha, b2, ldb)
			} else {
				gemm(true, false, d2, n, d1, -1, aOff, lda, b1, ldb, alpha, b2, ldb)
			}
			dtrsmRec(gemm, left, upper, transA, unit, d2, n, 1, a22, lda, b2, ldb)
		} else {
			dtrsmRec(gemm, left, upper, transA, unit, d2, n, alpha, a22, lda, b2, ldb)
			// B1 = alpha*B1 - op(Aoff)*X2, op(Aoff) is d1 x d2.
			if !transA {
				gemm(false, false, d1, n, d2, -1, aOff, lda, b2, ldb, alpha, b1, ldb)
			} else {
				gemm(true, false, d1, n, d2, -1, aOff, lda, b2, ldb, alpha, b1, ldb)
			}
			dtrsmRec(gemm, left, upper, transA, unit, d1, n, 1, a11, lda, b1, ldb)
		}
		return
	}

	// Right side: X op(A) = alpha*B; split B by columns.
	b1 := b           // cols [0, d1)
	b2 := b[d1*ldb:]  // cols [d1, n)
	solveFirst1 := upper != transA // upper+notrans or lower+trans: left cols first
	if solveFirst1 {
		dtrsmRec(gemm, left, upper, transA, unit, m, d1, alpha, a11, lda, b1, ldb)
		// B2 = alpha*B2 - X1*op(Aoff), op(Aoff) is d1 x d2.
		if !transA {
			gemm(false, false, m, d2, d1, -1, b1, ldb, aOff, lda, alpha, b2, ldb)
		} else {
			gemm(false, true, m, d2, d1, -1, b1, ldb, aOff, lda, alpha, b2, ldb)
		}
		dtrsmRec(gemm, left, upper, transA, unit, m, d2, 1, a22, lda, b2, ldb)
	} else {
		dtrsmRec(gemm, left, upper, transA, unit, m, d2, alpha, a22, lda, b2, ldb)
		// B1 = alpha*B1 - X2*op(Aoff), op(Aoff) is d2 x d1.
		if !transA {
			gemm(false, false, m, d1, d2, -1, b2, ldb, aOff, lda, alpha, b1, ldb)
		} else {
			gemm(false, true, m, d1, d2, -1, b2, ldb, aOff, lda, alpha, b1, ldb)
		}
		dtrsmRec(gemm, left, upper, transA, unit, m, d1, 1, a11, lda, b1, ldb)
	}
}

// dtrsmBase is the column-oriented reference trsm (netlib structure).
func dtrsmBase(left, upper, transA, unit bool, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int) {
	if left && !transA {
		// Per-column substitution on op(A) = A.
		for j := 0; j < n; j++ {
			col := b[j*ldb : j*ldb+m]
			if alpha != 1 {
				for i := range col {
					col[i] *= alpha
				}
			}
			if upper {
				for k := m - 1; k >= 0; k-- {
					if col[k] == 0 {
						continue
					}
					if !unit {
						col[k] /= a[k+k*lda]
					}
					ak := a[k*lda:]
					f := col[k]
					for i := 0; i < k; i++ {
						col[i] -= f * ak[i]
					}
				}
			} else {
				for k := 0; k < m; k++ {
					if col[k] == 0 {
						continue
					}
					if !unit {
						col[k] /= a[k+k*lda]
					}
					ak := a[k*lda:]
					f := col[k]
					for i := k + 1; i < m; i++ {
						col[i] -= f * ak[i]
					}
				}
			}
		}
		return
	}
	if left { // transA
		// Solve A^T X = alpha*B: dot-product form.
		for j := 0; j < n; j++ {
			col := b[j*ldb : j*ldb+m]
			if upper {
				// A^T lower: forward.
				for i := 0; i < m; i++ {
					s := alpha * col[i]
					ai := a[i*lda:]
					for k := 0; k < i; k++ {
						s -= ai[k] * col[k]
					}
					if !unit {
						s /= a[i+i*lda]
					}
					col[i] = s
				}
			} else {
				// A^T upper: backward.
				for i := m - 1; i >= 0; i-- {
					s := alpha * col[i]
					ai := a[i*lda:]
					for k := i + 1; k < m; k++ {
						s -= ai[k] * col[k]
					}
					if !unit {
						s /= a[i+i*lda]
					}
					col[i] = s
				}
			}
		}
		return
	}
	if !transA {
		// Right, NoTrans: X*A = alpha*B, column-by-column of B/X.
		if upper {
			for j := 0; j < n; j++ {
				colj := b[j*ldb : j*ldb+m]
				if alpha != 1 {
					for i := range colj {
						colj[i] *= alpha
					}
				}
				aj := a[j*lda:]
				for k := 0; k < j; k++ {
					if f := aj[k]; f != 0 {
						colk := b[k*ldb:]
						for i := range colj {
							colj[i] -= f * colk[i]
						}
					}
				}
				if !unit {
					f := 1 / a[j+j*lda]
					for i := range colj {
						colj[i] *= f
					}
				}
			}
		} else {
			for j := n - 1; j >= 0; j-- {
				colj := b[j*ldb : j*ldb+m]
				if alpha != 1 {
					for i := range colj {
						colj[i] *= alpha
					}
				}
				aj := a[j*lda:]
				for k := j + 1; k < n; k++ {
					if f := aj[k]; f != 0 {
						colk := b[k*ldb:]
						for i := range colj {
							colj[i] -= f * colk[i]
						}
					}
				}
				if !unit {
					f := 1 / a[j+j*lda]
					for i := range colj {
						colj[i] *= f
					}
				}
			}
		}
		return
	}
	// Right, Trans: X*A^T = alpha*B.
	if upper {
		// A^T lower: process columns right-to-left; X(:,j) depends on k > j
		// via A(j,k).
		for j := n - 1; j >= 0; j-- {
			colj := b[j*ldb : j*ldb+m]
			if alpha != 1 {
				for i := range colj {
					colj[i] *= alpha
				}
			}
			for k := j + 1; k < n; k++ {
				if f := a[j+k*lda]; f != 0 {
					colk := b[k*ldb:]
					for i := range colj {
						colj[i] -= f * colk[i]
					}
				}
			}
			if !unit {
				f := 1 / a[j+j*lda]
				for i := range colj {
					colj[i] *= f
				}
			}
		}
	} else {
		// A^T upper: left-to-right; X(:,j) depends on k < j via A(j,k).
		for j := 0; j < n; j++ {
			colj := b[j*ldb : j*ldb+m]
			if alpha != 1 {
				for i := range colj {
					colj[i] *= alpha
				}
			}
			for k := 0; k < j; k++ {
				if f := a[j+k*lda]; f != 0 {
					colk := b[k*ldb:]
					for i := range colj {
						colj[i] -= f * colk[i]
					}
				}
			}
			if !unit {
				f := 1 / a[j+j*lda]
				for i := range colj {
					colj[i] *= f
				}
			}
		}
	}
}

// --- Dsymm: C = alpha*A*B + beta*C with symmetric A (one triangle stored) ---

// symAt reads symmetric A(i,j) from the stored triangle.
func symAt(a []float64, lda, i, j int, upper bool) float64 {
	if (i <= j) == upper || i == j {
		return a[i+j*lda]
	}
	return a[j+i*lda]
}

func dsymmRef(left, upper bool, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int) {
	// C = beta*C.
	for j := 0; j < n; j++ {
		col := c[j*ldc : j*ldc+m]
		if beta == 0 {
			for i := range col {
				col[i] = 0
			}
		} else if beta != 1 {
			for i := range col {
				col[i] *= beta
			}
		}
	}
	if alpha == 0 {
		return
	}
	if left {
		// C += alpha * A(sym m x m) * B.
		for j := 0; j < n; j++ {
			cc := c[j*ldc : j*ldc+m]
			bc := b[j*ldb:]
			for k := 0; k < m; k++ {
				f := alpha * bc[k]
				if f == 0 {
					continue
				}
				for i := 0; i < m; i++ {
					cc[i] += f * symAt(a, lda, i, k, upper)
				}
			}
		}
		return
	}
	// C += alpha * B * A(sym n x n).
	for j := 0; j < n; j++ {
		cc := c[j*ldc : j*ldc+m]
		for k := 0; k < n; k++ {
			f := alpha * symAt(a, lda, k, j, upper)
			if f == 0 {
				continue
			}
			bc := b[k*ldb:]
			for i := 0; i < m; i++ {
				cc[i] += f * bc[i]
			}
		}
	}
}

// --- Dtrmm: B = alpha*op(A)*B or alpha*B*op(A), triangular A, in place ---

func dtrmmRef(left, upper, transA, unit bool, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int) {
	if left && !transA {
		for j := 0; j < n; j++ {
			col := b[j*ldb : j*ldb+m]
			if upper {
				// Ascending k: col[i<k] read original col[k] before overwrite.
				for k := 0; k < m; k++ {
					if col[k] == 0 {
						continue
					}
					f := alpha * col[k]
					ak := a[k*lda:]
					for i := 0; i < k; i++ {
						col[i] += f * ak[i]
					}
					if !unit {
						f *= ak[k]
					}
					col[k] = f
				}
			} else {
				for k := m - 1; k >= 0; k-- {
					if col[k] == 0 {
						continue
					}
					f := alpha * col[k]
					ak := a[k*lda:]
					col[k] = f
					if !unit {
						col[k] *= ak[k]
					}
					for i := k + 1; i < m; i++ {
						col[i] += f * ak[i]
					}
				}
			}
		}
		return
	}
	if left { // transA: B = alpha*A^T*B
		for j := 0; j < n; j++ {
			col := b[j*ldb : j*ldb+m]
			if upper {
				// Descending i: row i of A^T = column i of A.
				for i := m - 1; i >= 0; i-- {
					s := col[i]
					ai := a[i*lda:]
					if !unit {
						s *= ai[i]
					}
					for k := 0; k < i; k++ {
						s += ai[k] * col[k]
					}
					col[i] = alpha * s
				}
			} else {
				for i := 0; i < m; i++ {
					s := col[i]
					ai := a[i*lda:]
					if !unit {
						s *= ai[i]
					}
					for k := i + 1; k < m; k++ {
						s += ai[k] * col[k]
					}
					col[i] = alpha * s
				}
			}
		}
		return
	}
	if !transA {
		// Right: B = alpha*B*A.
		if upper {
			for j := n - 1; j >= 0; j-- {
				colj := b[j*ldb : j*ldb+m]
				f := alpha
				if !unit {
					f *= a[j+j*lda]
				}
				for i := range colj {
					colj[i] *= f
				}
				aj := a[j*lda:]
				for k := 0; k < j; k++ {
					if g := alpha * aj[k]; g != 0 {
						colk := b[k*ldb:]
						for i := range colj {
							colj[i] += g * colk[i]
						}
					}
				}
			}
		} else {
			for j := 0; j < n; j++ {
				colj := b[j*ldb : j*ldb+m]
				f := alpha
				if !unit {
					f *= a[j+j*lda]
				}
				for i := range colj {
					colj[i] *= f
				}
				aj := a[j*lda:]
				for k := j + 1; k < n; k++ {
					if g := alpha * aj[k]; g != 0 {
						colk := b[k*ldb:]
						for i := range colj {
							colj[i] += g * colk[i]
						}
					}
				}
			}
		}
		return
	}
	// Right, Trans: B = alpha*B*A^T.
	if upper {
		// Ascending k: B(:,j<k) += alpha*A(j,k)*B(:,k) before B(:,k) scaled.
		for k := 0; k < n; k++ {
			ak := a[k*lda:]
			colk := b[k*ldb : k*ldb+m]
			for j := 0; j < k; j++ {
				if g := alpha * ak[j]; g != 0 {
					colj := b[j*ldb:]
					for i := range colk {
						colj[i] += g * colk[i]
					}
				}
			}
			f := alpha
			if !unit {
				f *= ak[k]
			}
			for i := range colk {
				colk[i] *= f
			}
		}
	} else {
		for k := n - 1; k >= 0; k-- {
			ak := a[k*lda:]
			colk := b[k*ldb : k*ldb+m]
			for j := k + 1; j < n; j++ {
				if g := alpha * ak[j]; g != 0 {
					colj := b[j*ldb:]
					for i := range colk {
						colj[i] += g * colk[i]
					}
				}
			}
			f := alpha
			if !unit {
				f *= ak[k]
			}
			for i := range colk {
				colk[i] *= f
			}
		}
	}
}
