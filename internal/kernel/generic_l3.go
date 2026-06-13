package kernel

// Dgemm computes C = alpha*op(A)*op(B) + beta*C using the shared blocked,
// goroutine-parallel driver with the pure-Go micro-kernel. This is the
// portable fast path for hosts without an assembly kernel. Tiny problems
// skip the blocking machinery: packing overhead would dominate.
func (genericKernel) Dgemm(transA, transB bool, m, n, k int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int) {
	if m*n*k < 16*16*16 {
		dgemmNaive(transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
		return
	}
	dgemmBlocked(dgemmKernel8x4Go, dgemmNR, transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

// dgemmNaive is the straightforward column-at-a-time implementation. It is
// the correctness reference for the blocked/tiled implementations (tests
// compare both the generic and assembly kernels against it) and the fast
// path for tiny matrices where packing overhead dominates.
func dgemmNaive(transA, transB bool, m, n, k int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int) {
	if m == 0 || n == 0 {
		return
	}

	// C = beta*C first.
	if beta != 1 {
		for j := 0; j < n; j++ {
			col := c[j*ldc : j*ldc+m]
			if beta == 0 {
				for i := range col {
					col[i] = 0
				}
			} else {
				for i := range col {
					col[i] *= beta
				}
			}
		}
	}
	if alpha == 0 || k == 0 {
		return
	}

	// Accumulate C += alpha*op(A)*op(B), one output column at a time so writes
	// to C[:,j] stay contiguous (column-major). Inner loop is an axpy of an
	// op(A) column scaled by an op(B) element.
	for j := 0; j < n; j++ {
		cc := c[j*ldc : j*ldc+m]
		for l := 0; l < k; l++ {
			// op(B)(l,j)
			var bval float64
			if !transB {
				bval = b[l+j*ldb]
			} else {
				bval = b[j+l*ldb]
			}
			f := alpha * bval
			if f == 0 {
				continue
			}
			if !transA {
				// op(A)[:,l] is contiguous: a[l*lda : l*lda+m].
				ac := a[l*lda : l*lda+m]
				for i, v := range ac {
					cc[i] += f * v
				}
			} else {
				// op(A)(i,l) = A(l,i) = a[l + i*lda], strided by lda.
				for i := 0; i < m; i++ {
					cc[i] += f * a[l+i*lda]
				}
			}
		}
	}
}
