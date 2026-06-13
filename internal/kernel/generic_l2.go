package kernel

func dgemvGeneric[T float](trans bool, m, n int, alpha T, a []T, lda int, x []T, incX int, beta T, y []T, incY int) {
	// Length of y (rows of op(A)) and of x (cols of op(A)).
	lenY, lenX := m, n
	if trans {
		lenY, lenX = n, m
	}
	if lenY == 0 {
		return
	}

	// y = beta*y first, so the accumulation below only adds alpha*op(A)*x.
	scaleStrided(lenY, beta, y, incY)
	if alpha == 0 || lenX == 0 {
		return
	}

	if !trans {
		// y += alpha * A * x. Walk columns (contiguous in column-major) and
		// axpy each into y: y += (alpha*x[j]) * A[:,j].
		jx := firstIndex(lenX, incX)
		for j := 0; j < n; j++ {
			c := alpha * x[jx]
			if c != 0 {
				col := a[j*lda : j*lda+m]
				iy := firstIndex(lenY, incY)
				for _, v := range col {
					y[iy] += c * v
					iy += incY
				}
			}
			jx += incX
		}
		return
	}

	// trans: y += alpha * A^T * x. Each y[j] is the dot of column j with x.
	jy := firstIndex(lenY, incY)
	for j := 0; j < n; j++ {
		col := a[j*lda : j*lda+m]
		var sum T
		ix := firstIndex(lenX, incX)
		for _, v := range col {
			sum += v * x[ix]
			ix += incX
		}
		y[jy] += alpha * sum
		jy += incY
	}
}

func dgerGeneric[T float](m, n int, alpha T, x []T, incX int, y []T, incY int, a []T, lda int) {
	if alpha == 0 {
		return
	}
	jy := firstIndex(n, incY)
	for j := 0; j < n; j++ {
		if f := alpha * y[jy]; f != 0 {
			col := a[j*lda : j*lda+m]
			ix := firstIndex(m, incX)
			for i := range col {
				col[i] += f * x[ix]
				ix += incX
			}
		}
		jy += incY
	}
}

func dtrsvGeneric[T float](upper, transA, unit bool, n int, a []T, lda int, x []T, incX int) {
	if n == 0 {
		return
	}
	// idx maps vector index i to its slice position.
	x0 := firstIndex(n, incX)
	idx := func(i int) int { return x0 + i*incX }

	switch {
	case !transA && upper:
		// Back substitution: A is upper triangular.
		for j := n - 1; j >= 0; j-- {
			if !unit {
				x[idx(j)] /= a[j+j*lda]
			}
			xj := x[idx(j)]
			if xj != 0 {
				col := a[j*lda:]
				for i := 0; i < j; i++ {
					x[idx(i)] -= xj * col[i]
				}
			}
		}
	case !transA && !upper:
		// Forward substitution: A is lower triangular.
		for j := 0; j < n; j++ {
			if !unit {
				x[idx(j)] /= a[j+j*lda]
			}
			xj := x[idx(j)]
			if xj != 0 {
				col := a[j*lda:]
				for i := j + 1; i < n; i++ {
					x[idx(i)] -= xj * col[i]
				}
			}
		}
	case transA && upper:
		// A^T is lower triangular: forward substitution with columns as rows.
		for j := 0; j < n; j++ {
			s := x[idx(j)]
			col := a[j*lda:]
			for i := 0; i < j; i++ {
				s -= col[i] * x[idx(i)]
			}
			if !unit {
				s /= a[j+j*lda]
			}
			x[idx(j)] = s
		}
	default:
		// transA && !upper: A^T is upper triangular: back substitution.
		for j := n - 1; j >= 0; j-- {
			s := x[idx(j)]
			col := a[j*lda:]
			for i := j + 1; i < n; i++ {
				s -= col[i] * x[idx(i)]
			}
			if !unit {
				s /= a[j+j*lda]
			}
			x[idx(j)] = s
		}
	}
}

// scaleStrided computes y = beta*y for a length-n strided vector.
func scaleStrided[T float](n int, beta T, y []T, incY int) {
	if beta == 1 {
		return
	}
	iy := firstIndex(n, incY)
	if beta == 0 {
		for i := 0; i < n; i++ {
			y[iy] = 0
			iy += incY
		}
		return
	}
	for i := 0; i < n; i++ {
		y[iy] *= beta
		iy += incY
	}
}

// --- float64 (D) method wrappers ---

func (k genericKernel) Dgemv(trans bool, m, n int, alpha float64, a []float64, lda int, x []float64, incX int, beta float64, y []float64, incY int) {
	dgemvGeneric(trans, m, n, alpha, a, lda, x, incX, beta, y, incY)
}

func (genericKernel) Dger(m, n int, alpha float64, x []float64, incX int, y []float64, incY int, a []float64, lda int) {
	dgerGeneric(m, n, alpha, x, incX, y, incY, a, lda)
}

func (genericKernel) Dtrsv(upper, transA, unit bool, n int, a []float64, lda int, x []float64, incX int) {
	dtrsvGeneric(upper, transA, unit, n, a, lda, x, incX)
}
