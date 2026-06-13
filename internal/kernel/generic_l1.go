package kernel

// The Level-1 routines are written once as element-generic free functions and
// exposed as both the float64 (D) and float32 (S) kernel methods. The method
// wrappers live here for float64; the float32 wrappers are in generic32.go.

func ddotGeneric[T float](n int, x []T, incX int, y []T, incY int) T {
	var sum T
	if incX == 1 && incY == 1 {
		x = x[:n]
		for i, v := range x {
			sum += v * y[i]
		}
		return sum
	}
	ix, iy := firstIndex(n, incX), firstIndex(n, incY)
	for i := 0; i < n; i++ {
		sum += x[ix] * y[iy]
		ix += incX
		iy += incY
	}
	return sum
}

func daxpyGeneric[T float](n int, alpha T, x []T, incX int, y []T, incY int) {
	if alpha == 0 {
		return
	}
	if incX == 1 && incY == 1 {
		x = x[:n]
		for i, v := range x {
			y[i] += alpha * v
		}
		return
	}
	ix, iy := firstIndex(n, incX), firstIndex(n, incY)
	for i := 0; i < n; i++ {
		y[iy] += alpha * x[ix]
		ix += incX
		iy += incY
	}
}

func dscalGeneric[T float](n int, alpha T, x []T, incX int) {
	if incX == 1 {
		x = x[:n]
		for i := range x {
			x[i] *= alpha
		}
		return
	}
	ix := firstIndex(n, incX)
	for i := 0; i < n; i++ {
		x[ix] *= alpha
		ix += incX
	}
}

func dnrm2Generic[T float](n int, x []T, incX int) T {
	if n < 1 {
		return 0
	}
	if n == 1 {
		return absT(x[0])
	}
	// Scaled accumulation to avoid overflow/underflow (LAPACK dnrm2 algorithm).
	var scale T
	var ssq T = 1
	ix := firstIndex(n, incX)
	for i := 0; i < n; i++ {
		v := x[ix]
		if v != 0 {
			a := absT(v)
			if scale < a {
				ssq = 1 + ssq*(scale/a)*(scale/a)
				scale = a
			} else {
				ssq += (a / scale) * (a / scale)
			}
		}
		ix += incX
	}
	return scale * sqrtT(ssq)
}

func dasumGeneric[T float](n int, x []T, incX int) T {
	var sum T
	if incX == 1 {
		x = x[:n]
		for _, v := range x {
			sum += absT(v)
		}
		return sum
	}
	ix := firstIndex(n, incX)
	for i := 0; i < n; i++ {
		sum += absT(x[ix])
		ix += incX
	}
	return sum
}

func idamaxGeneric[T float](n int, x []T, incX int) int {
	if n < 1 {
		return -1
	}
	ix := firstIndex(n, incX)
	best := 0
	maxv := absT(x[ix])
	ix += incX
	for i := 1; i < n; i++ {
		if a := absT(x[ix]); a > maxv {
			maxv = a
			best = i
		}
		ix += incX
	}
	return best
}

func dcopyGeneric[T float](n int, x []T, incX int, y []T, incY int) {
	if incX == 1 && incY == 1 {
		copy(y[:n], x[:n])
		return
	}
	ix, iy := firstIndex(n, incX), firstIndex(n, incY)
	for i := 0; i < n; i++ {
		y[iy] = x[ix]
		ix += incX
		iy += incY
	}
}

func dswapGeneric[T float](n int, x []T, incX int, y []T, incY int) {
	if incX == 1 && incY == 1 {
		x, y = x[:n], y[:n]
		for i, v := range x {
			x[i] = y[i]
			y[i] = v
		}
		return
	}
	ix, iy := firstIndex(n, incX), firstIndex(n, incY)
	for i := 0; i < n; i++ {
		x[ix], y[iy] = y[iy], x[ix]
		ix += incX
		iy += incY
	}
}

// --- float64 (D) method wrappers ---

func (genericKernel) Ddot(n int, x []float64, incX int, y []float64, incY int) float64 {
	return ddotGeneric(n, x, incX, y, incY)
}

func (genericKernel) Daxpy(n int, alpha float64, x []float64, incX int, y []float64, incY int) {
	daxpyGeneric(n, alpha, x, incX, y, incY)
}

func (genericKernel) Dscal(n int, alpha float64, x []float64, incX int) {
	dscalGeneric(n, alpha, x, incX)
}

func (genericKernel) Dnrm2(n int, x []float64, incX int) float64 {
	return dnrm2Generic(n, x, incX)
}

func (genericKernel) Dasum(n int, x []float64, incX int) float64 {
	return dasumGeneric(n, x, incX)
}

func (genericKernel) Idamax(n int, x []float64, incX int) int {
	return idamaxGeneric(n, x, incX)
}

func (genericKernel) Dcopy(n int, x []float64, incX int, y []float64, incY int) {
	dcopyGeneric(n, x, incX, y, incY)
}

func (genericKernel) Dswap(n int, x []float64, incX int, y []float64, incY int) {
	dswapGeneric(n, x, incX, y, incY)
}

// firstIndex returns the starting offset into a strided vector of n elements.
// A negative stride starts at the high end so the vector is traversed forward.
func firstIndex(n, inc int) int {
	if inc < 0 {
		return (1 - n) * inc
	}
	return 0
}
