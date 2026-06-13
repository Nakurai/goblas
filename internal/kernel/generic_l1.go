package kernel

import "math"

func (genericKernel) Ddot(n int, x []float64, incX int, y []float64, incY int) float64 {
	var sum float64
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

func (genericKernel) Daxpy(n int, alpha float64, x []float64, incX int, y []float64, incY int) {
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

func (genericKernel) Dscal(n int, alpha float64, x []float64, incX int) {
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

func (genericKernel) Dnrm2(n int, x []float64, incX int) float64 {
	if n < 1 {
		return 0
	}
	if n == 1 {
		return math.Abs(x[0])
	}
	// Scaled accumulation to avoid overflow/underflow (LAPACK dnrm2 algorithm).
	var scale float64
	ssq := 1.0
	ix := firstIndex(n, incX)
	for i := 0; i < n; i++ {
		v := x[ix]
		if v != 0 {
			a := math.Abs(v)
			if scale < a {
				ssq = 1 + ssq*(scale/a)*(scale/a)
				scale = a
			} else {
				ssq += (a / scale) * (a / scale)
			}
		}
		ix += incX
	}
	return scale * math.Sqrt(ssq)
}

func (genericKernel) Dasum(n int, x []float64, incX int) float64 {
	var sum float64
	if incX == 1 {
		x = x[:n]
		for _, v := range x {
			sum += math.Abs(v)
		}
		return sum
	}
	ix := firstIndex(n, incX)
	for i := 0; i < n; i++ {
		sum += math.Abs(x[ix])
		ix += incX
	}
	return sum
}

func (genericKernel) Idamax(n int, x []float64, incX int) int {
	if n < 1 {
		return -1
	}
	ix := firstIndex(n, incX)
	best := 0
	max := math.Abs(x[ix])
	ix += incX
	for i := 1; i < n; i++ {
		if a := math.Abs(x[ix]); a > max {
			max = a
			best = i
		}
		ix += incX
	}
	return best
}

func (genericKernel) Dcopy(n int, x []float64, incX int, y []float64, incY int) {
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

func (genericKernel) Dswap(n int, x []float64, incX int, y []float64, incY int) {
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

// firstIndex returns the starting offset into a strided vector of n elements.
// A negative stride starts at the high end so the vector is traversed forward.
func firstIndex(n, inc int) int {
	if inc < 0 {
		return (1 - n) * inc
	}
	return 0
}
