package mat32

import (
	"fmt"
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/mat"
)

var benchSink float32

// BenchmarkMul compares mat32 float32 Dense32.Mul against gonum/mat float64
// Dense.Mul (stock gonum BLAS) at several sizes.
func BenchmarkMul(b *testing.B) {
	r := rand.New(rand.NewSource(1))
	for _, n := range []int{256, 512, 1024} {
		d32 := make([]float32, n*n)
		d64 := make([]float64, n*n)
		for i := range d32 {
			v := r.NormFloat64()
			d32[i], d64[i] = float32(v), v
		}
		a32, b32 := NewDense32(n, n, d32), NewDense32(n, n, append([]float32(nil), d32...))
		a64 := mat.NewDense(n, n, d64)
		b64 := mat.NewDense(n, n, append([]float64(nil), d64...))

		b.Run(fmt.Sprintf("mat32/%d", n), func(b *testing.B) {
			var c Dense32
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				c.Mul(a32, b32)
			}
			benchSink += c.At(0, 0)
		})
		b.Run(fmt.Sprintf("gonum64/%d", n), func(b *testing.B) {
			var c mat.Dense
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				c.Mul(a64, b64)
			}
			benchSink += float32(c.At(0, 0))
		})
	}
}

// BenchmarkCholeskySolve compares the native float32 Cholesky solve against
// gonum's float64 Cholesky. ReportAllocs documents that the float32 path
// allocates only float32 working buffers (no float64 temporaries).
func BenchmarkCholeskySolve(b *testing.B) {
	r := rand.New(rand.NewSource(2))
	for _, n := range []int{256, 512} {
		a, a64 := spd(r, n)
		rhs := make([]float32, n)
		rhs64 := make([]float64, n)
		for i := range rhs {
			v := r.NormFloat64()
			rhs[i], rhs64[i] = float32(v), v
		}
		bvec := NewDense32(n, 1, rhs)

		b.Run(fmt.Sprintf("mat32/%d", n), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				var c Cholesky32
				c.Factorize(a)
				var x Dense32
				_ = c.SolveTo(&x, bvec)
				benchSink += x.At(0, 0)
			}
		})
		b.Run(fmt.Sprintf("gonum64/%d", n), func(b *testing.B) {
			b.ReportAllocs()
			sym := mat.NewSymDense(n, a64)
			bv := mat.NewVecDense(n, rhs64)
			for i := 0; i < b.N; i++ {
				var c mat.Cholesky
				c.Factorize(sym)
				var x mat.VecDense
				_ = c.SolveVecTo(&x, bv)
				benchSink += float32(x.AtVec(0))
			}
		})
	}
}
