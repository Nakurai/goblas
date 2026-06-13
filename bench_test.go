package goblas

import (
	"fmt"
	"math/rand"
	"testing"
)

// sink defeats dead-code elimination: benchmarked routines write a result here
// so the compiler cannot prove their work is unused.
var sink float64

func benchMatrix(seed int64, n int) []float64 {
	r := rand.New(rand.NewSource(seed))
	return randSlice(r, n)
}

// BenchmarkDgemm reports GFLOPS (2*N^3 flops) across a range of square sizes so
// the L1->L2->memory performance curve is visible. Run with a longer benchtime
// (e.g. -benchtime=10s) for stable numbers under frequency scaling.
func BenchmarkDgemm(b *testing.B) {
	for _, n := range []int{32, 64, 128, 256, 512, 1024} {
		b.Run(fmt.Sprintf("%dx%d", n, n), func(b *testing.B) {
			a := benchMatrix(1, n*n)
			bb := benchMatrix(2, n*n)
			c := make([]float64, n*n)
			flops := 2 * float64(n) * float64(n) * float64(n)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				Dgemm(NoTrans, NoTrans, n, n, n, 1, a, n, bb, n, 0, c, n)
				sink += c[n*n-1]
			}
			b.ReportMetric(flops/secPerOp(b), "FLOPS/op")
			b.ReportMetric(flops/secPerOp(b)/1e9, "GFLOPS")
		})
	}
}

// BenchmarkDgemv reports GFLOPS (2*N^2 flops) for matrix-vector multiply.
func BenchmarkDgemv(b *testing.B) {
	for _, n := range []int{64, 256, 1024, 4096} {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			a := benchMatrix(1, n*n)
			x := benchMatrix(2, n)
			y := make([]float64, n)
			flops := 2 * float64(n) * float64(n)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				Dgemv(NoTrans, n, n, 1, a, n, x, 1, 0, y, 1)
				sink += y[n-1]
			}
			b.ReportMetric(flops/secPerOp(b)/1e9, "GFLOPS")
		})
	}
}

// BenchmarkDdot reports GFLOPS (2*N flops) for the dot product.
func BenchmarkDdot(b *testing.B) {
	for _, n := range []int{1024, 16384, 262144} {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			x := benchMatrix(1, n)
			y := benchMatrix(2, n)
			flops := 2 * float64(n)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sink += Ddot(n, x, 1, y, 1)
			}
			b.ReportMetric(flops/secPerOp(b)/1e9, "GFLOPS")
		})
	}
}

// BenchmarkDgemv reports GFLOPS (2*N^2 flops for square m=n) for the NoTrans
// unit-stride path, which is what the NEON kernel accelerates.
func BenchmarkDgemvNEON(b *testing.B) {
	for _, n := range []int{64, 256, 1024, 4096} {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			a := benchMatrix(1, n*n)
			x := benchMatrix(2, n)
			y := make([]float64, n)
			flops := 2 * float64(n) * float64(n)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				Dgemv(NoTrans, n, n, 1, a, n, x, 1, 0, y, 1)
				sink += y[n-1]
			}
			b.ReportMetric(flops/secPerOp(b)/1e9, "GFLOPS")
		})
	}
}

// BenchmarkDaxpy reports GFLOPS (2*N flops) for y += alpha*x.
func BenchmarkDaxpy(b *testing.B) {
	for _, n := range []int{1024, 16384, 262144} {
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			x := benchMatrix(1, n)
			y := benchMatrix(2, n)
			flops := 2 * float64(n)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				Daxpy(n, 1.0001, x, 1, y, 1)
			}
			sink += y[n-1]
			b.ReportMetric(flops/secPerOp(b)/1e9, "GFLOPS")
		})
	}
}

// secPerOp returns the average seconds per iteration of the just-run benchmark.
func secPerOp(b *testing.B) float64 {
	return b.Elapsed().Seconds() / float64(b.N)
}
