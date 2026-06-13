//go:build gonumbench

package goblas

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/blas"
	gonumblas "gonum.org/v1/gonum/blas/gonum"
)

// Gonum's pure-Go BLAS implementation, used as the primary baseline.
var gonum = gonumblas.Implementation{}

// BenchmarkGonumVsGoblasDdot compares pure-Go Gonum against goblas (NEON on
// Apple Silicon) for the dot product across a range of vector sizes.
func BenchmarkGonumVsGoblasDdot(b *testing.B) {
	for _, n := range []int{1024, 16384, 262144} {
		x := benchMatrix(1, n)
		y := benchMatrix(2, n)
		flops := 2 * float64(n)

		b.Run(fmt.Sprintf("gonum/%d", n), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sink += gonum.Ddot(n, x, 1, y, 1)
			}
			b.ReportMetric(flops/secPerOp(b)/1e9, "GFLOPS")
		})
		b.Run(fmt.Sprintf("goblas/%d", n), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sink += Ddot(n, x, 1, y, 1)
			}
			b.ReportMetric(flops/secPerOp(b)/1e9, "GFLOPS")
		})
	}
}

// BenchmarkGonumVsGoblasDgemv compares pure-Go Gonum against goblas (NEON) for
// matrix-vector multiply on the NoTrans unit-stride path.
func BenchmarkGonumVsGoblasDgemv(b *testing.B) {
	for _, n := range []int{256, 1024, 4096} {
		a := benchMatrix(1, n*n)
		x := benchMatrix(2, n)
		y := make([]float64, n)
		flops := 2 * float64(n) * float64(n)

		b.Run(fmt.Sprintf("gonum/%d", n), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				gonum.Dgemv(blas.NoTrans, n, n, 1, a, n, x, 1, 0, y, 1)
				sink += y[n-1]
			}
			b.ReportMetric(flops/secPerOp(b)/1e9, "GFLOPS")
		})
		b.Run(fmt.Sprintf("goblas/%d", n), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				Dgemv(NoTrans, n, n, 1, a, n, x, 1, 0, y, 1)
				sink += y[n-1]
			}
			b.ReportMetric(flops/secPerOp(b)/1e9, "GFLOPS")
		})
	}
}

// BenchmarkGonumVsGoblasDgemm compares pure-Go Gonum against goblas for
// matrix-matrix multiply across a range of sizes.
func BenchmarkGonumVsGoblasDgemm(b *testing.B) {
	for _, n := range []int{64, 256, 512, 1024} {
		a := benchMatrix(1, n*n)
		bb := benchMatrix(2, n*n)
		c := make([]float64, n*n)
		flops := 2 * float64(n) * float64(n) * float64(n)

		b.Run(fmt.Sprintf("gonum/%d", n), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				gonum.Dgemm(blas.NoTrans, blas.NoTrans, n, n, n, 1, a, n, bb, n, 0, c, n)
				sink += c[n*n-1]
			}
			b.ReportMetric(flops/secPerOp(b)/1e9, "GFLOPS")
		})
		b.Run(fmt.Sprintf("goblas/%d", n), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				Dgemm(NoTrans, NoTrans, n, n, n, 1, a, n, bb, n, 0, c, n)
				sink += c[n*n-1]
			}
			b.ReportMetric(flops/secPerOp(b)/1e9, "GFLOPS")
		})
	}
}
