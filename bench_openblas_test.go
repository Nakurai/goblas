//go:build openblasbench

package goblas

import (
	"fmt"
	"testing"

	"github.com/nakurai/goblas/internal/openblas"
)

// BenchmarkOpenblasVsGoblasDgemm measures OpenBLAS (the de-facto open-source
// optimized BLAS, NEON-tuned for ARM64) on the same problem as goblas. Unlike
// Accelerate it does not use the AMX units, so this is the like-for-like SIMD
// comparison. Build with:
//
//	go test -tags openblasbench -run '^$' -bench Openblas .
func BenchmarkOpenblasVsGoblasDgemm(b *testing.B) {
	for _, n := range []int{64, 256, 512, 1024} {
		a := benchMatrix(1, n*n)
		bb := benchMatrix(2, n*n)
		c := make([]float64, n*n)
		flops := 2 * float64(n) * float64(n) * float64(n)

		b.Run(fmt.Sprintf("openblas/%d", n), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				openblas.Dgemm(n, n, n, 1, a, n, bb, n, 0, c, n)
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
