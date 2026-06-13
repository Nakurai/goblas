//go:build accelbench && darwin

package goblas

import (
	"fmt"
	"testing"

	"github.com/nakurai/goblas/internal/accel"
)

// BenchmarkAccelerateVsGoblasDgemm measures Apple's Accelerate framework on
// the same problem as goblas. Accelerate uses the undocumented AMX matrix
// units, so it is the hardware ceiling, not a like-for-like SIMD comparison.
// Build with: go test -tags accelbench -run '^$' -bench Accelerate .
func BenchmarkAccelerateVsGoblasDgemm(b *testing.B) {
	for _, n := range []int{64, 256, 512, 1024} {
		a := benchMatrix(1, n*n)
		bb := benchMatrix(2, n*n)
		c := make([]float64, n*n)
		flops := 2 * float64(n) * float64(n) * float64(n)

		b.Run(fmt.Sprintf("accelerate/%d", n), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				accel.Dgemm(n, n, n, 1, a, n, bb, n, 0, c, n)
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
