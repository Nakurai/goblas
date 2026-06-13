//go:build arm64

package kernel

import (
	"fmt"
	"math/rand"
	"testing"
)

func kRandSlice(seed int64, n int) []float64 {
	r := rand.New(rand.NewSource(seed))
	s := make([]float64, n)
	for i := range s {
		s[i] = r.NormFloat64()
	}
	return s
}

func kSecPerOp(b *testing.B) float64 {
	return b.Elapsed().Seconds() / float64(b.N)
}

var kSink float64

func BenchmarkDgemvGenericVsNEON(b *testing.B) {
	g := genericKernel{}
	nk := neonKernel{}
	for _, n := range []int{256, 1024, 4096} {
		a := kRandSlice(1, n*n)
		x := kRandSlice(2, n)
		y := make([]float64, n)
		flops := 2 * float64(n) * float64(n)

		b.Run(fmt.Sprintf("generic/%d", n), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				g.Dgemv(false, n, n, 1, a, n, x, 1, 0, y, 1)
				kSink += y[n-1]
			}
			b.ReportMetric(flops/kSecPerOp(b)/1e9, "GFLOPS")
		})
		b.Run(fmt.Sprintf("neon/%d", n), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				nk.Dgemv(false, n, n, 1, a, n, x, 1, 0, y, 1)
				kSink += y[n-1]
			}
			b.ReportMetric(flops/kSecPerOp(b)/1e9, "GFLOPS")
		})
	}
}
