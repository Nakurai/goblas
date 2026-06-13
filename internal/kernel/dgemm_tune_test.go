//go:build arm64

package kernel

import (
	"fmt"
	"testing"
)

// BenchmarkDgemmBlockSweep sweeps the kc/mc cache-blocking parameters at a
// fixed problem size to find the best combination for the host. Run with:
//
//	go test -run '^$' -bench BenchmarkDgemmBlockSweep -benchtime=500ms ./internal/kernel/
func BenchmarkDgemmBlockSweep(b *testing.B) {
	nk := neonKernel{}
	const n = 1024
	a := kRandSlice(1, n*n)
	bb := kRandSlice(2, n*n)
	c := make([]float64, n*n)
	flops := 2 * float64(n) * float64(n) * float64(n)

	savedKC, savedMC := dgemmKC, dgemmMC
	defer func() { dgemmKC, dgemmMC = savedKC, savedMC }()

	savedMK, savedNR := neonMicroKernel, neonNR
	defer func() { neonMicroKernel, neonNR = savedMK, savedNR }()

	kernels := []struct {
		name string
		mk   microKernel[float64]
		nr   int
	}{
		{"8x4", dgemmKernel8x4, 4},
		{"8x6", dgemmKernel8x6, 6},
	}
	for _, kn := range kernels {
		for _, kc := range []int{384, 512, 640} {
			for _, mc := range []int{16, 24, 32} {
				b.Run(fmt.Sprintf("%s/kc%d/mc%d", kn.name, kc, mc), func(b *testing.B) {
					neonMicroKernel, neonNR = kn.mk, kn.nr
					dgemmKC, dgemmMC = kc, mc
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						nk.Dgemm(false, false, n, n, n, 1, a, n, bb, n, 0, c, n)
						kSink += c[n*n-1]
					}
					b.ReportMetric(flops/kSecPerOp(b)/1e9, "GFLOPS")
				})
			}
		}
	}
}

// BenchmarkDgemmWorkerSweep sweeps the dgemm worker cap to see whether
// restricting parallelism toward the P-core count beats using every core
// (macOS offers no hard affinity, so this is the portable lever we have).
func BenchmarkDgemmWorkerSweep(b *testing.B) {
	nk := neonKernel{}
	saved := dgemmMaxWorkers
	defer func() { dgemmMaxWorkers = saved }()

	for _, n := range []int{256, 512, 1024} {
		a := kRandSlice(1, n*n)
		bb := kRandSlice(2, n*n)
		c := make([]float64, n*n)
		flops := 2 * float64(n) * float64(n) * float64(n)
		for _, w := range []int{3, 5, 8, 10, 15, 0} {
			name := fmt.Sprintf("n%d/w%d", n, w)
			if w == 0 {
				name = fmt.Sprintf("n%d/wAll", n)
			}
			b.Run(name, func(b *testing.B) {
				dgemmMaxWorkers = w
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					nk.Dgemm(false, false, n, n, n, 1, a, n, bb, n, 0, c, n)
					kSink += c[n*n-1]
				}
				b.ReportMetric(flops/kSecPerOp(b)/1e9, "GFLOPS")
			})
		}
	}
}
