//go:build amd64

package cpu

import xcpu "golang.org/x/sys/cpu"

// Detect identifies an x86-64 host. AVX2 and FMA are detected via CPUID
// (golang.org/x/sys/cpu); they have shipped together since Haswell (2013) and
// the AVX2 dgemm kernel requires both. Cache sizing is left unknown — the
// kernel layer applies conservative blocking defaults.
func Detect() CPU {
	return CPU{
		Microarch:  X86,
		HasAVX2FMA: xcpu.X86.HasAVX2 && xcpu.X86.HasFMA,
	}
}
