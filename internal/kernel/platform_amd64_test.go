//go:build amd64

package kernel

import "github.com/nakurai/goblas/internal/cpu"

// platformKernels lists the accelerated kernels available on this platform so
// shared tests cover them alongside the generic reference. The AVX2 kernel is
// only included when the host supports it: running its assembly on a pre-AVX2
// CPU would fault with an illegal instruction.
func platformKernels() map[string]Kernel {
	if !cpu.Detect().HasAVX2FMA {
		return nil
	}
	return map[string]Kernel{"avx2": avx2Kernel{}}
}
