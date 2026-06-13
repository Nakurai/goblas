//go:build !arm64 && !amd64

package kernel

import "github.com/nakurai/goblas/internal/cpu"

// Select always returns the portable reference on hosts without assembly
// kernels (ARM64 NEON lives in arm64.go, x86-64 AVX2 in avx2_amd64.go).
func Select(c cpu.CPU) Kernel {
	_ = c
	return genericKernel{}
}
