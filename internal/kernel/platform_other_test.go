//go:build !arm64 && !amd64

package kernel

// platformKernels lists the accelerated kernels available on this platform.
// None outside ARM64 (NEON) and AMD64 (AVX2).
func platformKernels() map[string]Kernel { return nil }
