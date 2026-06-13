//go:build arm64

package kernel

// platformKernels lists the accelerated kernels available on this platform so
// shared tests cover them alongside the generic reference.
func platformKernels() map[string]Kernel {
	return map[string]Kernel{"neon": neonKernel{}}
}
