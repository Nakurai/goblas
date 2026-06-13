package kernel

// genericKernel is the portable pure-Go implementation of every routine. It is
// the correctness reference that accelerated kernels are tested against and the
// universal fallback on platforms without tuned assembly.
type genericKernel struct{}

// Generic returns the pure-Go kernel. It is exported so callers can force the
// reference implementation (e.g. in tests and benchmarks) regardless of CPU.
func Generic() Kernel { return genericKernel{} }
