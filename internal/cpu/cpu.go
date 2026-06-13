// Package cpu detects the host processor so the kernel layer can select the
// best available implementation at startup. Detection is best-effort: when the
// host is unknown the zero CPU value is returned and callers fall back to the
// portable pure-Go kernel.
package cpu

// Microarch identifies a processor family precise enough to choose a tuned
// kernel and its blocking parameters.
type Microarch int

const (
	Unknown Microarch = iota
	AppleSilicon
	X86
)

// CPU describes the host relevant to numerical kernel selection.
type CPU struct {
	Microarch Microarch
	// HasNEON reports ARM64 NEON/ASIMD availability.
	HasNEON bool
	// HasAVX2FMA reports x86-64 AVX2+FMA availability (both are required by
	// the AVX2 dgemm kernel, and in practice always ship together).
	HasAVX2FMA bool
	// L1DBytes is the per-core L1 data cache size in bytes (0 if unknown).
	// Used to size dgemm tiles. On heterogeneous chips this is the
	// performance-core value.
	L1DBytes int
}
