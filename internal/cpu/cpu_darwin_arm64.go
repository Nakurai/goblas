//go:build darwin && arm64

package cpu

import (
	"strings"

	"golang.org/x/sys/unix"
)

// Detect classifies an Apple Silicon host via sysctl. It reads the CPU brand
// string to recognize the M-series family and the performance-core L1 data
// cache size to seed dgemm tile selection. On heterogeneous Apple chips the
// performance cores (perflevel0) carry the larger caches.
func Detect() CPU {
	c := CPU{HasNEON: true}

	if brand, err := unix.Sysctl("machdep.cpu.brand_string"); err == nil {
		if strings.HasPrefix(brand, "Apple M") {
			c.Microarch = AppleSilicon
		}
	}

	// Prefer the performance-core L1d size; fall back to the global value.
	if v, err := unix.SysctlUint32("hw.perflevel0.l1dcachesize"); err == nil && v > 0 {
		c.L1DBytes = int(v)
	} else if v, err := unix.SysctlUint32("hw.l1dcachesize"); err == nil {
		c.L1DBytes = int(v)
	}

	return c
}
