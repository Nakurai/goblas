package goblas

import (
	"github.com/nakurai/goblas/internal/cpu"
	"github.com/nakurai/goblas/internal/kernel"
)

// active and active32 are the kernels all public routines delegate to (float64
// and float32 respectively). Both are selected once at startup from the
// detected host CPU and are safe for concurrent use.
var (
	detected = cpu.Detect()
	active   = kernel.Select(detected)
	active32 = kernel.Select32(detected)
)
