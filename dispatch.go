package goblas

import (
	"github.com/nakurai/goblas/internal/cpu"
	"github.com/nakurai/goblas/internal/kernel"
)

// active is the kernel all public routines delegate to. It is selected once at
// startup from the detected host CPU and is safe for concurrent use.
var active = kernel.Select(cpu.Detect())
