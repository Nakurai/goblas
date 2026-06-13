//go:build arm64 && !darwin

package cpu

// Detect identifies a non-Apple ARM64 host (e.g. Linux servers). NEON/ASIMD is
// mandatory on ARM64, so it is always present. These hosts use the portable
// kernel until a tuned kernel is added for them; cache sizing is left unknown.
func Detect() CPU {
	return CPU{
		Microarch: Unknown,
		HasNEON:   true,
	}
}
