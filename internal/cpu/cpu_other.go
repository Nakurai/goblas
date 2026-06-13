//go:build !arm64 && !amd64

package cpu

// Detect returns the zero CPU on hosts without arch-specific detection;
// callers use the pure-Go kernel. ARM64 detection lives in cpu_arm64.go /
// cpu_darwin_arm64.go, x86-64 in cpu_amd64.go.
func Detect() CPU { return CPU{} }
