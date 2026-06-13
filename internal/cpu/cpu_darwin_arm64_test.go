//go:build darwin && arm64

package cpu

import "testing"

func TestDetectAppleSilicon(t *testing.T) {
	c := Detect()
	if c.Microarch != AppleSilicon {
		t.Errorf("Microarch = %v, want AppleSilicon", c.Microarch)
	}
	if !c.HasNEON {
		t.Error("HasNEON = false, want true on ARM64")
	}
	if c.L1DBytes <= 0 {
		t.Errorf("L1DBytes = %d, want > 0", c.L1DBytes)
	}
	t.Logf("detected: microarch=%d L1d=%dKB", c.Microarch, c.L1DBytes/1024)
}
