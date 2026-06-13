//go:build arm64

package kernel

import "github.com/nakurai/goblas/internal/cpu"

// neonKernel is the ARM64 NEON-accelerated kernel. It embeds genericKernel and
// overrides only the routines that have hand-written assembly; everything else
// falls back to the portable reference automatically. As assembly kernels land
// (Phases 3–5) their methods are defined on neonKernel to shadow the embedded
// generic ones.
type neonKernel struct {
	genericKernel
}

// Dsyrk and Dtrsm use the shared recursive blocking, but feed it the
// NEON-backed Dgemm so the off-diagonal updates (where the FLOPs are) run on
// assembly.

func (nk neonKernel) Dsyrk(upper, trans bool, n, k int, alpha float64, a []float64, lda int, beta float64, c []float64, ldc int) {
	dsyrkRec(nk.Dgemm, upper, trans, n, k, alpha, a, lda, beta, c, ldc)
}

func (nk neonKernel) Dtrsm(left, upper, transA, unit bool, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int) {
	dtrsmRec(nk.Dgemm, left, upper, transA, unit, m, n, alpha, a, lda, b, ldb)
}

// Select routes any NEON-capable ARM64 host to the NEON kernel (NEON/ASIMD is
// mandatory on ARM64, so in practice all of them). Blocking parameters are
// tuned per microarchitecture: the M5 Pro values are the Phase 6 sweep
// winners; other ARM64 hosts get conservative defaults sized for the typical
// 64 KB L1d (packed A block = mc*kc*8 = 32*256*8 = 64 KB).
func Select(c cpu.CPU) Kernel {
	if !c.HasNEON {
		return genericKernel{}
	}
	if c.Microarch != cpu.AppleSilicon {
		dgemmKC, dgemmMC = 256, 32
		if c.L1DBytes > 0 {
			// Size the packed A block to the detected L1d.
			dgemmMC = max(dgemmMR, c.L1DBytes/(dgemmKC*8)/dgemmMR*dgemmMR)
		}
	}
	return neonKernel{}
}
