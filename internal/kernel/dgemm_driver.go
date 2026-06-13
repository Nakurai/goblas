package kernel

import (
	"runtime"
	"sync"
	"unsafe"
)

// sliceFrom reconstructs a []float64 of length n from a panel pointer. The
// micro-kernel API passes raw pointers so the assembly and Go kernels share a
// signature; the Go kernel needs slices back for bounds-checked access.
func sliceFrom(p *float64, n int) []float64 { return unsafe.Slice(p, n) }

// Micro-kernel tile shape. All micro-kernels use mr=8 rows; the tile width nr
// varies per kernel (4 for the pure-Go and NEON 8x4 kernels, 6 for the NEON
// 8x6 kernel) and is passed to the driver alongside the kernel function.
const (
	dgemmMR    = 8
	dgemmNR    = 4
	dgemmNRMax = 8 // scratch sizing bound for edge tiles
)

// Cache-blocking parameters. Defaults are tuned for the Apple M5 Pro (Phase 6
// sweep): mc=24 keeps each packed A block (mc x kc = 24*512*8 = 96 KB) inside
// the 128 KB P-core L1d while yielding enough row blocks to load-balance
// across all cores; kc=512 amortizes the C writeback over deep micro-panels.
// Select() overrides them with conservative values on hosts with smaller or
// unknown caches. Variables rather than constants so the tuning benchmarks can
// sweep them; production code treats them as fixed after init.
var (
	dgemmKC = 512
	dgemmMC = 24
)

// dgemmMaxWorkers caps dgemm parallelism when positive; 0 means use up to
// GOMAXPROCS. Exists for the worker-count tuning benchmark (P-core vs all-core
// scheduling on asymmetric chips).
var dgemmMaxWorkers = 0

// microKernel computes C[mr x nr] += Apanel * Bpanel for packed micro-panels:
// a holds k slices of mr contiguous values, b holds k slices of nr contiguous
// values, and c is column-major with leading dimension ldc (in elements).
type microKernel func(k int, a, b, c *float64, ldc int)

// dgemmBlocked is the blocked/tiled/parallel dgemm driver shared by every
// kernel implementation; only the micro-kernel differs (NEON assembly on
// ARM64, pure Go elsewhere). Packing normalizes all four transpose
// combinations, scaling by beta happens once up front, and alpha is folded
// into the packed copy of A.
func dgemmBlocked(mk microKernel, nr int, transA, transB bool, m, n, k int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int) {
	if m == 0 || n == 0 {
		return
	}

	// C = beta*C first (same contract as the reference).
	if beta != 1 {
		for j := 0; j < n; j++ {
			col := c[j*ldc : j*ldc+m]
			if beta == 0 {
				for i := range col {
					col[i] = 0
				}
			} else {
				for i := range col {
					col[i] *= beta
				}
			}
		}
	}
	if alpha == 0 || k == 0 {
		return
	}

	// Packed B panel is shared read-only by all workers; each worker packs its
	// own A block. Buffers are padded up to whole micro-panels so the kernel
	// never reads partial slices.
	kcMax := min(k, dgemmKC)
	packedB := make([]float64, roundUp(n, nr)*kcMax)

	// Parallelize across row blocks when there is enough work to amortize the
	// goroutine overhead. Each ic block touches a disjoint row range of C, so
	// workers never write the same memory.
	nBlocks := (m + dgemmMC - 1) / dgemmMC
	workers := 1
	if flops := float64(m) * float64(n) * float64(k); flops >= 2e6 {
		workers = min(runtime.GOMAXPROCS(0), nBlocks)
		if dgemmMaxWorkers > 0 {
			workers = min(workers, dgemmMaxWorkers)
		}
	}

	for pc := 0; pc < k; pc += dgemmKC {
		kc := min(dgemmKC, k-pc)

		// Pack B(pc:pc+kc, 0:n) once per k-block.
		packBPanels(kc, n, nr, b, ldb, transB, pc, packedB)

		if workers == 1 {
			dgemmRowBlocks(mk, nr, m, n, kc, alpha, a, lda, transA, pc, packedB, c, ldc, 0, 1)
			continue
		}
		var wg sync.WaitGroup
		for w := 0; w < workers; w++ {
			wg.Add(1)
			go func(w int) {
				defer wg.Done()
				dgemmRowBlocks(mk, nr, m, n, kc, alpha, a, lda, transA, pc, packedB, c, ldc, w, workers)
			}(w)
		}
		wg.Wait()
	}
}

// dgemmRowBlocks processes every workers-th mc-sized row block, starting at
// block index w. It owns its packed-A buffer and scratch tile, so concurrent
// calls with distinct w never share mutable state (C row ranges are disjoint).
func dgemmRowBlocks(mk microKernel, nr, m, n, kc int, alpha float64, a []float64, lda int, transA bool, pc int, packedB []float64, c []float64, ldc int, w, workers int) {
	packedA := make([]float64, roundUp(min(m, dgemmMC), dgemmMR)*kc)
	var scratchBuf [dgemmMR * dgemmNRMax]float64
	scratch := scratchBuf[:dgemmMR*nr]

	for blk := w; blk*dgemmMC < m; blk += workers {
		ic := blk * dgemmMC
		mc := min(dgemmMC, m-ic)

		// Pack alpha*A(ic:ic+mc, pc:pc+kc) into mr-wide micro-panels.
		packAPanels(mc, kc, alpha, a, lda, transA, ic, pc, packedA)

		for jr := 0; jr < n; jr += nr {
			nrr := min(nr, n-jr)
			bp := &packedB[(jr/nr)*kc*nr]

			for ir := 0; ir < mc; ir += dgemmMR {
				mrr := min(dgemmMR, mc-ir)
				ap := &packedA[(ir/dgemmMR)*kc*dgemmMR]

				if mrr == dgemmMR && nrr == nr {
					// Full tile: accumulate straight into C.
					mk(kc, ap, bp, &c[(ic+ir)+jr*ldc], ldc)
					continue
				}
				// Edge tile: compute into a zeroed scratch tile, then add
				// the valid region into C.
				for i := range scratch {
					scratch[i] = 0
				}
				mk(kc, ap, bp, &scratchBuf[0], dgemmMR)
				for j := 0; j < nrr; j++ {
					cc := c[(ic+ir)+(jr+j)*ldc:]
					sc := scratch[j*dgemmMR:]
					for i := 0; i < mrr; i++ {
						cc[i] += sc[i]
					}
				}
			}
		}
	}
}

// dgemmKernel8x4Go is the pure-Go micro-kernel: the portable counterpart of
// the assembly kernels, operating on the same packed-panel format. The 8x4
// accumulator tile lives in a fixed-size array the compiler can register-
// allocate aggressively; the k-loop body is 32 multiply-adds.
func dgemmKernel8x4Go(k int, a, b, c *float64, ldc int) {
	// Reconstruct slices from the panel pointers (length set by k).
	ap := sliceFrom(a, k*dgemmMR)
	bp := sliceFrom(b, k*dgemmNR)

	var acc [dgemmMR * dgemmNR]float64
	for l := 0; l < k; l++ {
		as := ap[l*dgemmMR : l*dgemmMR+dgemmMR]
		bs := bp[l*dgemmNR : l*dgemmNR+dgemmNR]
		b0, b1, b2, b3 := bs[0], bs[1], bs[2], bs[3]
		for i := 0; i < dgemmMR; i++ {
			ai := as[i]
			acc[i] += ai * b0
			acc[i+8] += ai * b1
			acc[i+16] += ai * b2
			acc[i+24] += ai * b3
		}
	}

	cs := sliceFrom(c, (dgemmNR-1)*ldc+dgemmMR)
	for j := 0; j < dgemmNR; j++ {
		col := cs[j*ldc : j*ldc+dgemmMR]
		av := acc[j*dgemmMR : j*dgemmMR+dgemmMR]
		for i := range col {
			col[i] += av[i]
		}
	}
}

// packAPanels packs alpha*op(A)(ic:ic+mc, pc:pc+kc) into mr-wide micro-panels:
// panel p holds rows [p*mr, p*mr+mr) as kc consecutive slices of mr contiguous
// values. Rows beyond mc are zero-padded so the kernel can always read mr.
func packAPanels(mc, kc int, alpha float64, a []float64, lda int, transA bool, ic, pc int, buf []float64) {
	for ir := 0; ir < mc; ir += dgemmMR {
		panel := buf[(ir/dgemmMR)*kc*dgemmMR:]
		rows := min(dgemmMR, mc-ir)
		for l := 0; l < kc; l++ {
			dst := panel[l*dgemmMR : l*dgemmMR+dgemmMR]
			if !transA {
				// op(A)(i,l) = a[(ic+ir+i) + (pc+l)*lda]
				src := a[(ic+ir)+(pc+l)*lda:]
				for i := 0; i < rows; i++ {
					dst[i] = alpha * src[i]
				}
			} else {
				// op(A)(i,l) = A(l,i) = a[(pc+l) + (ic+ir+i)*lda]
				for i := 0; i < rows; i++ {
					dst[i] = alpha * a[(pc+l)+(ic+ir+i)*lda]
				}
			}
			for i := rows; i < dgemmMR; i++ {
				dst[i] = 0
			}
		}
	}
}

// packBPanels packs op(B)(pc:pc+kc, 0:n) into nr-wide micro-panels: panel p
// holds columns [p*nr, p*nr+nr) as kc consecutive slices of nr contiguous
// values. Columns beyond n are zero-padded.
func packBPanels(kc, n, nr int, b []float64, ldb int, transB bool, pc int, buf []float64) {
	for jr := 0; jr < n; jr += nr {
		panel := buf[(jr/nr)*kc*nr:]
		cols := min(nr, n-jr)
		for l := 0; l < kc; l++ {
			dst := panel[l*nr : l*nr+nr]
			if !transB {
				// op(B)(l,j) = b[(pc+l) + (jr+j)*ldb]
				for j := 0; j < cols; j++ {
					dst[j] = b[(pc+l)+(jr+j)*ldb]
				}
			} else {
				// op(B)(l,j) = B(j,l) = b[(jr+j) + (pc+l)*ldb]
				src := b[jr+(pc+l)*ldb:]
				for j := 0; j < cols; j++ {
					dst[j] = src[j]
				}
			}
			for j := cols; j < nr; j++ {
				dst[j] = 0
			}
		}
	}
}

func roundUp(x, to int) int { return (x + to - 1) / to * to }
