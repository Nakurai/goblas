#include "textflag.h"

// func dgemvNoTransNEON(m, n int, alpha float64, a *float64, lda int, x, y *float64)
//
// Computes y += alpha * A * x for a column-major m×n matrix A with unit-stride
// x and y. The algorithm is column-axpy: for each column j, y += (alpha*x[j]) * A[:,j].
// Columns are contiguous in memory so A[:,j] streams naturally into NEON.
//
// Processes 4 rows per inner-loop iteration (two NEON regs unrolled), with a
// 2-wide step and a scalar tail for the remainder.
//
// Arguments (offsets into FP):
//   m     +0  (8)
//   n     +8  (8)
//   alpha +16 (8)
//   a     +24 (8)
//   lda   +32 (8)   — element count, converted to bytes below
//   x     +40 (8)
//   y     +48 (8)
TEXT ·dgemvNoTransNEON(SB), NOSPLIT, $0-56
	MOVD  m+0(FP), R0         // R0 = m
	MOVD  n+8(FP), R4         // R4 = n (column counter, counts down)
	FMOVD alpha+16(FP), F0    // F0 = alpha
	MOVD  a+24(FP), R1        // R1 = base of A; advances by lda*8 per column
	MOVD  lda+32(FP), R6      // R6 = lda (elements)
	MOVD  x+40(FP), R2        // R2 = x pointer; advances by 8 per column
	MOVD  y+48(FP), R3        // R3 = y base (fixed — reset each column iteration)
	LSL   $3, R6, R6          // R6 = lda in bytes

	CBZ R4, notrans_done

notrans_col:
	// Compute alpha*x[j] and broadcast to both lanes of V16.
	FMOVD  (R2), F1
	FMULD  F0, F1, F1          // F1 = alpha * x[j]
	ADD    $8, R2, R2
	VDUP   V1.D[0], V16.D2    // V16 = [alpha*x[j], alpha*x[j]]

	MOVD R1, R5               // R5 = column pointer (a + j*lda), advances
	MOVD R3, R7               // R7 = y pointer, reset to base each column
	MOVD R0, R8               // R8 = row count

	// 4-wide unrolled inner loop (2 NEON regs per iteration).
	LSR  $2, R8, R9           // R9 = m / 4
	CBZ  R9, notrans_rem2

notrans_row4:
	VLD1.P 16(R5), [V4.D2]    // A[i:i+2, j]
	VLD1   (R7), [V6.D2]      // y[i:i+2]
	VFMLA  V16.D2, V4.D2, V6.D2
	VST1.P [V6.D2], 16(R7)

	VLD1.P 16(R5), [V5.D2]    // A[i+2:i+4, j]
	VLD1   (R7), [V8.D2]      // y[i+2:i+4]
	VFMLA  V16.D2, V5.D2, V8.D2
	VST1.P [V8.D2], 16(R7)

	SUB    $1, R9, R9
	CBNZ   R9, notrans_row4

notrans_rem2:
	AND  $3, R8, R8            // R8 = m % 4
	LSR  $1, R8, R9            // R9 = remaining pairs
	CBZ  R9, notrans_row_tail

notrans_row2:
	VLD1.P 16(R5), [V4.D2]
	VLD1   (R7), [V6.D2]
	VFMLA  V16.D2, V4.D2, V6.D2
	VST1.P [V6.D2], 16(R7)
	SUB    $1, R9, R9
	CBNZ   R9, notrans_row2

notrans_row_tail:
	AND  $1, R8, R8
	CBZ  R8, notrans_next_col
	FMOVD (R5), F4
	FMOVD (R7), F6
	FMULD  F1, F4, F4
	FADDD  F4, F6, F6
	FMOVD  F6, (R7)

notrans_next_col:
	ADD R6, R1, R1             // A base → next column
	SUB $1, R4, R4
	CBNZ R4, notrans_col

notrans_done:
	RET
