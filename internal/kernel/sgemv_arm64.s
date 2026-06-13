#include "textflag.h"

// func sgemvNoTransNEON(m, n int, alpha float32, a *float32, lda int, x, y *float32)
//
// Computes y += alpha * A * x for a column-major m×n matrix A with unit-stride
// x and y, in single precision. Column-axpy: for each column j,
// y += (alpha*x[j]) * A[:,j]. Columns are contiguous so A[:,j] streams into
// NEON. Processes 8 rows per inner iteration (two .S4 regs), a 4-wide step, and
// a scalar tail.
//
// Arguments (offsets into FP):
//   m     +0  (8)
//   n     +8  (8)
//   alpha +16 (4)
//   a     +24 (8)
//   lda   +32 (8)   — element count, converted to bytes below
//   x     +40 (8)
//   y     +48 (8)
TEXT ·sgemvNoTransNEON(SB), NOSPLIT, $0-56
	MOVD  m+0(FP), R0         // R0 = m
	MOVD  n+8(FP), R4         // R4 = n (column counter, counts down)
	FMOVS alpha+16(FP), F0    // F0 = alpha
	MOVD  a+24(FP), R1        // R1 = base of A; advances by lda*4 per column
	MOVD  lda+32(FP), R6      // R6 = lda (elements)
	MOVD  x+40(FP), R2        // R2 = x pointer; advances by 4 per column
	MOVD  y+48(FP), R3        // R3 = y base (reset each column iteration)
	LSL   $2, R6, R6          // R6 = lda in bytes (float32 = 4)

	CBZ R4, notrans_done

notrans_col:
	// Compute alpha*x[j] and broadcast to all 4 lanes of V16.
	FMOVS  (R2), F1
	FMULS  F0, F1, F1          // F1 = alpha * x[j]
	ADD    $4, R2, R2
	VDUP   V1.S[0], V16.S4     // V16 = [alpha*x[j] x4]

	MOVD R1, R5               // R5 = column pointer (a + j*lda)
	MOVD R3, R7               // R7 = y pointer, reset to base each column
	MOVD R0, R8               // R8 = row count

	LSR  $3, R8, R9           // R9 = m / 8
	CBZ  R9, notrans_rem4

notrans_row8:
	VLD1.P 16(R5), [V4.S4]    // A[i:i+4, j]
	VLD1   (R7), [V6.S4]      // y[i:i+4]
	VFMLA  V16.S4, V4.S4, V6.S4
	VST1.P [V6.S4], 16(R7)

	VLD1.P 16(R5), [V5.S4]    // A[i+4:i+8, j]
	VLD1   (R7), [V8.S4]      // y[i+4:i+8]
	VFMLA  V16.S4, V5.S4, V8.S4
	VST1.P [V8.S4], 16(R7)

	SUB    $1, R9, R9
	CBNZ   R9, notrans_row8

notrans_rem4:
	AND  $7, R8, R8            // R8 = m % 8
	LSR  $2, R8, R9            // R9 = remaining quads
	CBZ  R9, notrans_row_tail

notrans_row4:
	VLD1.P 16(R5), [V4.S4]
	VLD1   (R7), [V6.S4]
	VFMLA  V16.S4, V4.S4, V6.S4
	VST1.P [V6.S4], 16(R7)
	SUB    $1, R9, R9
	CBNZ   R9, notrans_row4

notrans_row_tail:
	AND  $3, R8, R8            // R8 = m % 4
	CBZ  R8, notrans_next_col
notrans_tloop:
	FMOVS (R5), F4
	FMOVS (R7), F6
	FMULS F1, F4, F4
	FADDS F4, F6, F6
	FMOVS F6, (R7)
	ADD   $4, R5, R5
	ADD   $4, R7, R7
	SUB   $1, R8, R8
	CBNZ  R8, notrans_tloop

notrans_next_col:
	ADD R6, R1, R1             // A base → next column
	SUB $1, R4, R4
	CBNZ R4, notrans_col

notrans_done:
	RET
