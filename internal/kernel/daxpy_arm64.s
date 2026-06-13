#include "textflag.h"

// func daxpyUnitNEON(n int, alpha float64, x, y *float64)
//
// Unit-stride y += alpha*x. Broadcasts alpha across a NEON vector and processes
// 4 elements per iteration, with a 2-wide step and a scalar tail.
TEXT ·daxpyUnitNEON(SB), NOSPLIT, $0-32
	MOVD  n+0(FP), R0
	FMOVD alpha+8(FP), F0
	MOVD  x+16(FP), R1
	MOVD  y+24(FP), R2
	VDUP  V0.D[0], V16.D2     // V16 = [alpha, alpha]

	LSR  $2, R0, R3           // R3 = n / 4
	CBZ  R3, axpy_rem2
axpy_loop4:
	VLD1.P 16(R1), [V4.D2]
	VLD1   (R2), [V6.D2]
	VFMLA  V16.D2, V4.D2, V6.D2
	VST1.P [V6.D2], 16(R2)
	VLD1.P 16(R1), [V5.D2]
	VLD1   (R2), [V7.D2]
	VFMLA  V16.D2, V5.D2, V7.D2
	VST1.P [V7.D2], 16(R2)
	SUB    $1, R3, R3
	CBNZ   R3, axpy_loop4

axpy_rem2:
	AND  $3, R0, R0           // R0 = n % 4
	LSR  $1, R0, R5           // pairs remaining
	CBZ  R5, axpy_tail
axpy_loop2:
	VLD1.P 16(R1), [V4.D2]
	VLD1   (R2), [V6.D2]
	VFMLA  V16.D2, V4.D2, V6.D2
	VST1.P [V6.D2], 16(R2)
	SUB    $1, R5, R5
	CBNZ   R5, axpy_loop2

axpy_tail:
	AND   $1, R0, R0
	CBZ   R0, axpy_done
	FMOVD (R1), F2
	FMOVD (R2), F3
	FMULD F0, F2, F2          // F2 = alpha*x
	FADDD F2, F3, F3          // F3 = y + alpha*x
	FMOVD F3, (R2)
axpy_done:
	RET
