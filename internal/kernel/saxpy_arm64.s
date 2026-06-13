#include "textflag.h"

// func saxpyUnitNEON(n int, alpha float32, x, y *float32)
//
// Unit-stride y += alpha*x. Broadcasts alpha across a .S4 vector and processes
// 8 elements per iteration, with a 4-wide step and a scalar tail.
TEXT ·saxpyUnitNEON(SB), NOSPLIT, $0-32
	MOVD  n+0(FP), R0
	FMOVS alpha+8(FP), F0
	MOVD  x+16(FP), R1
	MOVD  y+24(FP), R2
	VDUP  V0.S[0], V16.S4     // V16 = [alpha, alpha, alpha, alpha]

	LSR  $3, R0, R3           // R3 = n / 8
	CBZ  R3, axpy_rem4
axpy_loop8:
	VLD1.P 16(R1), [V4.S4]
	VLD1   (R2), [V6.S4]
	VFMLA  V16.S4, V4.S4, V6.S4
	VST1.P [V6.S4], 16(R2)
	VLD1.P 16(R1), [V5.S4]
	VLD1   (R2), [V7.S4]
	VFMLA  V16.S4, V5.S4, V7.S4
	VST1.P [V7.S4], 16(R2)
	SUB    $1, R3, R3
	CBNZ   R3, axpy_loop8

axpy_rem4:
	AND  $7, R0, R0           // R0 = n % 8
	LSR  $2, R0, R5           // quads remaining
	CBZ  R5, axpy_tail
axpy_loop4:
	VLD1.P 16(R1), [V4.S4]
	VLD1   (R2), [V6.S4]
	VFMLA  V16.S4, V4.S4, V6.S4
	VST1.P [V6.S4], 16(R2)
	SUB    $1, R5, R5
	CBNZ   R5, axpy_loop4

axpy_tail:
	AND   $3, R0, R0          // R0 = n % 4
	CBZ   R0, axpy_done
axpy_tloop:
	FMOVS (R1), F2
	FMOVS (R2), F3
	FMULS F0, F2, F2          // F2 = alpha*x
	FADDS F2, F3, F3          // F3 = y + alpha*x
	FMOVS F3, (R2)
	ADD   $4, R1, R1
	ADD   $4, R2, R2
	SUB   $1, R0, R0
	CBNZ  R0, axpy_tloop
axpy_done:
	RET
