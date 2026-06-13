#include "textflag.h"

// func sscalUnitNEON(n int, alpha float32, x *float32)
//
// Unit-stride x *= alpha. Go's arm64 assembler has no vector FP multiply, so
// each chunk is computed as 0 + x*alpha via VFMLA into a freshly zeroed
// register. Processes 8 elements per iteration plus a 4-wide step and a tail.
TEXT ·sscalUnitNEON(SB), NOSPLIT, $0-24
	MOVD  n+0(FP), R0
	FMOVS alpha+8(FP), F0
	MOVD  x+16(FP), R1
	VDUP  V0.S[0], V16.S4     // V16 = [alpha, alpha, alpha, alpha]

	LSR  $3, R0, R3           // R3 = n / 8
	CBZ  R3, scal_rem4
scal_loop8:
	VLD1   (R1), [V4.S4]
	VEOR   V6.B16, V6.B16, V6.B16
	VFMLA  V16.S4, V4.S4, V6.S4   // V6 = x*alpha
	VST1.P [V6.S4], 16(R1)
	VLD1   (R1), [V5.S4]
	VEOR   V7.B16, V7.B16, V7.B16
	VFMLA  V16.S4, V5.S4, V7.S4
	VST1.P [V7.S4], 16(R1)
	SUB    $1, R3, R3
	CBNZ   R3, scal_loop8

scal_rem4:
	AND  $7, R0, R0           // R0 = n % 8
	LSR  $2, R0, R5           // quads remaining
	CBZ  R5, scal_tail
scal_loop4:
	VLD1   (R1), [V4.S4]
	VEOR   V6.B16, V6.B16, V6.B16
	VFMLA  V16.S4, V4.S4, V6.S4
	VST1.P [V6.S4], 16(R1)
	SUB    $1, R5, R5
	CBNZ   R5, scal_loop4

scal_tail:
	AND   $3, R0, R0          // R0 = n % 4
	CBZ   R0, scal_done
scal_tloop:
	FMOVS (R1), F2
	FMULS F0, F2, F2          // F2 = alpha*x
	FMOVS F2, (R1)
	ADD   $4, R1, R1
	SUB   $1, R0, R0
	CBNZ  R0, scal_tloop
scal_done:
	RET
