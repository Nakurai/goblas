#include "textflag.h"

// func dscalUnitNEON(n int, alpha float64, x *float64)
//
// Unit-stride x *= alpha. Go's arm64 assembler has no vector FP multiply, so
// each chunk is computed as 0 + x*alpha via VFMLA into a freshly zeroed
// register. Processes 4 elements per iteration plus a 2-wide step and a tail.
TEXT ·dscalUnitNEON(SB), NOSPLIT, $0-24
	MOVD  n+0(FP), R0
	FMOVD alpha+8(FP), F0
	MOVD  x+16(FP), R1
	VDUP  V0.D[0], V16.D2     // V16 = [alpha, alpha]

	LSR  $2, R0, R3           // R3 = n / 4
	CBZ  R3, scal_rem2
scal_loop4:
	VLD1   (R1), [V4.D2]
	VEOR   V6.B16, V6.B16, V6.B16
	VFMLA  V16.D2, V4.D2, V6.D2   // V6 = x*alpha
	VST1.P [V6.D2], 16(R1)
	VLD1   (R1), [V5.D2]
	VEOR   V7.B16, V7.B16, V7.B16
	VFMLA  V16.D2, V5.D2, V7.D2
	VST1.P [V7.D2], 16(R1)
	SUB    $1, R3, R3
	CBNZ   R3, scal_loop4

scal_rem2:
	AND  $3, R0, R0           // R0 = n % 4
	LSR  $1, R0, R5
	CBZ  R5, scal_tail
scal_loop2:
	VLD1   (R1), [V4.D2]
	VEOR   V6.B16, V6.B16, V6.B16
	VFMLA  V16.D2, V4.D2, V6.D2
	VST1.P [V6.D2], 16(R1)
	SUB    $1, R5, R5
	CBNZ   R5, scal_loop2

scal_tail:
	AND   $1, R0, R0
	CBZ   R0, scal_done
	FMOVD (R1), F2
	FMULD F0, F2, F2          // F2 = alpha*x
	FMOVD F2, (R1)
scal_done:
	RET
