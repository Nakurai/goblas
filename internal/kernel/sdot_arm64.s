#include "textflag.h"

// func sdotUnitNEON(n int, x, y *float32) float32
//
// Unit-stride float32 dot product. Four independent .S4 accumulators process
// 16 elements per iteration, then a 4-wide loop and a scalar tail. The four
// accumulators are folded together with VFMLA against a ones vector (Go's arm64
// assembler has no vector FP add — VADD .S4 is integer), then the 4 lanes are
// reduced to a scalar.
TEXT ·sdotUnitNEON(SB), NOSPLIT, $0-28
	MOVD n+0(FP), R0
	MOVD x+8(FP), R1
	MOVD y+16(FP), R2

	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16

	LSR  $4, R0, R3           // R3 = n / 16
	CBZ  R3, dot_rem4
dot_loop16:
	VLD1.P 16(R1), [V4.S4]
	VLD1.P 16(R2), [V8.S4]
	VFMLA  V8.S4, V4.S4, V0.S4
	VLD1.P 16(R1), [V5.S4]
	VLD1.P 16(R2), [V9.S4]
	VFMLA  V9.S4, V5.S4, V1.S4
	VLD1.P 16(R1), [V6.S4]
	VLD1.P 16(R2), [V10.S4]
	VFMLA  V10.S4, V6.S4, V2.S4
	VLD1.P 16(R1), [V7.S4]
	VLD1.P 16(R2), [V11.S4]
	VFMLA  V11.S4, V7.S4, V3.S4
	SUB    $1, R3, R3
	CBNZ   R3, dot_loop16

dot_rem4:
	AND  $15, R0, R0          // R0 = n % 16
	LSR  $2, R0, R5           // quads remaining
	CBZ  R5, dot_reduce
dot_loop4:
	VLD1.P 16(R1), [V4.S4]
	VLD1.P 16(R2), [V8.S4]
	VFMLA  V8.S4, V4.S4, V0.S4
	SUB    $1, R5, R5
	CBNZ   R5, dot_loop4

dot_reduce:
	// Fold V1,V2,V3 into V0 (VFMLA against ones), then sum V0's 4 lanes.
	FMOVS $1.0, F16
	VDUP  V16.S[0], V16.S4
	VFMLA V16.S4, V1.S4, V0.S4
	VFMLA V16.S4, V2.S4, V0.S4
	VFMLA V16.S4, V3.S4, V0.S4
	VDUP  V0.S[1], V1.S4
	VDUP  V0.S[2], V2.S4
	VDUP  V0.S[3], V3.S4
	FADDS F1, F0, F0
	FADDS F2, F0, F0
	FADDS F3, F0, F0

	// Scalar tail: n % 4 elements.
	AND   $3, R0, R0
	CBZ   R0, dot_done
dot_tail:
	FMOVS (R1), F4
	FMOVS (R2), F5
	FMULS F5, F4, F4
	FADDS F4, F0, F0
	ADD   $4, R1, R1
	ADD   $4, R2, R2
	SUB   $1, R0, R0
	CBNZ  R0, dot_tail

dot_done:
	FMOVS F0, ret+24(FP)
	RET
