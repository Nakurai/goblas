#include "textflag.h"

// func sssqUnitNEON(n int, x *float32) float32
//
// Unit-stride sum of squares (Sum x[i]^2). Four .S4 accumulators, a 16-wide
// main loop, a 4-wide loop, and a scalar tail, with VFMLA(x, x) doing
// acc += x*x. The caller (Snrm2) takes the square root and guards against the
// overflow/underflow this naive accumulation can suffer for extreme inputs.
TEXT ·sssqUnitNEON(SB), NOSPLIT, $0-20
	MOVD n+0(FP), R0
	MOVD x+8(FP), R1

	FMOVS $1.0, F17           // ones, for the accumulator fold
	VDUP  V17.S[0], V17.S4

	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16

	LSR  $4, R0, R3           // R3 = n / 16
	CBZ  R3, ssq_rem4
ssq_loop16:
	VLD1.P 16(R1), [V4.S4]
	VFMLA  V4.S4, V4.S4, V0.S4
	VLD1.P 16(R1), [V5.S4]
	VFMLA  V5.S4, V5.S4, V1.S4
	VLD1.P 16(R1), [V6.S4]
	VFMLA  V6.S4, V6.S4, V2.S4
	VLD1.P 16(R1), [V7.S4]
	VFMLA  V7.S4, V7.S4, V3.S4
	SUB    $1, R3, R3
	CBNZ   R3, ssq_loop16

ssq_rem4:
	AND  $15, R0, R0          // R0 = n % 16
	LSR  $2, R0, R5           // quads remaining
	CBZ  R5, ssq_reduce
ssq_loop4:
	VLD1.P 16(R1), [V4.S4]
	VFMLA  V4.S4, V4.S4, V0.S4
	SUB    $1, R5, R5
	CBNZ   R5, ssq_loop4

ssq_reduce:
	// Fold V1,V2,V3 into V0 (ones in V17), then sum V0's 4 lanes.
	VFMLA V17.S4, V1.S4, V0.S4
	VFMLA V17.S4, V2.S4, V0.S4
	VFMLA V17.S4, V3.S4, V0.S4
	VDUP  V0.S[1], V1.S4
	VDUP  V0.S[2], V2.S4
	VDUP  V0.S[3], V3.S4
	FADDS F1, F0, F0
	FADDS F2, F0, F0
	FADDS F3, F0, F0

	// Scalar tail: n % 4 elements.
	AND   $3, R0, R0
	CBZ   R0, ssq_done
ssq_tail:
	FMOVS (R1), F2
	FMULS F2, F2, F2
	FADDS F2, F0, F0
	ADD   $4, R1, R1
	SUB   $1, R0, R0
	CBNZ  R0, ssq_tail

ssq_done:
	FMOVS F0, ret+16(FP)
	RET
