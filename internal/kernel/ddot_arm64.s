#include "textflag.h"

// func ddotUnitNEON(n int, x, y *float64) float64
//
// Unit-stride float64 dot product. Processes 8 elements per iteration with four
// independent NEON accumulators (each holding 2 doubles) to expose instruction-
// level parallelism, then a 2-wide loop and a scalar tail for the remainder.
//
// Note: Go's arm64 assembler has no vector floating-point add (VADD on .D2 is an
// integer add). Accumulators are therefore reduced to scalars via VDUP + the
// scalar FADDD, which avoids any vector FP-add and is done once at the end.
TEXT ·ddotUnitNEON(SB), NOSPLIT, $0-32
	MOVD n+0(FP), R0
	MOVD x+8(FP), R1
	MOVD y+16(FP), R2

	// Zero the four vector accumulators.
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16

	LSR  $3, R0, R3            // R3 = n / 8
	CBZ  R3, rem2
loop8:
	VLD1.P 16(R1), [V4.D2]
	VLD1.P 16(R2), [V6.D2]
	VFMLA  V6.D2, V4.D2, V0.D2    // V0 += x[0:2]*y[0:2]
	VLD1.P 16(R1), [V5.D2]
	VLD1.P 16(R2), [V7.D2]
	VFMLA  V7.D2, V5.D2, V1.D2    // V1 += x[2:4]*y[2:4]
	VLD1.P 16(R1), [V8.D2]
	VLD1.P 16(R2), [V10.D2]
	VFMLA  V10.D2, V8.D2, V2.D2   // V2 += x[4:6]*y[4:6]
	VLD1.P 16(R1), [V9.D2]
	VLD1.P 16(R2), [V11.D2]
	VFMLA  V11.D2, V9.D2, V3.D2   // V3 += x[6:8]*y[6:8]
	SUB    $1, R3, R3
	CBNZ   R3, loop8

rem2:
	AND  $7, R0, R0           // R0 = n % 8 (remaining elements)
	LSR  $1, R0, R5           // R5 = pairs remaining
	CBZ  R5, reduce
loop2:
	VLD1.P 16(R1), [V4.D2]
	VLD1.P 16(R2), [V6.D2]
	VFMLA  V6.D2, V4.D2, V0.D2
	SUB    $1, R5, R5
	CBNZ   R5, loop2

reduce:
	// Reduce each accumulator's two lanes to a scalar (Fi aliases Vi's low
	// lane; VDUP brings the high lane into a temp's low lane), then sum.
	VDUP  V0.D[1], V4.D2
	VDUP  V1.D[1], V5.D2
	VDUP  V2.D[1], V6.D2
	VDUP  V3.D[1], V7.D2
	FADDD F4, F0, F0
	FADDD F5, F1, F1
	FADDD F6, F2, F2
	FADDD F7, F3, F3
	FADDD F1, F0, F0
	FADDD F3, F2, F2
	FADDD F2, F0, F0

	// Scalar tail: one element left when n was odd.
	AND   $1, R0, R0
	CBZ   R0, done
	FMOVD (R1), F2
	FMOVD (R2), F3
	FMULD F3, F2, F2
	FADDD F2, F0, F0

done:
	FMOVD F0, ret+24(FP)
	RET
