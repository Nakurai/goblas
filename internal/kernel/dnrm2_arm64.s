#include "textflag.h"

// func dssqUnitNEON(n int, x *float64) float64
//
// Unit-stride sum of squares (Sum x[i]^2). Same four-accumulator / 8-wide /
// 2-wide / scalar-tail structure as the ddot kernel, with VFMLA(x, x) doing
// acc += x*x. The caller (Dnrm2) takes the square root and guards against the
// overflow/underflow this naive accumulation can suffer for extreme inputs.
TEXT ·dssqUnitNEON(SB), NOSPLIT, $0-24
	MOVD n+0(FP), R0
	MOVD x+8(FP), R1

	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16

	LSR  $3, R0, R3           // R3 = n / 8
	CBZ  R3, rem2
loop8:
	VLD1.P 16(R1), [V4.D2]
	VFMLA  V4.D2, V4.D2, V0.D2
	VLD1.P 16(R1), [V5.D2]
	VFMLA  V5.D2, V5.D2, V1.D2
	VLD1.P 16(R1), [V6.D2]
	VFMLA  V6.D2, V6.D2, V2.D2
	VLD1.P 16(R1), [V7.D2]
	VFMLA  V7.D2, V7.D2, V3.D2
	SUB    $1, R3, R3
	CBNZ   R3, loop8

rem2:
	AND  $7, R0, R0           // R0 = n % 8
	LSR  $1, R0, R5           // pairs remaining
	CBZ  R5, reduce
loop2:
	VLD1.P 16(R1), [V4.D2]
	VFMLA  V4.D2, V4.D2, V0.D2
	SUB    $1, R5, R5
	CBNZ   R5, loop2

reduce:
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

	// Scalar tail: one element when n is odd.
	AND   $1, R0, R0
	CBZ   R0, done
	FMOVD (R1), F2
	FMULD F2, F2, F2
	FADDD F2, F0, F0

done:
	FMOVD F0, ret+16(FP)
	RET
