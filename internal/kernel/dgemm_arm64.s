#include "textflag.h"

// func dgemmKernel8x4(k int, a, b, c *float64, ldc int)
//
// Micro-kernel: C[8x4] += A_panel * B_panel, where:
//   - a points to a packed A micro-panel: k slices of 8 contiguous float64
//     (column fragment of A, alpha already folded in by the packer)
//   - b points to a packed B micro-panel: k slices of 4 contiguous float64
//     (row fragment of B)
//   - c points to C(0,0) of the tile, column-major with leading dimension ldc
//     (in elements)
//
// Register plan (32 NEON regs, 2 float64 each):
//   V0..V15  16 accumulators: column j of the tile lives in V(4j)..V(4j+3)
//   V16..V19 current 8-row slice of A
//   V20      broadcast of one B element (reused per column)
//   V21..V24 C column staging for the writeback
//   V31      [1.0, 1.0] — used to fold accumulators into C via VFMLA,
//            because Go's assembler has no vector FP add (VADD .D2 is integer).
//
// The k-loop body does 4 loads + 4 broadcasts + 16 VFMLA = 32 FLOPs/2 elements
// per VFMLA lane pair, i.e. 64 FLOPs per iteration.
TEXT ·dgemmKernel8x4(SB), NOSPLIT, $0-40
	MOVD k+0(FP), R0
	MOVD a+8(FP), R1
	MOVD b+16(FP), R2
	MOVD c+24(FP), R3
	MOVD ldc+32(FP), R4
	LSL  $3, R4, R4           // ldc in bytes

	// Zero the 16 accumulators.
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16
	VEOR V4.B16, V4.B16, V4.B16
	VEOR V5.B16, V5.B16, V5.B16
	VEOR V6.B16, V6.B16, V6.B16
	VEOR V7.B16, V7.B16, V7.B16
	VEOR V8.B16, V8.B16, V8.B16
	VEOR V9.B16, V9.B16, V9.B16
	VEOR V10.B16, V10.B16, V10.B16
	VEOR V11.B16, V11.B16, V11.B16
	VEOR V12.B16, V12.B16, V12.B16
	VEOR V13.B16, V13.B16, V13.B16
	VEOR V14.B16, V14.B16, V14.B16
	VEOR V15.B16, V15.B16, V15.B16

	CBZ R0, writeback

kloop:
	// Load the 8-row A slice (V16..V19) and the 4 B values (V20..V21) with
	// wide vector loads, then broadcast each B lane into its own register
	// (V22..V25) so the 16 VFMLAs below have no false dependencies.
	VLD1.P 64(R1), [V16.D2, V17.D2, V18.D2, V19.D2]
	FMOVD  (R2), F22
	FMOVD  8(R2), F23
	FMOVD  16(R2), F24
	FMOVD  24(R2), F25
	ADD    $32, R2, R2
	VDUP   V22.D[0], V22.D2
	VDUP   V23.D[0], V23.D2
	VDUP   V24.D[0], V24.D2
	VDUP   V25.D[0], V25.D2

	// Interleave columns so consecutive instructions hit different
	// accumulators (helps the OoO scheduler keep all FMA pipes busy).
	VFMLA  V22.D2, V16.D2, V0.D2
	VFMLA  V23.D2, V16.D2, V4.D2
	VFMLA  V24.D2, V16.D2, V8.D2
	VFMLA  V25.D2, V16.D2, V12.D2

	VFMLA  V22.D2, V17.D2, V1.D2
	VFMLA  V23.D2, V17.D2, V5.D2
	VFMLA  V24.D2, V17.D2, V9.D2
	VFMLA  V25.D2, V17.D2, V13.D2

	VFMLA  V22.D2, V18.D2, V2.D2
	VFMLA  V23.D2, V18.D2, V6.D2
	VFMLA  V24.D2, V18.D2, V10.D2
	VFMLA  V25.D2, V18.D2, V14.D2

	VFMLA  V22.D2, V19.D2, V3.D2
	VFMLA  V23.D2, V19.D2, V7.D2
	VFMLA  V24.D2, V19.D2, V11.D2
	VFMLA  V25.D2, V19.D2, V15.D2

	SUB  $1, R0, R0
	CBNZ R0, kloop

writeback:
	// V31 = [1.0, 1.0]; C(:,j) += acc via VFMLA (no vector FP add available).
	FMOVD $1.0, F31
	VDUP  V31.D[0], V31.D2

	// Column 0 at R3.
	MOVD R3, R5
	VLD1 (R5), [V21.D2, V22.D2, V23.D2, V24.D2]
	VFMLA V31.D2, V0.D2, V21.D2
	VFMLA V31.D2, V1.D2, V22.D2
	VFMLA V31.D2, V2.D2, V23.D2
	VFMLA V31.D2, V3.D2, V24.D2
	VST1 [V21.D2, V22.D2, V23.D2, V24.D2], (R5)

	// Column 1 at R3 + ldc.
	ADD  R4, R3, R5
	VLD1 (R5), [V21.D2, V22.D2, V23.D2, V24.D2]
	VFMLA V31.D2, V4.D2, V21.D2
	VFMLA V31.D2, V5.D2, V22.D2
	VFMLA V31.D2, V6.D2, V23.D2
	VFMLA V31.D2, V7.D2, V24.D2
	VST1 [V21.D2, V22.D2, V23.D2, V24.D2], (R5)

	// Column 2 at R3 + 2*ldc.
	ADD  R4, R5, R5
	VLD1 (R5), [V21.D2, V22.D2, V23.D2, V24.D2]
	VFMLA V31.D2, V8.D2, V21.D2
	VFMLA V31.D2, V9.D2, V22.D2
	VFMLA V31.D2, V10.D2, V23.D2
	VFMLA V31.D2, V11.D2, V24.D2
	VST1 [V21.D2, V22.D2, V23.D2, V24.D2], (R5)

	// Column 3 at R3 + 3*ldc.
	ADD  R4, R5, R5
	VLD1 (R5), [V21.D2, V22.D2, V23.D2, V24.D2]
	VFMLA V31.D2, V12.D2, V21.D2
	VFMLA V31.D2, V13.D2, V22.D2
	VFMLA V31.D2, V14.D2, V23.D2
	VFMLA V31.D2, V15.D2, V24.D2
	VST1 [V21.D2, V22.D2, V23.D2, V24.D2], (R5)

	RET
