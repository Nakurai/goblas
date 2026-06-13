#include "textflag.h"

// func dgemmKernel8x6(k int, a, b, c *float64, ldc int)
//
// Micro-kernel: C[8x6] += A_panel * B_panel, where:
//   - a points to a packed A micro-panel: k slices of 8 contiguous float64
//     (column fragment of A, alpha already folded in by the packer)
//   - b points to a packed B micro-panel: k slices of 6 contiguous float64
//     (row fragment of B)
//   - c points to C(0,0) of the tile, column-major with leading dimension ldc
//     (in elements)
//
// Compared with the 8x4 kernel this reuses each A load across 6 columns
// instead of 4: 24 VFMLA per 4-register A load (96 FLOPs/iteration vs 64),
// raising compute-per-load at the cost of using 24 of the 32 NEON registers
// as accumulators.
//
// Register plan:
//   V0..V23  24 accumulators: column j of the tile lives in V(4j)..V(4j+3)
//   V24..V27 current 8-row slice of A
//   V28..V31 B broadcasts: b0..b3 first, then b4..b5 reuse V28..V29
//   writeback: V24 = [1.0, 1.0], V28..V31 stage C columns (VFMLA fold,
//   because Go's assembler has no vector FP add — VADD .D2 is integer).
TEXT ·dgemmKernel8x6(SB), NOSPLIT, $0-40
	MOVD k+0(FP), R0
	MOVD a+8(FP), R1
	MOVD b+16(FP), R2
	MOVD c+24(FP), R3
	MOVD ldc+32(FP), R4
	LSL  $3, R4, R4           // ldc in bytes

	// Zero the 24 accumulators.
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
	VEOR V16.B16, V16.B16, V16.B16
	VEOR V17.B16, V17.B16, V17.B16
	VEOR V18.B16, V18.B16, V18.B16
	VEOR V19.B16, V19.B16, V19.B16
	VEOR V20.B16, V20.B16, V20.B16
	VEOR V21.B16, V21.B16, V21.B16
	VEOR V22.B16, V22.B16, V22.B16
	VEOR V23.B16, V23.B16, V23.B16

	CBZ R0, writeback

kloop:
	// 8-row A slice into V24..V27; first four B values broadcast into
	// V28..V31, used by columns 0-3.
	VLD1.P 64(R1), [V24.D2, V25.D2, V26.D2, V27.D2]
	FMOVD  (R2), F28
	FMOVD  8(R2), F29
	FMOVD  16(R2), F30
	FMOVD  24(R2), F31
	VDUP   V28.D[0], V28.D2
	VDUP   V29.D[0], V29.D2
	VDUP   V30.D[0], V30.D2
	VDUP   V31.D[0], V31.D2

	// Interleave columns so consecutive instructions hit different
	// accumulators (keeps all FMA pipes busy).
	VFMLA  V28.D2, V24.D2, V0.D2
	VFMLA  V29.D2, V24.D2, V4.D2
	VFMLA  V30.D2, V24.D2, V8.D2
	VFMLA  V31.D2, V24.D2, V12.D2

	VFMLA  V28.D2, V25.D2, V1.D2
	VFMLA  V29.D2, V25.D2, V5.D2
	VFMLA  V30.D2, V25.D2, V9.D2
	VFMLA  V31.D2, V25.D2, V13.D2

	VFMLA  V28.D2, V26.D2, V2.D2
	VFMLA  V29.D2, V26.D2, V6.D2
	VFMLA  V30.D2, V26.D2, V10.D2
	VFMLA  V31.D2, V26.D2, V14.D2

	VFMLA  V28.D2, V27.D2, V3.D2
	VFMLA  V29.D2, V27.D2, V7.D2
	VFMLA  V30.D2, V27.D2, V11.D2
	VFMLA  V31.D2, V27.D2, V15.D2

	// Last two B values reuse V28/V29 (columns 0-3 are done with them).
	FMOVD  32(R2), F28
	FMOVD  40(R2), F29
	ADD    $48, R2, R2
	VDUP   V28.D[0], V28.D2
	VDUP   V29.D[0], V29.D2

	VFMLA  V28.D2, V24.D2, V16.D2
	VFMLA  V29.D2, V24.D2, V20.D2
	VFMLA  V28.D2, V25.D2, V17.D2
	VFMLA  V29.D2, V25.D2, V21.D2
	VFMLA  V28.D2, V26.D2, V18.D2
	VFMLA  V29.D2, V26.D2, V22.D2
	VFMLA  V28.D2, V27.D2, V19.D2
	VFMLA  V29.D2, V27.D2, V23.D2

	SUB  $1, R0, R0
	CBNZ R0, kloop

writeback:
	// V24 = [1.0, 1.0]; C(:,j) += acc via VFMLA against ones.
	FMOVD $1.0, F24
	VDUP  V24.D[0], V24.D2

	// Column 0 at R3.
	MOVD R3, R5
	VLD1 (R5), [V28.D2, V29.D2, V30.D2, V31.D2]
	VFMLA V24.D2, V0.D2, V28.D2
	VFMLA V24.D2, V1.D2, V29.D2
	VFMLA V24.D2, V2.D2, V30.D2
	VFMLA V24.D2, V3.D2, V31.D2
	VST1 [V28.D2, V29.D2, V30.D2, V31.D2], (R5)

	// Column 1 at R3 + ldc.
	ADD  R4, R3, R5
	VLD1 (R5), [V28.D2, V29.D2, V30.D2, V31.D2]
	VFMLA V24.D2, V4.D2, V28.D2
	VFMLA V24.D2, V5.D2, V29.D2
	VFMLA V24.D2, V6.D2, V30.D2
	VFMLA V24.D2, V7.D2, V31.D2
	VST1 [V28.D2, V29.D2, V30.D2, V31.D2], (R5)

	// Column 2.
	ADD  R4, R5, R5
	VLD1 (R5), [V28.D2, V29.D2, V30.D2, V31.D2]
	VFMLA V24.D2, V8.D2, V28.D2
	VFMLA V24.D2, V9.D2, V29.D2
	VFMLA V24.D2, V10.D2, V30.D2
	VFMLA V24.D2, V11.D2, V31.D2
	VST1 [V28.D2, V29.D2, V30.D2, V31.D2], (R5)

	// Column 3.
	ADD  R4, R5, R5
	VLD1 (R5), [V28.D2, V29.D2, V30.D2, V31.D2]
	VFMLA V24.D2, V12.D2, V28.D2
	VFMLA V24.D2, V13.D2, V29.D2
	VFMLA V24.D2, V14.D2, V30.D2
	VFMLA V24.D2, V15.D2, V31.D2
	VST1 [V28.D2, V29.D2, V30.D2, V31.D2], (R5)

	// Column 4.
	ADD  R4, R5, R5
	VLD1 (R5), [V28.D2, V29.D2, V30.D2, V31.D2]
	VFMLA V24.D2, V16.D2, V28.D2
	VFMLA V24.D2, V17.D2, V29.D2
	VFMLA V24.D2, V18.D2, V30.D2
	VFMLA V24.D2, V19.D2, V31.D2
	VST1 [V28.D2, V29.D2, V30.D2, V31.D2], (R5)

	// Column 5.
	ADD  R4, R5, R5
	VLD1 (R5), [V28.D2, V29.D2, V30.D2, V31.D2]
	VFMLA V24.D2, V20.D2, V28.D2
	VFMLA V24.D2, V21.D2, V29.D2
	VFMLA V24.D2, V22.D2, V30.D2
	VFMLA V24.D2, V23.D2, V31.D2
	VST1 [V28.D2, V29.D2, V30.D2, V31.D2], (R5)

	RET
