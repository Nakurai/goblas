#include "textflag.h"

// func sgemmKernel8x8(k int, a, b, c *float32, ldc int)
//
// Single-precision micro-kernel: C[8x8] += A_panel * B_panel, where:
//   - a points to a packed A micro-panel: k slices of 8 contiguous float32
//     (column fragment of A, alpha already folded in by the packer)
//   - b points to a packed B micro-panel: k slices of 8 contiguous float32
//     (row fragment of B)
//   - c points to C(0,0) of the tile, column-major with leading dimension ldc
//     (in elements)
//
// A NEON .S4 register holds 4 float32, so the 8 rows of the tile occupy 2
// registers per column. The tile is 8 rows x 8 columns = 64 elements in 16
// accumulators, leaving room for the A slice and 8 B broadcasts. This is the
// float32 analogue of the 8x6 float64 kernel; see go-arm64-asm-neon-f32 notes:
// VADD.S4 is integer, so the C writeback folds via VFMLA against a ones vector.
//
// Register plan:
//   V0..V15  16 accumulators: column j lives in V(2j) (rows 0-3) and
//            V(2j+1) (rows 4-7)
//   V16,V17  current 8-row slice of A (rows 0-3, 4-7)
//   V18..V25 the 8 B values, each broadcast across 4 lanes
//   writeback: V16 = [1,1,1,1]; V18,V19 stage each C column
TEXT ·sgemmKernel8x8(SB), NOSPLIT, $0-40
	MOVD k+0(FP), R0
	MOVD a+8(FP), R1
	MOVD b+16(FP), R2
	MOVD c+24(FP), R3
	MOVD ldc+32(FP), R4
	LSL  $2, R4, R4           // ldc in bytes (float32 = 4)

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
	// 8-row A slice into V16,V17; the 8 B values broadcast into V18..V25.
	VLD1.P 32(R1), [V16.S4, V17.S4]
	FMOVS  (R2), F18
	FMOVS  4(R2), F19
	FMOVS  8(R2), F20
	FMOVS  12(R2), F21
	FMOVS  16(R2), F22
	FMOVS  20(R2), F23
	FMOVS  24(R2), F24
	FMOVS  28(R2), F25
	ADD    $32, R2, R2
	VDUP   V18.S[0], V18.S4
	VDUP   V19.S[0], V19.S4
	VDUP   V20.S[0], V20.S4
	VDUP   V21.S[0], V21.S4
	VDUP   V22.S[0], V22.S4
	VDUP   V23.S[0], V23.S4
	VDUP   V24.S[0], V24.S4
	VDUP   V25.S[0], V25.S4

	// Rows 0-3 of every column (A slice V16), then rows 4-7 (V17). Consecutive
	// instructions touch different accumulators to keep the FMA pipes busy.
	VFMLA V18.S4, V16.S4, V0.S4
	VFMLA V19.S4, V16.S4, V2.S4
	VFMLA V20.S4, V16.S4, V4.S4
	VFMLA V21.S4, V16.S4, V6.S4
	VFMLA V22.S4, V16.S4, V8.S4
	VFMLA V23.S4, V16.S4, V10.S4
	VFMLA V24.S4, V16.S4, V12.S4
	VFMLA V25.S4, V16.S4, V14.S4

	VFMLA V18.S4, V17.S4, V1.S4
	VFMLA V19.S4, V17.S4, V3.S4
	VFMLA V20.S4, V17.S4, V5.S4
	VFMLA V21.S4, V17.S4, V7.S4
	VFMLA V22.S4, V17.S4, V9.S4
	VFMLA V23.S4, V17.S4, V11.S4
	VFMLA V24.S4, V17.S4, V13.S4
	VFMLA V25.S4, V17.S4, V15.S4

	SUB  $1, R0, R0
	CBNZ R0, kloop

writeback:
	// V16 = [1,1,1,1]; C(:,j) += acc via VFMLA against ones.
	FMOVS $1.0, F16
	VDUP  V16.S[0], V16.S4

	MOVD R3, R5
	VLD1 (R5), [V18.S4, V19.S4]
	VFMLA V16.S4, V0.S4, V18.S4
	VFMLA V16.S4, V1.S4, V19.S4
	VST1 [V18.S4, V19.S4], (R5)

	ADD  R4, R3, R5
	VLD1 (R5), [V18.S4, V19.S4]
	VFMLA V16.S4, V2.S4, V18.S4
	VFMLA V16.S4, V3.S4, V19.S4
	VST1 [V18.S4, V19.S4], (R5)

	ADD  R4, R5, R5
	VLD1 (R5), [V18.S4, V19.S4]
	VFMLA V16.S4, V4.S4, V18.S4
	VFMLA V16.S4, V5.S4, V19.S4
	VST1 [V18.S4, V19.S4], (R5)

	ADD  R4, R5, R5
	VLD1 (R5), [V18.S4, V19.S4]
	VFMLA V16.S4, V6.S4, V18.S4
	VFMLA V16.S4, V7.S4, V19.S4
	VST1 [V18.S4, V19.S4], (R5)

	ADD  R4, R5, R5
	VLD1 (R5), [V18.S4, V19.S4]
	VFMLA V16.S4, V8.S4, V18.S4
	VFMLA V16.S4, V9.S4, V19.S4
	VST1 [V18.S4, V19.S4], (R5)

	ADD  R4, R5, R5
	VLD1 (R5), [V18.S4, V19.S4]
	VFMLA V16.S4, V10.S4, V18.S4
	VFMLA V16.S4, V11.S4, V19.S4
	VST1 [V18.S4, V19.S4], (R5)

	ADD  R4, R5, R5
	VLD1 (R5), [V18.S4, V19.S4]
	VFMLA V16.S4, V12.S4, V18.S4
	VFMLA V16.S4, V13.S4, V19.S4
	VST1 [V18.S4, V19.S4], (R5)

	ADD  R4, R5, R5
	VLD1 (R5), [V18.S4, V19.S4]
	VFMLA V16.S4, V14.S4, V18.S4
	VFMLA V16.S4, V15.S4, V19.S4
	VST1 [V18.S4, V19.S4], (R5)

	RET
