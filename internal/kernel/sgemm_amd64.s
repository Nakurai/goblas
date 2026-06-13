//go:build amd64

#include "textflag.h"

// func sgemmKernel8x8AVX2(k int, a, b, c *float32, ldc int)
//
// Single-precision micro-kernel: C[8x8] += A_panel * B_panel, where:
//   - a points to a packed A micro-panel: k slices of 8 contiguous float32
//     (column fragment of A, alpha already folded in by the packer)
//   - b points to a packed B micro-panel: k slices of 8 contiguous float32
//     (row fragment of B)
//   - c points to C(0,0) of the tile, column-major with leading dimension ldc
//     (in elements)
//
// A YMM register holds 8 float32, so the 8-row tile column is exactly one YMM.
// Register plan (16 YMM registers):
//   Y0..Y7   8 accumulators: column j lives in Y(j) (all 8 rows)
//   Y8       current 8-row slice of A
//   Y9, Y10  broadcasts of B elements (two at a time, for ILP)
//
// Each k-iteration: 1 A load + 8 B broadcasts + 8 VFMADD231PS (8 lanes each)
// = 128 FLOPs. This is the float32 analogue of the 8x4 float64 AVX2 kernel.
//
// Note: Go's assembler uses reversed operand order vs Intel syntax:
//   VFMADD231PS Y9, Y8, Y0  means  Y0 += Y8 * Y9
//   VADDPS      Y0, Y8, Y8  means  Y8 = Y8 + Y0
TEXT ·sgemmKernel8x8AVX2(SB), NOSPLIT, $0-40
	MOVQ k+0(FP), CX
	MOVQ a+8(FP), SI
	MOVQ b+16(FP), DI
	MOVQ c+24(FP), DX
	MOVQ ldc+32(FP), R8
	SHLQ $2, R8               // ldc in bytes (float32 = 4)

	// Zero the 8 accumulators.
	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	VXORPS Y4, Y4, Y4
	VXORPS Y5, Y5, Y5
	VXORPS Y6, Y6, Y6
	VXORPS Y7, Y7, Y7

	TESTQ CX, CX
	JZ    writeback

kloop:
	// 8-row A slice (one YMM).
	VMOVUPS (SI), Y8
	ADDQ    $32, SI

	// 8 columns, broadcasting two B elements at a time into Y9/Y10 so
	// consecutive FMAs hit different accumulators.
	VBROADCASTSS (DI), Y9
	VBROADCASTSS 4(DI), Y10
	VFMADD231PS  Y9, Y8, Y0
	VFMADD231PS  Y10, Y8, Y1
	VBROADCASTSS 8(DI), Y9
	VBROADCASTSS 12(DI), Y10
	VFMADD231PS  Y9, Y8, Y2
	VFMADD231PS  Y10, Y8, Y3
	VBROADCASTSS 16(DI), Y9
	VBROADCASTSS 20(DI), Y10
	VFMADD231PS  Y9, Y8, Y4
	VFMADD231PS  Y10, Y8, Y5
	VBROADCASTSS 24(DI), Y9
	VBROADCASTSS 28(DI), Y10
	VFMADD231PS  Y9, Y8, Y6
	VFMADD231PS  Y10, Y8, Y7
	ADDQ         $32, DI

	DECQ CX
	JNZ  kloop

writeback:
	// C(:,j) += acc, one YMM per column.
	VMOVUPS (DX), Y8
	VADDPS  Y0, Y8, Y8
	VMOVUPS Y8, (DX)

	ADDQ    R8, DX
	VMOVUPS (DX), Y8
	VADDPS  Y1, Y8, Y8
	VMOVUPS Y8, (DX)

	ADDQ    R8, DX
	VMOVUPS (DX), Y8
	VADDPS  Y2, Y8, Y8
	VMOVUPS Y8, (DX)

	ADDQ    R8, DX
	VMOVUPS (DX), Y8
	VADDPS  Y3, Y8, Y8
	VMOVUPS Y8, (DX)

	ADDQ    R8, DX
	VMOVUPS (DX), Y8
	VADDPS  Y4, Y8, Y8
	VMOVUPS Y8, (DX)

	ADDQ    R8, DX
	VMOVUPS (DX), Y8
	VADDPS  Y5, Y8, Y8
	VMOVUPS Y8, (DX)

	ADDQ    R8, DX
	VMOVUPS (DX), Y8
	VADDPS  Y6, Y8, Y8
	VMOVUPS Y8, (DX)

	ADDQ    R8, DX
	VMOVUPS (DX), Y8
	VADDPS  Y7, Y8, Y8
	VMOVUPS Y8, (DX)

	// Clear upper YMM halves before returning to Go's SSE world.
	VZEROUPPER
	RET
