//go:build amd64

#include "textflag.h"

// func dgemmKernel8x4AVX2(k int, a, b, c *float64, ldc int)
//
// Micro-kernel: C[8x4] += A_panel * B_panel, where:
//   - a points to a packed A micro-panel: k slices of 8 contiguous float64
//     (column fragment of A, alpha already folded in by the packer)
//   - b points to a packed B micro-panel: k slices of 4 contiguous float64
//     (row fragment of B)
//   - c points to C(0,0) of the tile, column-major with leading dimension ldc
//     (in elements)
//
// AVX2 holds 4 float64 per YMM register, so the 8-row tile column is a YMM
// pair. Register plan (16 YMM registers):
//   Y0..Y7   8 accumulators: column j lives in Y(2j) (rows 0-3) and
//            Y(2j+1) (rows 4-7)
//   Y8, Y9   current 8-row slice of A
//   Y10, Y11 broadcasts of B elements (two at a time)
//
// Each k-iteration: 2 A loads + 4 B broadcasts + 8 VFMADD231PD
// (4 lanes each) = 64 FLOPs.
//
// Note: Go's assembler uses reversed operand order vs Intel syntax:
//   VFMADD231PD Y10, Y8, Y0  means  Y0 += Y8 * Y10
//   VADDPD      Y0, Y8, Y8   means  Y8 = Y8 + Y0
TEXT ·dgemmKernel8x4AVX2(SB), NOSPLIT, $0-40
	MOVQ k+0(FP), CX
	MOVQ a+8(FP), SI
	MOVQ b+16(FP), DI
	MOVQ c+24(FP), DX
	MOVQ ldc+32(FP), R8
	SHLQ $3, R8               // ldc in bytes

	// Zero the 8 accumulators.
	VXORPD Y0, Y0, Y0
	VXORPD Y1, Y1, Y1
	VXORPD Y2, Y2, Y2
	VXORPD Y3, Y3, Y3
	VXORPD Y4, Y4, Y4
	VXORPD Y5, Y5, Y5
	VXORPD Y6, Y6, Y6
	VXORPD Y7, Y7, Y7

	TESTQ CX, CX
	JZ    writeback

kloop:
	// 8-row A slice.
	VMOVUPD (SI), Y8          // a[0:4]
	VMOVUPD 32(SI), Y9        // a[4:8]
	ADDQ    $64, SI

	// Columns 0 and 1.
	VBROADCASTSD (DI), Y10
	VBROADCASTSD 8(DI), Y11
	VFMADD231PD  Y10, Y8, Y0
	VFMADD231PD  Y10, Y9, Y1
	VFMADD231PD  Y11, Y8, Y2
	VFMADD231PD  Y11, Y9, Y3

	// Columns 2 and 3.
	VBROADCASTSD 16(DI), Y10
	VBROADCASTSD 24(DI), Y11
	ADDQ         $32, DI
	VFMADD231PD  Y10, Y8, Y4
	VFMADD231PD  Y10, Y9, Y5
	VFMADD231PD  Y11, Y8, Y6
	VFMADD231PD  Y11, Y9, Y7

	DECQ CX
	JNZ  kloop

writeback:
	// C(:,j) += acc, one YMM pair per column.
	VMOVUPD (DX), Y8
	VMOVUPD 32(DX), Y9
	VADDPD  Y0, Y8, Y8
	VADDPD  Y1, Y9, Y9
	VMOVUPD Y8, (DX)
	VMOVUPD Y9, 32(DX)

	ADDQ    R8, DX
	VMOVUPD (DX), Y8
	VMOVUPD 32(DX), Y9
	VADDPD  Y2, Y8, Y8
	VADDPD  Y3, Y9, Y9
	VMOVUPD Y8, (DX)
	VMOVUPD Y9, 32(DX)

	ADDQ    R8, DX
	VMOVUPD (DX), Y8
	VMOVUPD 32(DX), Y9
	VADDPD  Y4, Y8, Y8
	VADDPD  Y5, Y9, Y9
	VMOVUPD Y8, (DX)
	VMOVUPD Y9, 32(DX)

	ADDQ    R8, DX
	VMOVUPD (DX), Y8
	VMOVUPD 32(DX), Y9
	VADDPD  Y6, Y8, Y8
	VADDPD  Y7, Y9, Y9
	VMOVUPD Y8, (DX)
	VMOVUPD Y9, 32(DX)

	// Clear upper YMM halves before returning to Go's SSE world (avoids the
	// AVX-SSE transition penalty).
	VZEROUPPER
	RET
