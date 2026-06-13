//go:build accelbench && darwin

// Package accel wraps Apple Accelerate's CBLAS for benchmark comparisons only.
// It is excluded from normal builds by the accelbench build tag so the library
// itself stays CGo-free.
package accel

/*
#cgo LDFLAGS: -framework Accelerate
#include <Accelerate/Accelerate.h>
*/
import "C"

import "unsafe"

// Dgemm computes C = alpha*A*B + beta*C (column-major, NoTrans) via Accelerate.
func Dgemm(m, n, k int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int) {
	C.cblas_dgemm(
		C.CblasColMajor, C.CblasNoTrans, C.CblasNoTrans,
		C.int(m), C.int(n), C.int(k),
		C.double(alpha),
		(*C.double)(unsafe.Pointer(&a[0])), C.int(lda),
		(*C.double)(unsafe.Pointer(&b[0])), C.int(ldb),
		C.double(beta),
		(*C.double)(unsafe.Pointer(&c[0])), C.int(ldc),
	)
}

// Sgemm computes C = alpha*A*B + beta*C (column-major, NoTrans) in single
// precision via Accelerate.
func Sgemm(m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	C.cblas_sgemm(
		C.CblasColMajor, C.CblasNoTrans, C.CblasNoTrans,
		C.int(m), C.int(n), C.int(k),
		C.float(alpha),
		(*C.float)(unsafe.Pointer(&a[0])), C.int(lda),
		(*C.float)(unsafe.Pointer(&b[0])), C.int(ldb),
		C.float(beta),
		(*C.float)(unsafe.Pointer(&c[0])), C.int(ldc),
	)
}
