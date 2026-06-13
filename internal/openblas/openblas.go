//go:build openblasbench

// Package openblas wraps OpenBLAS's CBLAS for benchmark comparisons only.
// It is excluded from normal builds by the openblasbench build tag so the
// library itself stays CGo-free. Requires: brew install openblas.
package openblas

/*
#cgo CFLAGS: -I/opt/homebrew/opt/openblas/include
#cgo LDFLAGS: -L/opt/homebrew/opt/openblas/lib -lopenblas
#include <cblas.h>
*/
import "C"

import "unsafe"

// Dgemm computes C = alpha*A*B + beta*C (column-major, NoTrans) via OpenBLAS.
func Dgemm(m, n, k int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int) {
	C.cblas_dgemm(
		C.CblasColMajor, C.CblasNoTrans, C.CblasNoTrans,
		C.blasint(m), C.blasint(n), C.blasint(k),
		C.double(alpha),
		(*C.double)(unsafe.Pointer(&a[0])), C.blasint(lda),
		(*C.double)(unsafe.Pointer(&b[0])), C.blasint(ldb),
		C.double(beta),
		(*C.double)(unsafe.Pointer(&c[0])), C.blasint(ldc),
	)
}
