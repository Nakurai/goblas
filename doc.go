// Package goblas is a pure-Go BLAS (Basic Linear Algebra Subprograms)
// implementation for float64, with hand-tuned ARM64 NEON assembly kernels on
// supported processors.
//
// It is pure Go by default — no CGo — so it cross-compiles anywhere and always
// runs. On Apple Silicon it dispatches the hot kernels to NEON assembly tuned
// for the host; on every other platform it uses the portable reference, which
// is identical in behavior, just slower.
//
// # Conventions
//
// Matrices are column-major (Fortran order): element A(i,j) is stored at
// a[i+j*lda], where lda — the leading dimension — is the column stride and must
// be at least the number of rows. Vectors carry an increment (incX); a negative
// increment traverses the vector from the high end. Operations follow standard
// BLAS semantics, e.g. Dgemm computes C = alpha*op(A)*op(B) + beta*C.
//
// The active kernel is chosen once at process start based on CPU detection.
package goblas
