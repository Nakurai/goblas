package main

import (
	"fmt"

	goblas "github.com/nakurai/goblas"
)

// enum decoders: map goblast's single-letter tokens to goblas's bool types.

func trans(s string) goblas.Transpose {
	if s == "T" {
		return goblas.Trans
	}
	return goblas.NoTrans
}

func uplo(s string) goblas.Uplo {
	if s == "U" {
		return goblas.Upper
	}
	return goblas.Lower
}

func diag(s string) goblas.Diag {
	if s == "U" {
		return goblas.Unit
	}
	return goblas.NonUnit
}

func side(s string) goblas.Side {
	if s == "L" {
		return goblas.Left
	}
	return goblas.Right
}

// to32 narrows a parsed float64 buffer to float32 for the S-prefixed routines
// (FORMAT.md §2: the file holds the exact value, narrow back after parsing).
func to32(xs []float64) []float32 {
	out := make([]float32, len(xs))
	for i, v := range xs {
		out[i] = float32(v)
	}
	return out
}

// dispatch reads a parsed case, runs the matching goblas routine, and writes
// the op's output field(s) into w. The input arrays already hold the full
// ld-/inc-strided buffer (FORMAT.md §3), so they are passed straight through;
// in/out operands are mutated in place and written back.
func dispatch(in *Doc, w *writer) error {
	op := in.Str("op")
	pad := in.hasTag("ld-padding")

	switch op {

	// ---------- Level 1, float64 ----------
	case "ddot":
		r := goblas.Ddot(in.Int("n"), in.Arr("x"), in.Int("incx"), in.Arr("y"), in.Int("incy"))
		w.float("result", r)
	case "daxpy":
		y := in.Arr("y")
		goblas.Daxpy(in.Int("n"), in.F64("alpha"), in.Arr("x"), in.Int("incx"), y, in.Int("incy"))
		w.arr("y", y)
	case "dscal":
		x := in.Arr("x")
		goblas.Dscal(in.Int("n"), in.F64("alpha"), x, in.Int("incx"))
		w.arr("x", x)
	case "dnrm2":
		w.float("result", goblas.Dnrm2(in.Int("n"), in.Arr("x"), in.Int("incx")))
	case "dasum":
		w.float("result", goblas.Dasum(in.Int("n"), in.Arr("x"), in.Int("incx")))
	case "idamax":
		w.int("result", goblas.Idamax(in.Int("n"), in.Arr("x"), in.Int("incx")))
	case "dcopy":
		y := in.Arr("y")
		goblas.Dcopy(in.Int("n"), in.Arr("x"), in.Int("incx"), y, in.Int("incy"))
		w.arr("y", y)
	case "dswap":
		x, y := in.Arr("x"), in.Arr("y")
		goblas.Dswap(in.Int("n"), x, in.Int("incx"), y, in.Int("incy"))
		w.arr("x", x)
		w.arr("y", y)

	// ---------- Level 1, float32 ----------
	case "sdot":
		r := goblas.Sdot(in.Int("n"), to32(in.Arr("x")), in.Int("incx"), to32(in.Arr("y")), in.Int("incy"))
		w.float("result", float64(r))
	case "saxpy":
		y := to32(in.Arr("y"))
		goblas.Saxpy(in.Int("n"), float32(in.F64("alpha")), to32(in.Arr("x")), in.Int("incx"), y, in.Int("incy"))
		w.arr32("y", y)
	case "sscal":
		x := to32(in.Arr("x"))
		goblas.Sscal(in.Int("n"), float32(in.F64("alpha")), x, in.Int("incx"))
		w.arr32("x", x)
	case "snrm2":
		w.float("result", float64(goblas.Snrm2(in.Int("n"), to32(in.Arr("x")), in.Int("incx"))))
	case "sasum":
		w.float("result", float64(goblas.Sasum(in.Int("n"), to32(in.Arr("x")), in.Int("incx"))))
	case "isamax":
		w.int("result", goblas.Isamax(in.Int("n"), to32(in.Arr("x")), in.Int("incx")))
	case "scopy":
		y := to32(in.Arr("y"))
		goblas.Scopy(in.Int("n"), to32(in.Arr("x")), in.Int("incx"), y, in.Int("incy"))
		w.arr32("y", y)
	case "sswap":
		x, y := to32(in.Arr("x")), to32(in.Arr("y"))
		goblas.Sswap(in.Int("n"), x, in.Int("incx"), y, in.Int("incy"))
		w.arr32("x", x)
		w.arr32("y", y)

	// ---------- Level 2, float64 ----------
	case "dgemv":
		y := in.Arr("y")
		goblas.Dgemv(trans(in.Str("transa")), in.Int("m"), in.Int("n"), in.F64("alpha"),
			in.Arr("a"), in.Int("lda"), in.Arr("x"), in.Int("incx"), in.F64("beta"), y, in.Int("incy"))
		w.arr("y", y)
		if pad {
			w.arr("a", in.Arr("a")) // read-only operand echoed back unchanged
		}
	case "dger":
		a := in.Arr("a")
		goblas.Dger(in.Int("m"), in.Int("n"), in.F64("alpha"), in.Arr("x"), in.Int("incx"),
			in.Arr("y"), in.Int("incy"), a, in.Int("lda"))
		w.arr("a", a)
	case "dtrsv":
		x := in.Arr("x")
		goblas.Dtrsv(uplo(in.Str("uplo")), trans(in.Str("transa")), diag(in.Str("diag")),
			in.Int("n"), in.Arr("a"), in.Int("lda"), x, in.Int("incx"))
		w.arr("x", x)
		if pad {
			w.arr("a", in.Arr("a"))
		}

	// ---------- Level 2, float32 ----------
	case "sgemv":
		y := to32(in.Arr("y"))
		goblas.Sgemv(trans(in.Str("transa")), in.Int("m"), in.Int("n"), float32(in.F64("alpha")),
			to32(in.Arr("a")), in.Int("lda"), to32(in.Arr("x")), in.Int("incx"), float32(in.F64("beta")), y, in.Int("incy"))
		w.arr32("y", y)
		if pad {
			w.arr("a", in.Arr("a"))
		}
	case "sger":
		a := to32(in.Arr("a"))
		goblas.Sger(in.Int("m"), in.Int("n"), float32(in.F64("alpha")), to32(in.Arr("x")), in.Int("incx"),
			to32(in.Arr("y")), in.Int("incy"), a, in.Int("lda"))
		w.arr32("a", a)
	case "strsv":
		x := to32(in.Arr("x"))
		goblas.Strsv(uplo(in.Str("uplo")), trans(in.Str("transa")), diag(in.Str("diag")),
			in.Int("n"), to32(in.Arr("a")), in.Int("lda"), x, in.Int("incx"))
		w.arr32("x", x)
		if pad {
			w.arr("a", in.Arr("a"))
		}

	// ---------- Level 3, float64 ----------
	case "dgemm":
		c := in.Arr("c")
		goblas.Dgemm(trans(in.Str("transa")), trans(in.Str("transb")), in.Int("m"), in.Int("n"), in.Int("k"),
			in.F64("alpha"), in.Arr("a"), in.Int("lda"), in.Arr("b"), in.Int("ldb"), in.F64("beta"), c, in.Int("ldc"))
		w.arr("c", c)
		if pad {
			w.arr("a", in.Arr("a"))
			w.arr("b", in.Arr("b"))
		}
	case "dsyrk":
		c := in.Arr("c")
		goblas.Dsyrk(uplo(in.Str("uplo")), trans(in.Str("trans")), in.Int("n"), in.Int("k"),
			in.F64("alpha"), in.Arr("a"), in.Int("lda"), in.F64("beta"), c, in.Int("ldc"))
		w.arr("c", c)
		if pad {
			w.arr("a", in.Arr("a"))
		}
	case "dsymm":
		c := in.Arr("c")
		goblas.Dsymm(side(in.Str("side")), uplo(in.Str("uplo")), in.Int("m"), in.Int("n"),
			in.F64("alpha"), in.Arr("a"), in.Int("lda"), in.Arr("b"), in.Int("ldb"), in.F64("beta"), c, in.Int("ldc"))
		w.arr("c", c)
		if pad {
			w.arr("a", in.Arr("a"))
		}
	case "dtrmm":
		b := in.Arr("b")
		goblas.Dtrmm(side(in.Str("side")), uplo(in.Str("uplo")), trans(in.Str("transa")), diag(in.Str("diag")),
			in.Int("m"), in.Int("n"), in.F64("alpha"), in.Arr("a"), in.Int("lda"), b, in.Int("ldb"))
		w.arr("b", b)
		if pad {
			w.arr("a", in.Arr("a"))
		}
	case "dtrsm":
		b := in.Arr("b")
		goblas.Dtrsm(side(in.Str("side")), uplo(in.Str("uplo")), trans(in.Str("transa")), diag(in.Str("diag")),
			in.Int("m"), in.Int("n"), in.F64("alpha"), in.Arr("a"), in.Int("lda"), b, in.Int("ldb"))
		w.arr("b", b)
		if pad {
			w.arr("a", in.Arr("a"))
		}

	// ---------- Level 3, float32 ----------
	case "sgemm":
		c := to32(in.Arr("c"))
		goblas.Sgemm(trans(in.Str("transa")), trans(in.Str("transb")), in.Int("m"), in.Int("n"), in.Int("k"),
			float32(in.F64("alpha")), to32(in.Arr("a")), in.Int("lda"), to32(in.Arr("b")), in.Int("ldb"), float32(in.F64("beta")), c, in.Int("ldc"))
		w.arr32("c", c)
		if pad {
			w.arr("a", in.Arr("a"))
			w.arr("b", in.Arr("b"))
		}
	case "ssyrk":
		c := to32(in.Arr("c"))
		goblas.Ssyrk(uplo(in.Str("uplo")), trans(in.Str("trans")), in.Int("n"), in.Int("k"),
			float32(in.F64("alpha")), to32(in.Arr("a")), in.Int("lda"), float32(in.F64("beta")), c, in.Int("ldc"))
		w.arr32("c", c)
		if pad {
			w.arr("a", in.Arr("a"))
		}
	case "ssymm":
		c := to32(in.Arr("c"))
		goblas.Ssymm(side(in.Str("side")), uplo(in.Str("uplo")), in.Int("m"), in.Int("n"),
			float32(in.F64("alpha")), to32(in.Arr("a")), in.Int("lda"), to32(in.Arr("b")), in.Int("ldb"), float32(in.F64("beta")), c, in.Int("ldc"))
		w.arr32("c", c)
		if pad {
			w.arr("a", in.Arr("a"))
		}
	case "strmm":
		b := to32(in.Arr("b"))
		goblas.Strmm(side(in.Str("side")), uplo(in.Str("uplo")), trans(in.Str("transa")), diag(in.Str("diag")),
			in.Int("m"), in.Int("n"), float32(in.F64("alpha")), to32(in.Arr("a")), in.Int("lda"), b, in.Int("ldb"))
		w.arr32("b", b)
		if pad {
			w.arr("a", in.Arr("a"))
		}
	case "strsm":
		b := to32(in.Arr("b"))
		goblas.Strsm(side(in.Str("side")), uplo(in.Str("uplo")), trans(in.Str("transa")), diag(in.Str("diag")),
			in.Int("m"), in.Int("n"), float32(in.F64("alpha")), to32(in.Arr("a")), in.Int("lda"), b, in.Int("ldb"))
		w.arr32("b", b)
		if pad {
			w.arr("a", in.Arr("a"))
		}

	default:
		return fmt.Errorf("unknown op %q", op)
	}
	return nil
}
