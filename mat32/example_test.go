package mat32_test

import (
	"fmt"

	"github.com/nakurai/goblas/mat32"
)

// Example_ridgeRegression fits ridge regression weights end-to-end in float32:
// w = (XᵀX + λI)⁻¹ Xᵀy, solved with the native float32 Cholesky (XᵀX + λI is
// symmetric positive definite). No float64 casting occurs on this path.
func Example_ridgeRegression() {
	// 4 samples, 2 features. Underlying model y ≈ 2*x0 + 3*x1.
	X := mat32.NewDense32(4, 2, []float32{
		1, 0,
		0, 1,
		1, 1,
		2, 1,
	})
	y := mat32.NewVecDense32(4, []float32{2, 3, 5, 7})

	// Normal-equation matrix A = XᵀX + λI and right-hand side Xᵀy.
	var xtx mat32.Dense32
	xtx.Mul(X.T(), X)
	const lambda float32 = 1e-3
	r, _ := xtx.Dims()
	for i := 0; i < r; i++ {
		xtx.Set(i, i, xtx.At(i, i)+lambda)
	}
	a := mat32.SymDense32FromDense(&xtx, true)

	var xty mat32.VecDense32
	xty.MulVec(X.T(), y)

	// Solve A w = Xᵀy with the native float32 Cholesky.
	var chol mat32.Cholesky32
	if !chol.Factorize(a) {
		fmt.Println("not positive definite")
		return
	}
	var w mat32.Dense32
	if err := chol.SolveTo(&w, &xty); err != nil {
		fmt.Println(err)
		return
	}

	fmt.Printf("w0=%.2f w1=%.2f\n", w.At(0, 0), w.At(1, 0))
	// Output: w0=2.00 w1=3.00
}
