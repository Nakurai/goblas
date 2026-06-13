# Gonum Fundamentals

Before diving into complex machine learning algorithms, it is helpful to understand the basics of `gonum`. This tutorial will walk you through reading a dataset, creating a matrix, normalizing the data, and plotting the results.

## Loading a Dataset

In Go, the standard library makes it easy to read a CSV file. Let's load the synthetic `housing.csv` dataset from `data/housing.csv`.

```go
package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"
)

func main() {
	// Open the CSV file
	file, err := os.Open("data/housing.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	// Read the records
	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	// Skip the header and parse the data into float64 slices
	var data []float64
	rows := len(records) - 1
	cols := len(records[0])

	for i, row := range records {
		if i == 0 {
			continue // skip header
		}
		for _, valStr := range row {
			val, err := strconv.ParseFloat(valStr, 64)
			if err != nil {
				log.Fatal(err)
			}
			data = append(data, val)
		}
	}
	fmt.Printf("Loaded %d rows and %d columns.\n", rows, cols)
}
```

## Creating a Matrix

With our data loaded into a flat slice (`[]float64`), we can easily create a dense matrix using `mat.NewDense`.

```go
import "gonum.org/v1/gonum/mat"

// Create a Dense matrix with the given dimensions and flat data slice
X := mat.NewDense(rows, cols, data)

// We can inspect dimensions using Dims()
r, c := X.Dims()
fmt.Printf("Matrix X has dimensions %d x %d\n", r, c)

// We can slice out the first two columns (features) into a new matrix
features := X.Slice(0, rows, 0, 2).(*mat.Dense)
```

## Normalizing the Dataset Efficiently

Machine learning models often perform much better when the features are on a similar scale (e.g., house square footage is in the thousands, but age is in the tens). A standard approach is **Z-score normalization**: subtract the mean from each column, and divide by the standard deviation.

Gonum makes it easy to compute the mean and standard deviation along columns using the `stat` and `floats` packages.

```go
import (
	"math"
	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/gonum/floats"
)

func normalizeColumns(m *mat.Dense) {
	r, c := m.Dims()
	colData := make([]float64, r)

	for j := 0; j < c; j++ {
		// Extract the column into a slice
		mat.Col(colData, j, m)

		// Calculate mean and standard deviation
		mean := stat.Mean(colData, nil)
		std := stat.StdDev(colData, nil)

		if std == 0 {
			std = 1 // Prevent division by zero
		}

		// Normalize the column in place: (x - mean) / std
		for i := 0; i < r; i++ {
			val := m.At(i, j)
			m.Set(i, j, (val-mean)/std)
		}
	}
}

// Normalize the features of our dataset
normalizeColumns(features)
```

## Plotting the Data

Visualization is a great way to understand what your data looks like. We can use the `gonum.org/v1/plot` library to plot points. Let's create a scatter plot of house size (square footage) versus price.

```go
import (
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

// Create a new plot
p := plot.New()
p.Title.Text = "House Price vs Size"
p.X.Label.Text = "Square Footage (Normalized)"
p.Y.Label.Text = "Price"

// Create a plotter.XYs slice to hold the points
pts := make(plotter.XYs, rows)
for i := 0; i < rows; i++ {
    // X is the size feature (column 0), Y is the price target (column 2)
	pts[i].X = X.At(i, 0)
	pts[i].Y = X.At(i, 2)
}

// Create a scatter plot and add it to the plot
s, err := plotter.NewScatter(pts)
if err != nil {
	log.Fatal(err)
}
p.Add(s)

// Save the plot to a PNG file
if err := p.Save(4*vg.Inch, 4*vg.Inch, "housing_plot.png"); err != nil {
	log.Fatal(err)
}
```

This fundamental workflow—load data, convert to a Gonum matrix, normalize features, and visualize—will serve you well throughout the other tutorials.

Next: [linear-regression.md](linear-regression.md) starts applying this knowledge to predict continuous values.
