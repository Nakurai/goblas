// Command goblast-verify is the middle step of the goblast correctness
// pipeline: it reads each case's input.txt from a goblast corpus, runs the
// operation through goblas, and writes output.txt in goblast's format. Run
// `goblast gen` before it and `goblast check` after it (see verify.sh).
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
)

func main() {
	corpus := flag.String("corpus", "./corpus", "goblast corpus directory (the one passed to `goblast gen`)")
	flag.Parse()

	casesDir := filepath.Join(*corpus, "cases")
	entries, err := os.ReadDir(casesDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "goblast-verify: cannot read %s: %v\n", casesDir, err)
		fmt.Fprintln(os.Stderr, "did you run `goblast gen` first?")
		os.Exit(1)
	}

	var done, failed int
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		caseDir := filepath.Join(casesDir, e.Name())
		if err := runCase(caseDir); err != nil {
			fmt.Fprintf(os.Stderr, "goblast-verify: %s: %v\n", e.Name(), err)
			failed++
			continue
		}
		done++
	}

	fmt.Printf("goblast-verify: wrote output.txt for %d cases", done)
	if failed > 0 {
		fmt.Printf(", %d failed", failed)
	}
	fmt.Println()
	if failed > 0 {
		os.Exit(1)
	}
}

// runCase parses one input.txt, runs goblas, and writes output.txt. A panic
// from a malformed field is recovered and returned as an error so one bad
// case never aborts the whole run.
func runCase(caseDir string) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("panic: %v", r)
		}
	}()

	in, err := ParseFile(filepath.Join(caseDir, "input.txt"))
	if err != nil {
		return err
	}

	var w writer
	if err := dispatch(in, &w); err != nil {
		return err
	}
	return w.writeFile(filepath.Join(caseDir, "output.txt"))
}
