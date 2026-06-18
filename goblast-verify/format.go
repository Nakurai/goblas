package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// Doc is a parsed goblast case file: scalar fields keyed by name and array
// fields keyed by name. It implements the grammar documented in goblast's
// FORMAT.md §1–§2 (we re-implement it here because goblast's own
// internal/format package is not importable from another module).
type Doc struct {
	scalars map[string]string
	arrays  map[string][]float64
}

// ParseFile reads a case file (input.txt) and returns its fields.
func ParseFile(path string) (*Doc, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	d := &Doc{scalars: map[string]string{}, arrays: map[string][]float64{}}
	seen := map[string]bool{}

	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	line := 0
	for sc.Scan() {
		line++
		text := sc.Text()
		if text == "" || strings.HasPrefix(text, "#") {
			continue // blank or comment
		}

		if strings.HasPrefix(text, "@") {
			// Array header: "@name count" then exactly count float lines.
			parts := strings.Fields(text)
			if len(parts) != 2 {
				return nil, fmt.Errorf("%s:%d: malformed array header %q", path, line, text)
			}
			name := strings.TrimPrefix(parts[0], "@")
			count, err := strconv.Atoi(parts[1])
			if err != nil || count < 0 {
				return nil, fmt.Errorf("%s:%d: bad array count %q", path, line, parts[1])
			}
			if seen[name] {
				return nil, fmt.Errorf("%s:%d: duplicate field %q", path, line, name)
			}
			seen[name] = true

			vals := make([]float64, 0, count)
			for i := 0; i < count; i++ {
				if !sc.Scan() {
					return nil, fmt.Errorf("%s: array %q ran out of values (wanted %d, got %d)", path, name, count, i)
				}
				line++
				v, err := strconv.ParseFloat(strings.TrimSpace(sc.Text()), 64)
				if err != nil {
					return nil, fmt.Errorf("%s:%d: array %q value %q not a float", path, line, name, sc.Text())
				}
				vals = append(vals, v)
			}
			d.arrays[name] = vals
			continue
		}

		// Scalar field: "name value" (value = rest of line after first space).
		sp := strings.IndexByte(text, ' ')
		if sp < 0 {
			return nil, fmt.Errorf("%s:%d: not a scalar, comment, or array header: %q", path, line, text)
		}
		name := text[:sp]
		if seen[name] {
			return nil, fmt.Errorf("%s:%d: duplicate field %q", path, line, name)
		}
		seen[name] = true
		d.scalars[name] = text[sp+1:]
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return d, nil
}

func (d *Doc) Str(name string) string {
	return d.scalars[name]
}

func (d *Doc) Int(name string) int {
	v, err := strconv.Atoi(strings.TrimSpace(d.scalars[name]))
	if err != nil {
		panic(fmt.Sprintf("field %q is not an integer: %q", name, d.scalars[name]))
	}
	return v
}

func (d *Doc) F64(name string) float64 {
	v, err := strconv.ParseFloat(strings.TrimSpace(d.scalars[name]), 64)
	if err != nil {
		panic(fmt.Sprintf("field %q is not a float: %q", name, d.scalars[name]))
	}
	return v
}

func (d *Doc) Arr(name string) []float64 {
	return d.arrays[name]
}

func (d *Doc) hasTag(tag string) bool {
	for _, t := range strings.Split(d.scalars["tags"], ",") {
		if t == tag {
			return true
		}
	}
	return false
}

// writer accumulates output fields and flushes them to output.txt.
type writer struct {
	b strings.Builder
}

func fmtFloat(v float64) string {
	return strconv.FormatFloat(v, 'g', -1, 64)
}

func (w *writer) float(name string, v float64) {
	fmt.Fprintf(&w.b, "%s %s\n", name, fmtFloat(v))
}

func (w *writer) int(name string, v int) {
	fmt.Fprintf(&w.b, "%s %d\n", name, v)
}

func (w *writer) arr(name string, vals []float64) {
	fmt.Fprintf(&w.b, "@%s %d\n", name, len(vals))
	for _, v := range vals {
		w.b.WriteString(fmtFloat(v))
		w.b.WriteByte('\n')
	}
}

// arr32 widens a float32 buffer to float64 (lossless) and writes it; the file
// always carries float64-width decimal text, even for S-prefixed routines.
func (w *writer) arr32(name string, vals []float32) {
	fmt.Fprintf(&w.b, "@%s %d\n", name, len(vals))
	for _, v := range vals {
		w.b.WriteString(fmtFloat(float64(v)))
		w.b.WriteByte('\n')
	}
}

func (w *writer) writeFile(path string) error {
	return os.WriteFile(path, []byte(w.b.String()), 0o644)
}
