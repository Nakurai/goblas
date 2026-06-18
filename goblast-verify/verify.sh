#!/bin/sh
# Run the full goblast correctness pipeline against goblas:
#   1. goblast gen   -> generate the deterministic case corpus
#   2. goblast-verify -> run every case through goblas, write output.txt
#   3. goblast check -> grade output.txt against the oracle, print the report
#
# Usage: ./verify.sh [corpus-dir]   (default: ./corpus)
# Requires `goblast` on PATH (go install github.com/nakurai/goblast/cmd/goblast@latest).
set -e

CORPUS="${1:-./corpus}"
cd "$(dirname "$0")"

echo "==> goblast gen $CORPUS"
goblast gen "$CORPUS"

echo "==> running goblas over the corpus"
go run . -corpus "$CORPUS"

echo "==> goblast check $CORPUS"
goblast check "$CORPUS"
