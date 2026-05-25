#!/bin/bash
# Run the same checks CI runs (lint + tests + doc) against the current Python.
# Mirrors .github/workflows/lint.yaml, tests.yaml, and doc.yaml.
set -uo pipefail

cd "$(dirname "$0")/.."

fail=0
run() {
    local name=$1
    shift
    echo "=== $name ==="
    if ! "$@"; then
        fail=1
        echo "--- $name FAILED ---"
    fi
}

run ruff   ruff check .
run black  black . --check
run mypy   python -m mypy
run pytest python -m pytest -q
run sphinx sphinx-build -W -E doc doc/_check

if [ "$fail" -ne 0 ]; then
    echo "One or more checks failed."
    exit 1
fi
echo "All checks passed."
