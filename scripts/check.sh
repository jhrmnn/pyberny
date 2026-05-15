#!/bin/bash
# Run the same checks CI runs (lint + tests) against the current Python.
# Mirrors .github/workflows/lint.yaml and .github/workflows/tests.yaml.
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

run flake8     flake8
run black      black . --check
run isort      isort . --check
run pydocstyle pydocstyle src
run pytest     python -m pytest -q

if [ "$fail" -ne 0 ]; then
    echo "One or more checks failed."
    exit 1
fi
echo "All checks passed."
