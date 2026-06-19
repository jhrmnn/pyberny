#!/bin/bash
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"
# [doc] is needed for the sphinx-build step in scripts/check.sh. The Sphinx
# step relies on `node` for sphinxcontrib-katex's server-side prerender; the
# Claude Code on the web base image ships node, so no extra install is required
# here. [benchmark] pulls in tblite, the GFN-xTB backend for
# berny.solvers.XTBSolver, so its tests run rather than skip.
pip install -e ".[test,doc,benchmark]"

pip install ruff black
