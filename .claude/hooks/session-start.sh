#!/bin/bash
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

MOPAC_VERSION=23.2.5
MOPAC_DIR="$CLAUDE_PROJECT_DIR/.claude/mopac"

if [ ! -x "$MOPAC_DIR/bin/mopac" ]; then
  mkdir -p "$MOPAC_DIR"
  cd "$MOPAC_DIR"
  wget -q "https://github.com/openmopac/mopac/releases/download/v${MOPAC_VERSION}/mopac-${MOPAC_VERSION}-linux.tar.gz"
  tar --strip-components=1 -xzf "mopac-${MOPAC_VERSION}-linux.tar.gz"
  rm "mopac-${MOPAC_VERSION}-linux.tar.gz"
fi

echo "export PATH=\"$MOPAC_DIR/bin:\$PATH\"" >> "$CLAUDE_ENV_FILE"
echo "export LD_LIBRARY_PATH=\"$MOPAC_DIR/lib:\${LD_LIBRARY_PATH:-}\"" >> "$CLAUDE_ENV_FILE"

cd "$CLAUDE_PROJECT_DIR"
# [doc] is needed for the sphinx-build step in scripts/check.sh. The Sphinx
# step relies on `node` for sphinxcontrib-katex's server-side prerender; the
# Claude Code on the web base image ships node, so no extra install is required
# here. [benchmark] pulls in tblite, the GFN-xTB backend for
# berny.solvers.XTBSolver, so its tests run rather than skip.
pip install -e ".[test,doc,benchmark]"

pip install ruff black
