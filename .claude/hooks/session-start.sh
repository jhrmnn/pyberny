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
# [doc] is needed for the sphinx-build step in scripts/check.sh. The
# Sphinx step relies on `node` for sphinxcontrib-katex's server-side
# prerender; the Claude Code on the web base image ships node, so no
# extra install is required here.
pip install -e ".[test,doc]"

pip install ruff black

# xtb provides berny.solvers.XTBSolver. The Claude Code on the web image's
# Python has a prebuilt xtb wheel, so pip works here (unlike CI's Python 3.12,
# which has no wheel and provisions xtb from conda-forge). typing_extensions is
# imported by the bindings but not always pulled in. Best-effort: never fail
# session startup if a wheel is unavailable -- XTBSolver tests just skip then.
pip install xtb typing_extensions ||
  echo "warning: xtb wheel unavailable; XTBSolver tests will skip" >&2

