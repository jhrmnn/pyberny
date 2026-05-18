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

# `flake8` on PATH comes from a uv-managed tool environment that is
# isolated from system site-packages, so `pip install flake8-bugbear ...`
# would silently no-op (the plugins go to system Python, the uv-isolated
# flake8 can't see them, and pre-push `flake8` misses B-codes that CI's
# fresh `pip install flake8 ...` would catch). Install plugins into the
# same tool env via `uv tool install --with`.
if command -v uv >/dev/null; then
  uv tool install --quiet --with flake8-bugbear --with flake8-comprehensions \
    --with pep8-naming flake8
else
  pip install flake8 flake8-bugbear flake8-comprehensions pep8-naming
fi
pip install black isort pydocstyle
