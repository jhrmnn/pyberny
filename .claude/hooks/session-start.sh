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

# Standalone investigation reports live on the `reports` orphan branch (one
# YYYY-MM-DD-<slug>/ folder each, README.md as the main content), kept off the
# package history on master. Check that branch out as a git worktree at
# ./reports so past reports are available locally and new ones can be added
# without mixing into the package tree; .gitignore keeps ./reports out of the
# main working tree. See CLAUDE.md ("Investigation reports"). Every git call is
# guarded so a missing branch or offline remote never aborts session start.
reports_dir="$CLAUDE_PROJECT_DIR/reports"
if [ ! -e "$reports_dir" ]; then
  git fetch origin reports || true
  if git show-ref --verify --quiet refs/heads/reports; then
    git worktree add "$reports_dir" reports || true
  elif git show-ref --verify --quiet refs/remotes/origin/reports; then
    git worktree add --track -b reports "$reports_dir" origin/reports || true
  fi
fi
