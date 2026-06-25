#!/bin/bash
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"

# The web sandbox routes every github.com URL through a repo-scoped git relay
# (a catch-all `url.<relay>.insteadOf = https://github.com/`). That relay 403s
# every repo except this one, which breaks the `molsym` git dependency pulled in
# by the pip install below. Narrow the rewrite to just `origin` so all other
# github.com URLs fall through to the egress proxy (which permits them). The
# `case` guard means we only act on a real relay URL and never write a malformed
# empty-section entry when `git remote get-url origin` returns nothing.
origin_url=$(git remote get-url origin 2>/dev/null || true)
case "$origin_url" in
  *//*/git/*)
    repo_path=${origin_url##*/git/}
    git config --global --unset-all url."${origin_url%"$repo_path"}".insteadOf 2>/dev/null || true
    git config --global url."$origin_url".insteadOf "https://github.com/$repo_path"
    ;;
esac

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
