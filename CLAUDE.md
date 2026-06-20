# Project conventions

## Pre-push checks

Before `git push`, run `scripts/check.sh`. It runs the same lint, test, and
doc-build commands CI runs (`ruff check`, `black --check`, `mypy`, `pytest`,
and `sphinx-build -W -E doc doc/_check`) against the current Python, so most
CI failures are catchable locally without waiting for the 5-version matrix.
The Sphinx step uses server-side KaTeX prerender, so `node` and the `[doc]`
extra (`pip install -e ".[doc]"`) must be installed.

## CI results

After pushing a PR, subscribe to it with `subscribe_pr_activity` and wait for
CI results before closing the task. When CI events arrive, investigate failures
and push fixes. Only mark CI checklist items as done once CI actually passes —
do not leave them unchecked at end of turn.

Do not chase inconsequential, non-deterministic CI slips. A sub-percent
`codecov/project` dip from a single flaky line flipping hit/miss — especially
when the diff adds no measured `berny` source and `codecov/patch` is clean — is
noise, not a regression. Likewise, GFN2-xTB benchmark step counts are not
bitwise-reproducible across runners (see
`src/berny/benchmarks/birkholz_schlegel/SOURCE.md`), so an occasional `birkholz xtb`
batch drifting past tolerance is flaky. Report these and move on; do not add
coverage-gate config, loosen tolerances, or rewrite reference values to make
them green.

## Investigation reports

Longer-form investigations — benchmark studies, root-cause write-ups, anything
that produces a standalone report with prose, figures, and throwaway analysis
scripts — do **not** live on `master`. They go on the `reports` orphan branch,
which shares no history with `master` and holds nothing but the reports
themselves, so the package tree stays free of one-off scratch material.

Layout on `reports`: one top-level folder per report, named
`YYYY-MM-DD-<slug>`, with the main content as `README.md` and all supporting
files (figures, data, scripts) alongside it in that same folder. Keep each
report self-contained in its folder.

The session-start hook checks the `reports` branch out as a git worktree at
`./reports` (ignored via `.gitignore`), so past reports are available locally
in every session. To add one, create a new dated folder under `./reports`,
commit it on the `reports` branch from that worktree, push, and open a PR
**against `reports`** (not `master`), and add the `report` label to it. PRs that
only add a report should target `reports` and carry the `report` label; code
changes target `master` as usual.

## CHANGELOG

`CHANGELOG.md` is reserved for important user-facing changes — new features,
breaking changes, and bug fixes whose effect a user is likely to notice.

Do not add entries for internal refactors, lint/test/CI changes,
documentation-only fixes, or small bugs whose impact is invisible in normal
use. The commit message and PR description are the durable record for those.

When in doubt, leave it out: an over-long changelog buries the changes that
actually matter to users.
