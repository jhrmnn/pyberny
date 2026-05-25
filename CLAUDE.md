# Project conventions

## Pre-push checks

Before `git push`, run `scripts/check.sh`. It runs the same lint, test, and
doc-build commands CI runs (`ruff check`, `black --check`, `pytest`, and
`sphinx-build -W -E doc doc/_check`) against the current Python, so most CI
failures are catchable locally without waiting for the 5-version matrix. The
Sphinx step uses server-side KaTeX prerender, so `node` and the `[doc]` extra
(`pip install -e ".[doc]"`) must be installed.

## CI results

After pushing a PR, subscribe to it with `subscribe_pr_activity` and wait for
CI results before closing the task. When CI events arrive, investigate failures
and push fixes. Only mark CI checklist items as done once CI actually passes —
do not leave them unchecked at end of turn.

Do not chase inconsequential, non-deterministic CI slips. A sub-percent
`codecov/project` dip from a single flaky line flipping hit/miss — especially
when the diff adds no measured `berny` source and `codecov/patch` is clean — is
noise, not a regression. Likewise, MOPAC PM7 benchmark step counts are not
bitwise-reproducible across runners (see
`src/berny/benchmarks/birkholz_schlegel/SOURCE.md`), so an occasional `birkholz mopac`
batch drifting past tolerance is flaky. Report these and move on; do not add
coverage-gate config, loosen tolerances, or rewrite reference values to make
them green.

## CHANGELOG

`CHANGELOG.md` is reserved for important user-facing changes — new features,
breaking changes, and bug fixes whose effect a user is likely to notice.

Do not add entries for internal refactors, lint/test/CI changes,
documentation-only fixes, or small bugs whose impact is invisible in normal
use. The commit message and PR description are the durable record for those.

When in doubt, leave it out: an over-long changelog buries the changes that
actually matter to users.
