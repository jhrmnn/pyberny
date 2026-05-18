# Project conventions

## Pre-push checks

Before `git push`, run `scripts/check.sh`. It runs the same lint and test
commands CI runs (`flake8`, `black --check`, `isort --check`, `pydocstyle src`,
`pytest`) against the current Python, so most CI failures are catchable
locally without waiting for the 5-version matrix.

## CI results

After pushing a PR, subscribe to it with `subscribe_pr_activity` and wait for
CI results before closing the task. When CI events arrive, investigate failures
and push fixes. Only mark CI checklist items as done once CI actually passes —
do not leave them unchecked at end of turn.

## CHANGELOG

`CHANGELOG.md` is reserved for important user-facing changes — new features,
breaking changes, and bug fixes whose effect a user is likely to notice.

Do not add entries for internal refactors, lint/test/CI changes,
documentation-only fixes, or small bugs whose impact is invisible in normal
use. The commit message and PR description are the durable record for those.

When in doubt, leave it out: an over-long changelog buries the changes that
actually matter to users.
