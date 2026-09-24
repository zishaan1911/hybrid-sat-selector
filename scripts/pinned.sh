#!/usr/bin/env bash
# Run `hsat` from a worktree pinned to the current commit, with data/ and experiments/
# still resolved in this checkout. Long experiments then keep running the code they
# started with while development continues, and the run registry records that commit.
set -euo pipefail
cd "$(dirname "$0")/.."
commit="$(git rev-parse --short HEAD)"
tree="../hsat-pinned-$commit"
[ -d "$tree" ] || git worktree add --detach "$tree" "$commit" >/dev/null
PYTHONPATH="$(cd "$tree" && pwd)/src" exec python -m hsat.cli.main "$@"
