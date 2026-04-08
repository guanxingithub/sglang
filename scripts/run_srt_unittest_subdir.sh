#!/usr/bin/env bash
# Run all unittest files under test/registered/unit/srt/<subdir>/ (mirror tests for
# python/sglang/srt/<subdir>/).
#
# Usage:
#   ./scripts/run_srt_unittest_subdir.sh lora
#   ./scripts/run_srt_unittest_subdir.sh layers -v -f
#
# Requires: repo layout with python/sglang and test/registered/unit/srt.
# Working directory does not matter.
#
# Note: ``python -m unittest discover`` only descends into directories that contain
# ``__init__.py``. The generated mirror tree usually has no package inits, so this
# script runs every ``test_*.py`` under <subdir> recursively (same as CI: one process
# per file).

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBDIR="${1:?Usage: $0 <subdir-under-srt> [unittest options passed to each file...]
  Examples:
    $0 lora
    $0 layers -v -f
}"
shift

TEST_DIR="$REPO_ROOT/test/registered/unit/srt/$SUBDIR"
if [[ ! -d "$TEST_DIR" ]]; then
  echo "error: test directory not found: $TEST_DIR" >&2
  exit 1
fi

cd "$REPO_ROOT/python"
export PYTHONPATH=.

mapfile -d '' -t FILES < <(find "$TEST_DIR" -name 'test_*.py' -print0 | sort -z)
if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "error: no test_*.py under $TEST_DIR" >&2
  exit 1
fi

FAIL=0
for f in "${FILES[@]}"; do
  echo "== $f ==" >&2
  if ! python3 "$f" "$@"; then
    FAIL=1
  fi
done
exit "$FAIL"
