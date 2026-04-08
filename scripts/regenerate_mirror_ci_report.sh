#!/usr/bin/env bash
# Regenerate test/registered/unit/mirror_ci_test_index.csv and MIRROR_CI_TEST_REPORT.md
# after mirror CI tests or sources change. Does not require pip-installing sglang.
#
# Usage (from anywhere):
#   ./scripts/regenerate_mirror_ci_report.sh
#   ./scripts/regenerate_mirror_ci_report.sh --csv-only
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
exec python3 "${SCRIPT_DIR}/regenerate_mirror_ci_report.py" "$@"
