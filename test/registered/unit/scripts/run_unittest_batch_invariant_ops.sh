#!/usr/bin/env bash
# Run Python unittest discovery under test/registered/unit/batch_invariant_ops/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
run_unittest_in_subdir batch_invariant_ops "$@"
