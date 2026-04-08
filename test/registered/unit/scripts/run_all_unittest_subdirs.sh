#!/usr/bin/env bash
# Run unittest discovery for every top-level subdirectory of test/registered/unit/
# that contains at least one file importing unittest (see individual run_unittest_*.sh).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Keep in sync with directories that contain unittest-based tests (see grep for import unittest).
for name in \
  batch_invariant_ops \
  constrained \
  entrypoints \
  function_call \
  layers \
  managers \
  mem_cache \
  model_loader \
  models \
  observability \
  parser \
  sampling \
  server_args \
  spec \
  srt \
  tokenizer \
  utils \
  ; do
  echo "========== unittest: ${name} =========="
  "${SCRIPT_DIR}/run_unittest_${name}.sh"
done
