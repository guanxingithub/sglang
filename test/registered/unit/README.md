# Unit Tests

Component-level tests that do **not** launch a server or load model weights.
Tests can use CPU or GPU — the key criterion is **no server process**.

## Quick Start

1. Find the source file under `python/sglang/srt/`.
2. Create the corresponding test here, mirroring the source tree:
   ```
   srt/mem_cache/radix_cache.py       →  unit/mem_cache/test_radix_cache.py
   srt/sampling/sampling_params.py    →  unit/sampling/test_sampling_params.py
   ```
3. Register for CI at the **top of the file** (after imports, before test classes):
   ```python
   from sglang.test.ci.ci_register import register_cpu_ci
   register_cpu_ci(est_time=5, suite="stage-a-test-cpu")
   # or: register_cuda_ci(est_time=10, suite="stage-b-test-1-gpu-small")
   ```
4. Run locally:
   ```bash
   pytest test/registered/unit/ -v            # all unit tests
   pytest test/registered/unit/mem_cache/ -v  # one module
   ```
5. Run with coverage:
   ```bash
   # summary
   pytest test/registered/unit/ --cov --cov-config=.coveragerc -v

   # PR incremental check (require ≥60% on changed lines)
   pytest test/registered/unit/ --cov --cov-config=.coveragerc --cov-report=xml
   diff-cover coverage.xml --compare-branch=origin/main --fail-under=60
   ```

## Examples

### Basic unit test

```python
"""Unit tests for <module> — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

import unittest

from sglang.srt.<module> import TargetClass
from sglang.test.test_utils import CustomTestCase


class TestTargetClass(CustomTestCase):
    def test_basic_behavior(self):
        obj = TargetClass(...)
        self.assertEqual(obj.method(), expected)


if __name__ == "__main__":
    unittest.main()
```

### Stubbing GPU-only imports for CPU tests

Some modules (e.g. `scheduler.py`, `io_struct.py`) transitively import packages like
`sgl_kernel` that require a GPU to initialize. To run pure-mock tests against these
modules on CPU-only CI, stub the problematic package **before** importing it.

`maybe_stub_sgl_kernel()` in `test_utils.py` does this for `sgl_kernel`: it's a no-op
on GPU machines, and on CPU it installs a `sys.meta_path` finder that auto-creates empty
stub modules for all `sgl_kernel.*` submodules.

```python
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

from sglang.srt.managers.io_struct import FlushCacheReqInput
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")
```

The same pattern (`sys.meta_path` finder) can be applied to other GPU-only packages.
See `maybe_stub_sgl_kernel()` in `python/sglang/test/test_utils.py` for the
implementation. Do not directly mutate `sys.modules` at module level — pytest
imports all test files before running any, so such mutations pollute the entire
process. If you must stub, use `patch.dict("sys.modules", ...)` with proper cleanup.

## Rules

- **No** `popen_launch_server()` or `Engine(...)`.
- **No** model weight loading.
- Use `CustomTestCase` (from `sglang.test.test_utils`, adds CI retry).
- Use `unittest.mock` for dependencies that are expensive to construct.

---

## Auto-generated mirror smoke tests (SRT and `sgl_kernel`)

These are **machine-generated** `unittest` files that register with CPU CI (`register_cpu_ci`). They are **smoke** checks, not full behavioral coverage:

1. **Syntax** — the mirrored source file parses (`ast.parse`).
2. **Symbols (when import works)** — top-level `def` / `async def` and direct class methods found in the AST exist on the imported module (nested inner functions are ignored).

If a module fails to import (optional deps, GPU-only init, etc.), the symbol test **skips**. Add hand-written tests next to the feature for real behavior.

### SRT mirror tests (`test/registered/unit/srt/`)

Mirrors every `*.py` under `python/sglang/srt/` into the same relative path under `test/registered/unit/srt/` as `test_<name>.py` (e.g. `utils/common.py` → `srt/utils/test_common.py`; `__init__.py` → `test___init__.py`).

| Item | Location |
|------|----------|
| Generator | `scripts/generate_srt_mirror_unit_tests.py` |
| AST / import helpers | `python/sglang/test/srt_mirror_smoke.py` |

**Regenerate everything** (and optionally delete stale mirror files):

```bash
# from the sglang repo root
python3 scripts/generate_srt_mirror_unit_tests.py
python3 scripts/generate_srt_mirror_unit_tests.py --clean
```

**Only one subtree** of `python/sglang/srt` (faster; `--clean` only prunes that subtree’s mirror output):

```bash
python3 scripts/generate_srt_mirror_unit_tests.py --subdir layers
python3 scripts/generate_srt_mirror_unit_tests.py --subdir lora --clean
```

**Run all mirror tests under a subtree** (e.g. `lora`) with `unittest` discover:

```bash
cd python
PYTHONPATH=. python3 -m unittest discover \
  -s ../test/registered/unit/srt/lora \
  -p 'test_*.py' \
  -v
```

**Helper script** (from repo root; runs **every** `test_*.py` under the subtree recursively):

```bash
./scripts/run_srt_unittest_subdir.sh lora
./scripts/run_srt_unittest_subdir.sh layers -v -f
```

The script sets `PYTHONPATH` to `python/` and executes each mirror file (same model as CI: one Python process per file). Plain `unittest discover -s …` only descends into directories that contain `__init__.py`, so it would miss nested mirror files unless those inits exist—prefer this script for subtrees like `layers/`.

### `sgl_kernel` mirror tests (`test/registered/unit/sgl_kernel/`)

Mirrors every `*.py` under `sgl-kernel/python/sgl_kernel/` into `test/registered/unit/sgl_kernel/` (monorepo layout: repo root contains both `sgl-kernel/` and `test/`).

| Item | Location |
|------|----------|
| Generator | `scripts/generate_sgl_kernel_mirror_unit_tests.py` |
| AST / import helpers | `python/sglang/test/sgl_kernel_mirror_smoke.py` |

**Regenerate:**

```bash
python3 scripts/generate_sgl_kernel_mirror_unit_tests.py
python3 scripts/generate_sgl_kernel_mirror_unit_tests.py --clean
```

**Run** (needs both `sglang` and the `sgl_kernel` package on `PYTHONPATH`):

```bash
cd python
PYTHONPATH=.:../sgl-kernel/python python3 -m unittest discover \
  -s ../test/registered/unit/sgl_kernel \
  -p 'test_*.py' \
  -v
```

CI jobs that execute these files must include `sgl-kernel/python` on `PYTHONPATH` (or an installed `sgl_kernel` wheel) so `import sgl_kernel` succeeds where the environment allows.

### Relation to `sgl-kernel/tests/` (pytest)

The **`sgl-kernel/tests/`** tree (pytest) exercises the **`sgl_kernel`** Python API and the compiled kernels under `sgl-kernel/csrc/` (accuracy / performance vs PyTorch or reference code). It is **separate** from the registered **unittest** mirror suite above. Many tests `import sgl_kernel` or `from sgl_kernel import ...`; some also import `sglang.srt.*` for references or utilities. See `sgl-kernel/README.md` and that directory for how to run pytest there.

### CI registration

Each generated file calls:

```python
from sglang.test.ci.ci_register import register_cpu_ci
register_cpu_ci(est_time=1, suite="stage-a-test-cpu")
```

`test/run_suite.py` discovers all `*.py` under `test/registered/` (except `__init__.py` / `conftest.py`) and requires this registration. Large numbers of mirror files increase the **estimated** CPU time printed by the suite; adjust `est_time` in the generator templates if scheduling needs it.
