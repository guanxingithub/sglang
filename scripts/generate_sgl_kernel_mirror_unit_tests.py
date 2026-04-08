#!/usr/bin/env python3
# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0.

"""Generate ``test/registered/unit/sgl_kernel/**/test_*.py`` mirroring ``sgl-kernel/python/sgl_kernel``.

Requires a monorepo layout with ``sgl-kernel/python/sgl_kernel`` and ``test/registered``.

Run::

    python scripts/generate_sgl_kernel_mirror_unit_tests.py
    python scripts/generate_sgl_kernel_mirror_unit_tests.py --clean

``--clean`` removes generated ``test_*.py`` under ``test/registered/unit/sgl_kernel`` with no source.

Import tests need ``sgl_kernel`` on ``PYTHONPATH`` (e.g. ``sgl-kernel/python``) when running unittest.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _test_filename_for_source(source: Path) -> str:
    return f"test_{source.stem}.py"


TEST_TEMPLATE = '''# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0.

"""Auto-generated smoke tests for ``sgl-kernel/python/sgl_kernel/{rel_posix}``."""

from __future__ import annotations

import importlib
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.sgl_kernel_mirror_smoke import (
    assert_symbol_resolves,
    module_name_from_sgl_kernel_relative_path,
    parse_source_file,
)

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")

RELATIVE_PKG_PATH = Path({rel_repr})


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "sgl-kernel" / "python" / "sgl_kernel").is_dir():
            return parent
    raise RuntimeError(
        "Could not locate repo root (directory containing sgl-kernel/python/sgl_kernel)."
    )


def _source_path() -> Path:
    return _repo_root() / "sgl-kernel" / "python" / "sgl_kernel" / RELATIVE_PKG_PATH


MODULE_NAME = module_name_from_sgl_kernel_relative_path(RELATIVE_PKG_PATH)


class TestMirrorSmoke(unittest.TestCase):
    def test_source_parses(self) -> None:
        parse_source_file(_source_path())

    def test_symbols_when_import_succeeds(self) -> None:
        _, symbols = parse_source_file(_source_path())
        if not symbols:
            return
        try:
            mod = importlib.import_module(MODULE_NAME)
        except BaseException as exc:  # noqa: BLE001
            self.skipTest(
                "optional import failed ("
                f"{{type(exc).__name__}}: {{exc}}); "
                "install or PYTHONPATH sgl-kernel/python if this module should import."
            )
        for sym in symbols:
            with self.subTest(symbol=sym.name, class_name=sym.class_name):
                try:
                    assert_symbol_resolves(MODULE_NAME, mod, sym)
                except AssertionError:
                    raise
                except BaseException as exc:  # noqa: BLE001
                    self.fail(
                        f"{{MODULE_NAME}}: error resolving {{sym}}: "
                        f"{{type(exc).__name__}}: {{exc}}"
                    )


if __name__ == "__main__":
    unittest.main()
'''


def generate(*, clean: bool) -> None:
    root = _repo_root()
    pkg_root = root / "sgl-kernel" / "python" / "sgl_kernel"
    out_root = root / "test" / "registered" / "unit" / "sgl_kernel"

    if not pkg_root.is_dir():
        raise SystemExit(f"missing sgl_kernel package tree: {pkg_root}")

    sources = sorted(
        p
        for p in pkg_root.rglob("*.py")
        if "__pycache__" not in p.parts
    )
    expected_outputs: set[Path] = set()

    for src in sources:
        rel = src.relative_to(pkg_root)
        rel_posix = rel.as_posix()
        out_dir = out_root / rel.parent
        out_path = out_dir / _test_filename_for_source(src)
        expected_outputs.add(out_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        rel_repr = repr(rel_posix)
        content = TEST_TEMPLATE.format(rel_posix=rel_posix, rel_repr=rel_repr)
        out_path.write_text(content, encoding="utf-8")

    if clean and out_root.is_dir():
        for path in sorted(out_root.rglob("test_*.py")):
            if path not in expected_outputs:
                path.unlink()
        for d in sorted({p for p in out_root.rglob("*") if p.is_dir()}, reverse=True):
            try:
                d.rmdir()
            except OSError:
                pass

    print(f"wrote {len(expected_outputs)} test file(s) under {out_root}", flush=True)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="delete stale test_*.py with no matching sgl_kernel source",
    )
    args = parser.parse_args(argv)
    generate(clean=args.clean)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
