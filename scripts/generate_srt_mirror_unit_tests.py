#!/usr/bin/env python3
# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0.

"""Generate ``test/registered/unit/srt/**/test_*.py`` mirroring ``python/sglang/srt``.

Run from any working directory::

    python scripts/generate_srt_mirror_unit_tests.py

Or::

    python scripts/generate_srt_mirror_unit_tests.py --clean

``--clean`` removes generated ``test_*.py`` files under ``test/registered/unit/srt``
that no longer have a matching source file.

Only a subtree of ``python/sglang/srt`` (e.g. ``layers``)::

    python scripts/generate_srt_mirror_unit_tests.py --subdir layers
    python scripts/generate_srt_mirror_unit_tests.py --subdir layers --clean
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _test_filename_for_source(source: Path) -> str:
    stem = source.stem
    return f"test_{stem}.py"


TEST_TEMPLATE = '''# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0.

"""Auto-generated smoke tests for ``python/sglang/srt/{rel_posix}``."""

from __future__ import annotations

import importlib
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.srt_mirror_smoke import (
    assert_symbol_resolves,
    module_name_from_srt_relative_path,
    parse_source_file,
)

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")

RELATIVE_SRT_PATH = Path({rel_repr})


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "python" / "sglang" / "srt").is_dir():
            return parent
    raise RuntimeError(
        "Could not locate repo root (directory containing python/sglang/srt)."
    )


def _source_path() -> Path:
    return _repo_root() / "python" / "sglang" / "srt" / RELATIVE_SRT_PATH


MODULE_NAME = module_name_from_srt_relative_path(RELATIVE_SRT_PATH)


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
                "add behavioral tests if this module should always import."
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


def generate(*, clean: bool, subdir: str | None) -> None:
    root = _repo_root()
    srt_root = root / "python" / "sglang" / "srt"
    out_root = root / "test" / "registered" / "unit" / "srt"

    if not srt_root.is_dir():
        raise SystemExit(f"missing srt tree: {srt_root}")

    if subdir is not None:
        source_root = srt_root / subdir
        if not source_root.is_dir():
            raise SystemExit(f"missing srt subtree: {source_root}")
        scan_roots = [source_root]
    else:
        scan_roots = [srt_root]

    sources = sorted(
        p
        for scan in scan_roots
        for p in scan.rglob("*.py")
        if "__pycache__" not in p.parts
    )
    expected_outputs: set[Path] = set()

    for src in sources:
        rel = src.relative_to(srt_root)
        rel_posix = rel.as_posix()
        out_dir = out_root / rel.parent
        out_path = out_dir / _test_filename_for_source(src)
        expected_outputs.add(out_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        rel_repr = repr(rel_posix)
        content = TEST_TEMPLATE.format(rel_posix=rel_posix, rel_repr=rel_repr)
        out_path.write_text(content, encoding="utf-8")

    if clean and out_root.is_dir():
        if subdir is not None:
            clean_root = out_root / subdir
            if not clean_root.is_dir():
                clean_root = None
        else:
            clean_root = out_root

        if clean_root is not None:
            for path in sorted(clean_root.rglob("test_*.py")):
                if path not in expected_outputs:
                    path.unlink()
            # prune empty dirs (bottom-up) under clean_root only when scoped
            prune_root = clean_root if subdir is not None else out_root
            for d in sorted({p for p in prune_root.rglob("*") if p.is_dir()}, reverse=True):
                try:
                    d.rmdir()
                except OSError:
                    pass

    scope = f" (subdir={subdir!r})" if subdir else ""
    print(
        f"wrote {len(expected_outputs)} test file(s) under {out_root}{scope}",
        flush=True,
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="delete stale test_*.py with no matching srt source",
    )
    parser.add_argument(
        "--subdir",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "only generate tests for python/sglang/srt/PATH (e.g. layers). "
            "With --clean, only prune test_*.py under test/registered/unit/srt/PATH."
        ),
    )
    args = parser.parse_args(argv)
    generate(clean=args.clean, subdir=args.subdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
