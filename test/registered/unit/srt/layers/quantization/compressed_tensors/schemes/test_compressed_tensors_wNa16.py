# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0.

"""Auto-generated smoke tests for ``python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py``."""

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

RELATIVE_SRT_PATH = Path('layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py')


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
                f"{type(exc).__name__}: {exc}); "
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
                        f"{MODULE_NAME}: error resolving {sym}: "
                        f"{type(exc).__name__}: {exc}"
                    )


if __name__ == "__main__":
    unittest.main()
