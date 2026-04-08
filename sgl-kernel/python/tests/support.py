# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for sgl_kernel unit tests."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
PYTHON_DIR = TESTS_DIR.parent
SG_KERNEL_DIR = PYTHON_DIR / "sgl_kernel"

try:
    import torch as torch_mod
except ModuleNotFoundError:  # pragma: no cover - optional in minimal dev envs
    torch_mod = None


def load_sgl_source_module(as_name: str, relative_path: str):
    """
    Load a file under python/sgl_kernel/ as a top-level module without
    executing sgl_kernel/__init__.py (no common_ops required).
    """
    path = SG_KERNEL_DIR / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(as_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[as_name] = module
    spec.loader.exec_module(module)
    return module


def import_sgl_submodule(module_path: str):
    """Import sgl_kernel.* after package init (requires built extensions)."""
    return importlib.import_module(module_path)


class PackageExportMixin:
    """Mixin: import ``module_path`` and assert ``exported_names`` exist."""

    module_path: str = ""
    exported_names: tuple[str, ...] = ()

    @classmethod
    def setUpClass(cls):
        if torch_mod is None:
            raise unittest.SkipTest("torch is not installed")
        super().setUpClass()
        try:
            cls._mod = import_sgl_submodule(cls.module_path)
        except ImportError as e:
            raise unittest.SkipTest(str(e)) from e

    def test_exported_names_exist(self):
        if not self.exported_names:
            return
        missing = [n for n in self.exported_names if not hasattr(self._mod, n)]
        self.assertFalse(missing, f"Missing on {self.module_path}: {missing}")
