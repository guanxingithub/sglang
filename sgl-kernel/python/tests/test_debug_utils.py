# SPDX-License-Identifier: Apache-2.0
"""Unit tests for sgl_kernel.debug_utils."""

import os
import unittest
from unittest import mock

from support import PackageExportMixin, load_sgl_source_module


class TestDebugUtilsPackage(PackageExportMixin, unittest.TestCase):
    module_path = "sgl_kernel.debug_utils"
    exported_names = ("maybe_wrap_debug_kernel",)


class TestDebugUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.du = load_sgl_source_module("sgl_debug_utils_ut", "debug_utils.py")

    def test_maybe_wrap_returns_same_when_loglevel_zero(self):
        def f(x):
            return x + 1

        with mock.patch.dict(os.environ, {"SGLANG_KERNEL_API_LOGLEVEL": "0"}, clear=False):
            wrapped = self.du.maybe_wrap_debug_kernel(f, "op")
        self.assertIs(wrapped, f)
        self.assertEqual(wrapped(1), 2)


if __name__ == "__main__":
    unittest.main()
