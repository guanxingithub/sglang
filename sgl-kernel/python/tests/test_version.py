# SPDX-License-Identifier: Apache-2.0
"""Unit tests for sgl_kernel.version."""

import unittest

from support import PackageExportMixin, load_sgl_source_module


class TestVersionPackage(PackageExportMixin, unittest.TestCase):
    module_path = "sgl_kernel.version"
    exported_names = ("__version__",)


class TestVersion(unittest.TestCase):
    def test_version_string(self):
        v = load_sgl_source_module("sgl_version_ut", "version.py")
        self.assertIsInstance(v.__version__, str)
        self.assertTrue(len(v.__version__) > 0)


if __name__ == "__main__":
    unittest.main()
