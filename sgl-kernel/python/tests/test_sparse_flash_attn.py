# SPDX-License-Identifier: Apache-2.0
"""Unit tests for sgl_kernel.sparse_flash_attn."""

import unittest

from support import PackageExportMixin, load_sgl_source_module, torch_mod


class TestSparseFlashAttnPure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch_mod is None:
            raise unittest.SkipTest("torch is not installed")
        cls.sfa = load_sgl_source_module("sgl_sparse_flash_attn_ut", "sparse_flash_attn.py")

    def test_maybe_contiguous_none(self):
        self.assertIsNone(self.sfa.maybe_contiguous(None))

    def test_maybe_contiguous_already_row_major(self):
        x = torch_mod.randn(2, 4)
        y = self.sfa.maybe_contiguous(x)
        self.assertIs(y, x)

    def test_maybe_contiguous_non_contiguous_last_dim(self):
        x = torch_mod.randn(4, 6).t()
        self.assertNotEqual(x.stride(-1), 1)
        y = self.sfa.maybe_contiguous(x)
        self.assertTrue(y.is_contiguous())


class TestSparseFlashAttnSmoke(PackageExportMixin, unittest.TestCase):
    module_path = "sgl_kernel.sparse_flash_attn"
    exported_names = (
        "maybe_contiguous",
        "convert_vertical_slash_indexes",
        "convert_vertical_slash_indexes_mergehead",
        "sparse_attn_func",
        "sparse_attn_varlen_func",
    )


if __name__ == "__main__":
    unittest.main()
