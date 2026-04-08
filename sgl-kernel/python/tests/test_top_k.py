# SPDX-License-Identifier: Apache-2.0
"""Unit tests for sgl_kernel.top_k."""

import unittest

from support import PackageExportMixin, load_sgl_source_module, torch_mod


class TestTopKPackage(PackageExportMixin, unittest.TestCase):
    module_path = "sgl_kernel.top_k"
    exported_names = (
        "fast_topk",
        "fast_topk_v2",
        "fast_topk_transform_fused",
        "fast_topk_transform_ragged_fused",
    )


class TestTopK(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch_mod is None:
            raise unittest.SkipTest("torch is not installed")
        cls.tk = load_sgl_source_module("sgl_top_k_ut", "top_k.py")

    def test_fast_topk_k1_matches_torch_max(self):
        x = torch_mod.randn(4, 8)
        v1, i1 = self.tk.fast_topk(x, 1, dim=-1)
        v2, i2 = torch_mod.max(x, dim=-1, keepdim=True)
        self.assertTrue(torch_mod.equal(v1, v2))
        self.assertTrue(torch_mod.equal(i1, i2))

    def test_fast_topk_k2_matches_torch_topk(self):
        x = torch_mod.randn(3, 5)
        v1, i1 = self.tk.fast_topk(x, 2, dim=-1)
        v2, i2 = torch_mod.topk(x, 2, dim=-1)
        self.assertTrue(torch_mod.equal(v1, v2))
        self.assertTrue(torch_mod.equal(i1, i2))


if __name__ == "__main__":
    unittest.main()
