# SPDX-License-Identifier: Apache-2.0
"""Unit tests for sgl_kernel.test_utils helpers (CPU-safe parts)."""

import unittest

from support import PackageExportMixin, load_sgl_source_module, torch_mod


class TestSglTestUtilsPackage(PackageExportMixin, unittest.TestCase):
    module_path = "sgl_kernel.test_utils"
    exported_names = (
        "create_per_token_group_quant_test_data",
        "assert_all_close_or_tiny_diff",
    )


class TestSglTestUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch_mod is None:
            raise unittest.SkipTest("torch is not installed")
        cls.tu = load_sgl_source_module("sgl_test_utils_ut", "test_utils.py")

    def test_compute_balanced_split(self):
        m = self.tu._compute_balanced_split(10, 3)
        self.assertEqual(m.tolist(), [4, 3, 3])
        self.assertEqual(int(m.sum()), 10)

    def test_compute_imbalanced_split_sums_to_total(self):
        gen = torch_mod.Generator(device="cpu")
        gen.manual_seed(0)
        arr = self.tu._compute_imbalanced_split(100, 8, gen_cpu=gen)
        self.assertEqual(sum(arr.tolist()), 100)
        self.assertEqual(len(arr), 8)


if __name__ == "__main__":
    unittest.main()
